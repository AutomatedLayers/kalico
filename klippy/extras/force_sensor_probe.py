# force_sensor_probe.py   (Kalico / AutomatedLayers fork)
#
# Klippy extra: "scrapes" the bed by driving ONLY the two rear Z steppers
# (stepper_z2, stepper_z3) upward (negative Z) in a true probing move that
# halts at the MCU when a force sensor switch fires. The bed counts as fully
# scraped when the sensor triggers at/above the threshold OR the rear pair
# reaches the full scrape distance; an early (sub-threshold) trigger backs off
# and retries. After any outcome it re-homes Z.
#
# Written for the Kalico fork's (older, pre-motion_queuing) motion API:
#   - chelper trapq alloc; step gen + clock driven via toolhead._advance_move_time
#   - HomingMove built directly (Kalico manual_home() has no probe_pos arg)
#   - per-stepper stepper_enable.lookup_enable().motor_enable/disable
#   - clear_homing_state() takes axis INDICES ([2] for Z)
#
# Install: cp force_sensor_probe.py ~/kalico/klippy/extras/   (or ~/klipper/...)
#
# printer.cfg:
#   [force_sensor_probe]
#   pin: ^PA7                    # NC switch to GND: pullup, no invert
#   rear_steppers: stepper_z2, stepper_z3
#   front_steppers: stepper_z, stepper_z1
#   scrape_speed: 15.0           # mm/s, rear-pair travel speed (up & down)
#   scrape_accel: 100            # mm/s^2 for the isolated rear-only moves
#   scrape_threshold_z_pos: 390      # mm TOTAL travel; a trigger at/above this = fully scraped
#   scrape_distance: 400             # mm full scrape travel; reaching it (trigger or not) = fully scraped (must be > threshold)
#   retries: 5                       # sub-threshold attempts allowed; REFILLS whenever an attempt digs deeper
#   retry_tolerance: 1.0             # mm; a gain at/below this doesn't count as progress (won't refill retries)
#   max_total_attempts: 20           # absolute safety cap on total attempts (runaway guard)
#   delay_with_pressure: 1.0         # s to dwell while still loaded, BEFORE retracting
#   delay_between_retries: 60.0      # s to dwell AFTER retract, before the next attempt
#   retract_distance: 5.0            # mm to back off (descend) between retries
#   rehome_gcode: G28 Z
#   on_scraped_gcode:                # optional, runs after a successful scrape + home
#   on_failed_gcode:                 # optional (stuck below threshold); empty -> Mainsail prompt + PAUSE
#
# There is no fault/emergency-stop path: the only non-scraped outcome is "stuck"
# -- the sensor keeps triggering below the threshold until retries are
# exhausted -- which runs on_failed_gcode (or a Mainsail prompt + PAUSE).
#
# Usage: SCRAPE_BED_PROBE [DISTANCE=mm] [SPEED=mm/s] [RETRIES=n] [THRESHOLD=mm]
#        [RETRY_TOLERANCE=mm] [DELAY_WITH_PRESSURE=s] [DELAY_BETWEEN_RETRIES=s]
#        [MAX_ATTEMPTS=n]
#
# !! SAFETY !!  This drives motors into the bed and trusts the force sensor to
# stop them. Bench-test with a small scrape_distance and a hand on the power
# first. Both Z pairs are held energized during the scrape: only the rear pair
# moves, while the front pair holds position so those corners cannot drop.

import logging

from klippy import chelper
from . import force_move, homing


# Drip pacing, mirroring toolhead.py: feed the probing move in small segments
# so the host never queues far ahead of the MCU. Without this, a long move to
# the limit gets queued in full and everything after the trigger waits for that
# whole queue to drain.
DRIP_SEGMENT_TIME = 0.050
DRIP_TIME = 0.100

# How close to the commanded scrape distance the achieved travel must land to
# count as "reached the full distance with no trigger" (vs. an actual trigger
# that halted the move early). The full move always falls a little short of the
# target due to step-rounding -- a few microns in practice -- so this just has
# to be larger than that rounding and far smaller than any real early trigger.
FULL_MOVE_TOLERANCE = 0.1

# Fallback step-generation flush horizon if the toolhead doesn't expose
# kin_flush_delay (it normally does). See _move_start_time() for why this
# margin is required.
DEFAULT_KIN_FLUSH_DELAY = 0.250


# ===========================================================================
#  Minimal "toolhead + kinematics" that drives ONLY the rear Z steppers on a
#  private trapq. Passed to homing.HomingMove() as the toolhead so a probing
#  move halts at the MCU on the force-sensor trigger while moving just the
#  rear pair. Modeled on Kalico's extras/manual_stepper.py (ManualStepper),
#  with the trapq detach/restore trick from extras/force_move.py.
# ===========================================================================
class _RearZHomer:
    def __init__(self, printer, steppers, accel):
        self.printer = printer
        self.steppers = list(steppers)
        self.accel = accel
        self.commanded_pos = 0.0
        self.next_cmd_time = 0.0
        self._saved = []  # [(stepper, prev_sk, prev_trapq, our_sk), ...]
        ffi_main, ffi_lib = chelper.get_ffi()
        self.ffi_main = ffi_main
        self.ffi_lib = ffi_lib
        self.trapq = ffi_main.gc(ffi_lib.trapq_alloc(), ffi_lib.trapq_free)
        self.trapq_append = ffi_lib.trapq_append
        self.trapq_finalize_moves = ffi_lib.trapq_finalize_moves

    # -- detach the rear steppers from the toolhead Z trapq onto ours --
    def attach(self):
        toolhead = self.printer.lookup_object("toolhead")
        toolhead.flush_step_generation()
        self._saved = []
        for s in self.steppers:
            sk = self.ffi_main.gc(
                self.ffi_lib.cartesian_stepper_alloc(b"x"), self.ffi_lib.free
            )
            prev_sk = s.set_stepper_kinematics(sk)
            prev_trapq = s.set_trapq(self.trapq)
            s.set_position((0.0, 0.0, 0.0))
            self._saved.append((s, prev_sk, prev_trapq, sk))
        self.commanded_pos = 0.0
        self.next_cmd_time = toolhead.get_last_move_time()

    # -- restore the rear steppers back onto the toolhead Z trapq --
    def detach(self):
        self.sync_print_time()
        for s, prev_sk, prev_trapq, _sk in self._saved:
            s.set_trapq(prev_trapq)
            s.set_stepper_kinematics(prev_sk)
        self._saved = []

    def sync_print_time(self):
        toolhead = self.printer.lookup_object("toolhead")
        print_time = toolhead.get_last_move_time()
        if self.next_cmd_time > print_time:
            toolhead.dwell(self.next_cmd_time - print_time)
        else:
            self.next_cmd_time = print_time

    # The earliest time a new move appended to our private trapq may start.
    #
    # Our rear steppers stay registered as TOOLHEAD step generators even while
    # detached onto this private trapq, so the toolhead's background flushing
    # keeps advancing their step-generator clock. After a toolhead.dwell()
    # (e.g. the inter-attempt delays) the dwell is queued ahead and, as it
    # drains, the generators get advanced up to ~print_time + kin_flush_delay
    # while our trapq is empty. A new move must therefore begin past that
    # horizon; starting merely at print_time appends a move behind the
    # generators' committed time, which makes stepcompress emit non-monotonic
    # steps ("stepcompress ... Invalid sequence" -> Internal error).
    #
    # For back-to-back moves next_cmd_time already leads print_time, so the
    # max() keeps the margin a no-op except right after a dwell -- exactly the
    # case that needs it.
    def _move_start_time(self, toolhead):
        flush_margin = getattr(
            toolhead, "kin_flush_delay", DEFAULT_KIN_FLUSH_DELAY
        )
        return max(self.next_cmd_time, toolhead.print_time + flush_margin)

    # plain (non-homing) isolated move, e.g. retract. Drives ONLY the rear
    # steppers (they are on our private trapq) but advances the real toolhead
    # clock + flushes its step generators via _advance_move_time, so the two
    # clocks stay aligned and nothing schedules far in the future.
    def move(self, movepos, speed):
        toolhead = self.printer.lookup_object("toolhead")
        self.sync_print_time()
        cp = self.commanded_pos
        dist = movepos - cp
        axis_r, accel_t, cruise_t, cruise_v = force_move.calc_move_time(
            dist, speed, self.accel
        )
        start = self._move_start_time(toolhead)
        self.trapq_append(
            self.trapq, start, accel_t, cruise_t, accel_t,
            cp, 0.0, 0.0, axis_r, 0.0, 0.0, 0.0, cruise_v, self.accel,
        )
        end = start + accel_t + cruise_t + accel_t
        # Newer toolhead API: report queued-move activity so step_gen_time /
        # need_flush_time track our private-trapq steps. Without this the
        # background flush handler desyncs our rear steppers' step generation
        # (notably across a long dwell), corrupting the next move.
        toolhead.note_mcu_movequeue_activity(
            end + toolhead.kin_flush_delay, set_step_gen_time=True
        )
        toolhead._advance_move_time(end)
        self.trapq_finalize_moves(self.trapq, end + 99999.9, end + 99999.9)
        self.commanded_pos = movepos
        self.next_cmd_time = toolhead.print_time

    # ---- toolhead/kinematics interface used by HomingMove ----
    def flush_step_generation(self):
        self.sync_print_time()

    def get_kinematics(self):
        return self

    def get_steppers(self):
        return self.steppers

    def calc_position(self, stepper_positions):
        # Report the first rear stepper's position as the "Z" travel value.
        return [stepper_positions[self.steppers[0].get_name()], 0.0, 0.0]

    def get_position(self):
        return [self.commanded_pos, 0.0, 0.0, 0.0]

    def set_position(self, newpos, homing_axes=()):
        for s in self.steppers:
            s.set_position((newpos[0], 0.0, 0.0))
        self.commanded_pos = newpos[0]

    def get_last_move_time(self):
        self.sync_print_time()
        return self.next_cmd_time

    def dwell(self, delay):
        self.next_cmd_time += max(0.0, delay)

    def drip_move(self, newpos, speed, drip_completion):
        # Paced, truncating drip (mirrors toolhead._update_drip_move_time).
        # Feed the move to the limit in 50ms segments; stop the instant the
        # endstop fires. _advance_move_time advances the real toolhead clock
        # AND flushes step generators -- z2/z3 read our private trapq, so only
        # they move; z/z1 read the (empty) toolhead trapq and stay put.
        toolhead = self.printer.lookup_object("toolhead")
        reactor = self.printer.get_reactor()
        mcu = toolhead.mcu
        flush_delay = DRIP_TIME + toolhead.kin_flush_delay
        cp = self.commanded_pos
        target = newpos[0]
        dist = target - cp
        axis_r, accel_t, cruise_t, cruise_v = force_move.calc_move_time(
            dist, speed, self.accel
        )
        start = self._move_start_time(toolhead)
        end = start + accel_t + cruise_t + accel_t
        self.trapq_append(
            self.trapq, start, accel_t, cruise_t, accel_t,
            cp, 0.0, 0.0, axis_r, 0.0, 0.0, 0.0, cruise_v, self.accel,
        )
        pt = start
        while pt < end:
            if drip_completion.test():
                break
            curtime = reactor.monotonic()
            est = mcu.estimated_print_time(curtime)
            wait_time = pt - est - flush_delay
            if wait_time > 0.0:
                # Don't run ahead of the MCU; wake early if the endstop fires.
                drip_completion.wait(curtime + wait_time)
                continue
            pt = min(pt + DRIP_SEGMENT_TIME, end)
            # Mirror toolhead._update_drip_move_time: report activity (advancing
            # step_gen_time / need_flush_time) BEFORE advancing the move time, so
            # the toolhead's flush bookkeeping matches the steps we just queued.
            toolhead.note_mcu_movequeue_activity(
                pt + toolhead.kin_flush_delay, set_step_gen_time=True
            )
            toolhead._advance_move_time(pt)
        # Triggered or full move: free the (possibly un-traveled) remainder and
        # leave our clock at the real stop, not the far end of the move.
        self.trapq_finalize_moves(self.trapq, pt + 99999.9, pt + 99999.9)
        self.commanded_pos = target
        self.next_cmd_time = toolhead.print_time


class ForceSensorProbe:
    def __init__(self, config):
        self.printer = config.get_printer()
        self.name = config.get_name()

        self.scrape_speed = config.getfloat("scrape_speed", 15.0, above=0.0)
        self.scrape_accel = config.getfloat("scrape_accel", 100.0, above=0.0)
        # Full scrape distance (a RELATIVE travel, not a Z position). Reaching
        # this travel counts as a fully scraped bed all by itself (no trigger
        # required) -- it's one of the two ways to succeed.
        self.scrape_distance = config.getfloat(
            "scrape_distance", 395.0, above=0.0
        )
        # A trigger at/above this travel also counts as fully scraped. Sitting
        # below the full distance, it's a safety margin: an early-but-sufficient
        # trigger still ends the scrape cleanly.
        self.scrape_threshold = config.getfloat(
            "scrape_threshold_z_pos", 390.0, above=0.0
        )
        if self.scrape_threshold >= self.scrape_distance:
            logging.warning(
                "force_sensor_probe: scrape_threshold_z_pos (%.2f) should be "
                "less than scrape_distance (%.2f); otherwise the trigger "
                "success window collapses", self.scrape_threshold,
                self.scrape_distance)
        self.retries = config.getint("retries", 5, minval=0)
        # Minimum extra travel (mm) an attempt must gain over the deepest point
        # so far to count as real progress. Gains at/below this are treated as
        # non-progress and burn a retry.
        self.retry_tolerance = config.getfloat(
            "retry_tolerance", 1.0, minval=0.0
        )
        self.max_total_attempts = config.getint(
            "max_total_attempts", 20, minval=1
        )
        # Two dwell periods in the retry cycle.
        self.delay_with_pressure = config.getfloat(
            "delay_with_pressure", 1.0, minval=0.0
        )
        self.delay_between_retries = config.getfloat(
            "delay_between_retries", 1.0, minval=0.0
        )
        self.retract_distance = config.getfloat(
            "retract_distance", 5.0, minval=0.0
        )
        self.rehome_gcode = config.get("rehome_gcode", "G28 Z")
        self.on_scraped_gcode = config.get("on_scraped_gcode", "")
        self.on_failed_gcode = config.get("on_failed_gcode", "")

        self.rear_names = [
            s.strip()
            for s in config.get("rear_steppers", "stepper_z2, stepper_z3").split(",")
        ]
        self.front_names = [
            s.strip()
            for s in config.get("front_steppers", "stepper_z, stepper_z1").split(",")
        ]

        # Force sensor pin -> endstop. Both rear steppers are added so they are
        # watched together and halt together at the trigger.
        ppins = self.printer.lookup_object("pins")
        self.mcu_endstop = ppins.setup_pin("endstop", config.get("pin"))

        # Register the endstop with query_endstops so it reports through
        # QUERY_ENDSTOPS -- which is what Mainsail's Endstops panel reads. The
        # display name defaults to the section name; override with endstop_name.
        endstop_name = config.get("endstop_name", self.name)
        query_endstops = self.printer.load_object(config, "query_endstops")
        query_endstops.register_endstop(self.mcu_endstop, endstop_name)

        self.rear_steppers = []
        self.printer.register_event_handler(
            "klippy:mcu_identify", self._handle_mcu_identify
        )

        gcode = self.printer.lookup_object("gcode")
        gcode.register_command(
            "SCRAPE_BED_PROBE",
            self.cmd_SCRAPE_BED_PROBE,
            desc=self.cmd_SCRAPE_BED_PROBE_help,
        )

    def _handle_mcu_identify(self):
        kin = self.printer.lookup_object("toolhead").get_kinematics()
        by_name = {s.get_name(): s for s in kin.get_steppers()}
        for name in self.rear_names:
            stepper = by_name.get(name)
            if stepper is None:
                raise self.printer.config_error(
                    "force_sensor_probe: rear stepper '%s' not found in Z "
                    "kinematics" % (name,)
                )
            self.mcu_endstop.add_stepper(stepper)
            self.rear_steppers.append(stepper)
            logging.info("force_sensor_probe: registered %s with endstop", name)

    # ------------------------------------------------------------------ #
    def _set_enable(self, names, enable):
        toolhead = self.printer.lookup_object("toolhead")
        stepper_enable = self.printer.lookup_object("stepper_enable")
        print_time = toolhead.get_last_move_time()
        changed = False
        for name in names:
            en = stepper_enable.lookup_enable(name)
            if enable and not en.is_motor_enabled():
                en.motor_enable(print_time)
                changed = True
            elif not enable and en.is_motor_enabled():
                en.motor_disable(print_time)
                changed = True
        if changed:
            toolhead.dwell(0.100)

    def _emit_failure_prompt(self, gcode):
        # Mainsail/Fluidd interactive prompt + PAUSE. Requires [respond] and
        # [pause_resume] in printer.cfg.
        for line in [
            'RESPOND TYPE=error MSG="Bed could not be fully scraped"',
            'RESPOND TYPE=command MSG="action:prompt_begin Scrape failed"',
            'RESPOND TYPE=command MSG="action:prompt_text The bed could not be'
            ' fully scraped. Resume to continue or cancel the print."',
            'RESPOND TYPE=command MSG="action:prompt_footer_button'
            ' Resume|RESUME|primary"',
            'RESPOND TYPE=command MSG="action:prompt_footer_button'
            ' Cancel|CANCEL_PRINT|error"',
            'RESPOND TYPE=command MSG="action:prompt_show"',
            "PAUSE",
        ]:
            gcode.run_script_from_command(line)

    cmd_SCRAPE_BED_PROBE_help = (
        "Drive the rear Z steppers up to scrape the bed. Fully scraped when the "
        "force sensor triggers at/above the threshold OR the rear pair reaches "
        "the full scrape distance; a trigger below threshold backs off and "
        "retries."
    )

    def cmd_SCRAPE_BED_PROBE(self, gcmd):
        distance = gcmd.get_float("DISTANCE", self.scrape_distance, above=0.0)
        speed = gcmd.get_float("SPEED", self.scrape_speed, above=0.0)
        retries = gcmd.get_int("RETRIES", self.retries, minval=0)
        delay_pressure = gcmd.get_float(
            "DELAY_WITH_PRESSURE", self.delay_with_pressure, minval=0.0
        )
        delay_between = gcmd.get_float(
            "DELAY_BETWEEN_RETRIES", self.delay_between_retries, minval=0.0
        )
        threshold = gcmd.get_float("THRESHOLD", self.scrape_threshold, above=0.0)
        tolerance = gcmd.get_float(
            "RETRY_TOLERANCE", self.retry_tolerance, minval=0.0
        )
        hard_cap = gcmd.get_int(
            "MAX_ATTEMPTS", self.max_total_attempts, minval=1
        )

        gcode = self.printer.lookup_object("gcode")
        toolhead = self.printer.lookup_object("toolhead")

        # Enable BOTH pairs. The rear pair is what moves; the front pair is
        # held energized so those corners can't drop and so the bed pivots
        # about a rigid front. The scrape only ever feeds steps to the rear
        # pair (their own private trapq), so the front stays put -- enabled but
        # not moving -- throughout.
        self._set_enable(self.rear_names, True)
        self._set_enable(self.front_names, True)

        homer = _RearZHomer(self.printer, self.rear_steppers, self.scrape_accel)
        scraped = False

        try:
            homer.attach()
            # Replenishing-retries loop.
            #   best       = deepest TOTAL travel (from start) reached so far
            #   tries_left = non-improving attempts remaining before giving up;
            #                refilled to `retries` whenever an attempt beats
            #                `best`. So progress earns a fresh budget.
            #   hard_cap   = absolute iteration guard against runaway tiny gains.
            # Position is zeroed once in attach(), never re-zeroed, so trigpos
            # is always measured from the start of the whole command.
            best = 0.0
            tries_left = retries
            attempt_num = 0
            while True:
                attempt_num += 1
                if attempt_num > hard_cap:
                    gcmd.respond_info(
                        "SCRAPE_BED_PROBE: hit safety cap of %d attempts; "
                        "aborting" % (hard_cap,)
                    )
                    break
                gcmd.respond_info(
                    "SCRAPE_BED_PROBE: attempt %d (best %.3f mm, %d non-progress "
                    "tr%s left)"
                    % (attempt_num, best, tries_left,
                       "y" if tries_left == 1 else "ies")
                )

                # Up is negative Z; drive toward the full scrape distance
                # (-distance) from the start. Build HomingMove directly so we
                # get the real trigger position (Kalico manual_home has no
                # probe_pos arg). check_triggered=False is deliberate: reaching
                # the full distance with no trigger is a valid success here, not
                # an error, so we don't want homing_move to raise.
                hmove = homing.HomingMove(
                    self.printer,
                    [(self.mcu_endstop, "force_sensor")],
                    toolhead=homer,
                )
                trigpos = hmove.homing_move(
                    [-distance, 0.0, 0.0, 0.0], speed,
                    probe_pos=True, triggered=True, check_triggered=False,
                )
                total_travel = abs(trigpos[0])
                already = hmove.check_no_movement() is not None
                reached_full = (not already) and \
                    total_travel >= distance - FULL_MOVE_TOLERANCE

                if already:
                    # Sensor was active before the move (didn't clear on the last
                    # retract). Non-progress attempt.
                    gcmd.respond_info(
                        "SCRAPE_BED_PROBE: sensor already triggered at start of "
                        "attempt %d" % (attempt_num,)
                    )
                elif reached_full:
                    gcmd.respond_info(
                        "SCRAPE_BED_PROBE: completed full scrape distance "
                        "(%.1f mm) with no trigger" % (distance,)
                    )
                else:
                    gcmd.respond_info(
                        "SCRAPE_BED_PROBE: triggered at %.3f mm total travel "
                        "from start" % (total_travel,)
                    )

                # Two ways to be fully scraped: a trigger at/above the threshold,
                # OR reaching the full scrape distance. Both satisfy
                # total_travel >= threshold (the distance is above the
                # threshold), so a single check covers them.
                if not already and total_travel >= threshold:
                    scraped = True
                    break

                # Below threshold (an early trigger / jam): back off and retry,
                # digging deeper each time.
                if already:
                    tries_left -= 1
                elif total_travel > best + tolerance:
                    # Gained more than the tolerance past the previous deepest
                    # point: real progress, so refill the budget and keep
                    # digging.
                    best = total_travel
                    tries_left = retries
                    gcmd.respond_info(
                        "SCRAPE_BED_PROBE: progressed to %.3f mm (< %.3f); "
                        "attempts replenished to %d"
                        % (total_travel, threshold, retries)
                    )
                else:
                    tries_left -= 1
                    gcmd.respond_info(
                        "SCRAPE_BED_PROBE: no real progress (%.3f <= best %.3f "
                        "+ tol %.3f); %d attempt(s) left"
                        % (total_travel, best, tolerance, tries_left)
                    )

                if tries_left <= 0:
                    gcmd.respond_info(
                        "SCRAPE_BED_PROBE: no further progress past %.3f mm "
                        "after %d attempt(s); giving up" % (best, retries)
                    )
                    break

                # Still loaded against the part. Dwell under pressure, back off,
                # then settle before the next attempt.
                if delay_pressure > 0.0:
                    gcmd.respond_info(
                        "SCRAPE_BED_PROBE: dwelling %.1fs under pressure"
                        % (delay_pressure,)
                    )
                    toolhead.dwell(delay_pressure)
                    homer.sync_print_time()
                gcmd.respond_info(
                    "SCRAPE_BED_PROBE: retracting %.2fmm" % (self.retract_distance,)
                )
                homer.move(homer.commanded_pos + self.retract_distance, speed)
                if delay_between > 0.0:
                    gcmd.respond_info(
                        "SCRAPE_BED_PROBE: dwelling %.1fs before next attempt"
                        % (delay_between,)
                    )
                    toolhead.dwell(delay_between)
                    homer.sync_print_time()

            # No repositioning here -- the G28 Z below re-establishes Z from
            # wherever the rear pair ended up.
        finally:
            homer.detach()

        # The bed is now tilted / kinematic Z is stale: mark Z un-homed. The
        # front pair stayed energized throughout, so nothing to re-enable here.
        toolhead.get_kinematics().clear_homing_state([2])

        # Normal outcomes re-home Z (Steps 4/6 end).
        if self.rehome_gcode:
            gcode.run_script_from_command(self.rehome_gcode)

        if scraped:
            gcmd.respond_info("SCRAPE_BED_PROBE: bed fully scraped")
            if self.on_scraped_gcode:
                gcode.run_script_from_command(self.on_scraped_gcode)
        else:
            gcmd.respond_info(
                "SCRAPE_BED_PROBE: FAILED -- stuck below threshold after "
                "exhausting retries (best %.3f mm)" % (best,)
            )
            if self.on_failed_gcode:
                gcode.run_script_from_command(self.on_failed_gcode)
            else:
                self._emit_failure_prompt(gcode)


def load_config(config):
    return ForceSensorProbe(config)