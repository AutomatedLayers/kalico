# vertigo_z_homing.py   (Kalico / AutomatedLayers fork)
#
# Two-stage Z homing + per-pair calibration jogs for the Vertigo printer, which
# has FOUR independent Z steppers and a flexible front bar with a lever that
# couples/decouples the front pair from the bed:
#
#   stepper_z, stepper_z1  -> FRONT pair (flexible bar + lever)
#   stepper_z2, stepper_z3 -> REAR pair
#
# The Z endstops are at the BOTTOM of travel. At the bottom the lever is
# DISENGAGED, which frees the rear pair to lift and tilt the bed for scraping.
# Raising the bed is NEGATIVE Z on this machine -- that's fixed in hardware, so
# it's hardcoded here (UP_SIGN), not a config option.
#
# Commands:
#
#   Z_HOME_FOR_SCRAPE
#       Home all four Z to the bottom endstops and stop. Lever disengaged ->
#       ready for SCRAPE_BED_PROBE (rear pair lifts to scrape).
#
#   Z_HOME_FOR_PRINT [FRONT=mm] [REAR=mm]
#       Home all four Z to the bottom, then raise ONLY the front pair to engage
#       the lever, then raise ONLY the rear pair to couple the bed. Leaves a
#       defined kinematic Z ("bed ready to print").
#
#   FORCE_MOVE_Z_FRONT DISTANCE=mm [SPEED=mm/s]
#   FORCE_MOVE_Z_REAR  DISTANCE=mm [SPEED=mm/s]
#       Calibration jog: move ONLY the named pair while the other pair stays
#       energized and holds. DISTANCE is signed, + = up (bed rises). Afterward
#       Z is marked UNHOMED, so the next G28 Z re-homes all four steppers.
#
# Independent pair moves (the raises and the force-jogs) are done by detaching
# the moving pair onto a private trapq and moving them together -- the same
# trick force_sensor_probe.py uses for the scrape. The other pair stays on the
# idle toolhead Z trapq: it generates no steps and HOLDS position while
# energized. No enable-pin toggling.
#
# Requires real enable_pins on the four Z steppers (so Klipper's M84 / idle
# disable them and homing energizes them). This module energizes all four via
# stepper_enable before a force-jog (which doesn't go through homing).
#
# Install: cp vertigo_z_homing.py ~/kalico/klippy/extras/   (then restart
# klipper -- a host RESTART/FIRMWARE_RESTART does not reload .py files).
#
# printer.cfg:
#   [vertigo_z_homing]
#   front_steppers: stepper_z, stepper_z1
#   rear_steppers: stepper_z2, stepper_z3
#   front_engage_distance: 25      # mm to raise the FRONT pair to engage the lever
#   rear_couple_distance: 13       # mm to raise the REAR pair to couple the bed
#   lift_speed: 25                 # mm/s for the isolated front/rear moves (raises + jogs)
#   lift_accel: 100                # mm/s^2 for the isolated front/rear moves
#   #max_print_z:                  # optional explicit kinematic Z after print-home;
#                                  # default = post-home Z - front - rear

import logging

from klippy import chelper
from . import force_move


# Raising the bed = NEGATIVE Z on this machine. Fixed in hardware.
UP_SIGN = -1.0

# All four Z home to the bottom endstops with a normal Z home.
# Internal "home all four Z to the bottom" command. This is intentionally NOT a
# bare "G28 Z": the printer.cfg homing_override rewrites a user G28 Z into a full
# print-ready home (engage lever + couple bed). The module must reach ONLY the
# bottom endstops, so it calls a raw-home macro that flags the override to skip
# the print-ready raises. See [gcode_macro _HOME_Z_RAW] in macros.cfg.
HOME_Z_GCODE = "_HOME_Z_RAW"

# Lead time (s) so the move's first step lands safely ahead of the live MCU
# clock. toolhead.print_time can lag the real clock after an idle period, so
# we must NOT trust it alone as the move start -- doing so schedules steps in
# the past and trips "Stepper too far in past".
MOVE_START_BUFFER = 0.250


# ===========================================================================
#  Move a SUBSET of the Z steppers together on a private trapq, leaving the
#  rest of the Z steppers energized and holding position. (Same trapq-detach
#  trick as force_sensor_probe.py.) One attach/move/detach per instance.
# ===========================================================================
class _SubsetZMover:
    def __init__(self, printer, steppers, accel):
        self.printer = printer
        self.steppers = list(steppers)
        self.accel = accel
        self._saved = []  # [(stepper, prev_sk, prev_trapq, our_sk), ...]
        ffi_main, ffi_lib = chelper.get_ffi()
        self.ffi_main = ffi_main
        self.ffi_lib = ffi_lib
        self.trapq = ffi_main.gc(ffi_lib.trapq_alloc(), ffi_lib.trapq_free)
        self.trapq_append = ffi_lib.trapq_append
        self.trapq_finalize_moves = ffi_lib.trapq_finalize_moves
        self._move_end = 0.0
        self._move_duration = 0.0

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

    def detach(self):
        toolhead = self.printer.lookup_object("toolhead")
        for s, prev_sk, prev_trapq, _sk in self._saved:
            s.set_trapq(prev_trapq)
            s.set_stepper_kinematics(prev_sk)
        self._saved = []
        # Hand the move back to the toolhead's normal flush path. This is the
        # part force_move.manual_move does that the old code skipped: register
        # the scheduled MCU activity, dwell for the move's duration so the
        # toolhead clock advances past it, then flush. Without this the steps
        # sit in the host step-compress buffer with their original (now stale)
        # clocks and only get transmitted on the NEXT activity -- by which time
        # the MCU clock has moved on, giving "Stepper too far in past" (or, if
        # the gap is short, "Timer too close").
        toolhead.note_mcu_movequeue_activity(self._move_end)
        toolhead.dwell(self._move_duration)
        toolhead.flush_step_generation()

    def move(self, dist, speed):
        toolhead = self.printer.lookup_object("toolhead")
        reactor = self.printer.get_reactor()
        axis_r, accel_t, cruise_t, cruise_v = force_move.calc_move_time(
            dist, speed, self.accel
        )
        # Start ahead of BOTH the toolhead's print_time and the live MCU clock.
        # The MCU-clock floor is what keeps a move issued after idle from being
        # scheduled in the past.
        last_move = toolhead.get_last_move_time()
        est = toolhead.mcu.estimated_print_time(reactor.monotonic())
        start = max(last_move, est + MOVE_START_BUFFER)
        self.trapq_append(
            self.trapq, start, accel_t, cruise_t, accel_t,
            0.0, 0.0, 0.0, axis_r, 0.0, 0.0, 0.0, cruise_v, self.accel,
        )
        end = start + accel_t + cruise_t + accel_t
        self._move_end = end
        self._move_duration = accel_t + cruise_t + accel_t
        # Generate the steps for THIS subset off the private trapq while the
        # steppers are still attached to it. (detach() then restores them and
        # drives the toolhead flush.)
        for s in self.steppers:
            s.generate_steps(end)
        self.trapq_finalize_moves(self.trapq, end + 99999.9, end + 99999.9)


class VertigoZHoming:
    def __init__(self, config):
        self.printer = config.get_printer()

        self.front_names = [
            s.strip()
            for s in config.get("front_steppers", "stepper_z, stepper_z1").split(",")
        ]
        self.rear_names = [
            s.strip()
            for s in config.get("rear_steppers", "stepper_z2, stepper_z3").split(",")
        ]
        self.front_engage = config.getfloat("front_engage_distance", above=0.0)
        self.rear_couple = config.getfloat("rear_couple_distance", above=0.0)
        self.lift_speed = config.getfloat("lift_speed", 25.0, above=0.0)
        self.lift_accel = config.getfloat("lift_accel", 100.0, above=0.0)
        # Optional explicit kinematic Z after the print-home. Default (None)
        # reproduces the old READY_BED end state (bottom shifted up by both raises).
        self.max_print_z = config.getfloat("max_print_z", None)

        self.front_steppers = []
        self.rear_steppers = []

        # Signed net displacement of each pair from "nominal", in user
        # up-positive mm: + = up, - = down. "Nominal" is wherever the pair was
        # resting the last time the motors were disabled (M84 / M18 / idle
        # timeout) or the Z steppers were homed -- both reset this to 0. Only
        # the FORCE_MOVE_Z_FRONT/REAR jogs accumulate here; the Z_HOME_FOR_PRINT
        # engage/couple raises do not. A move made by hand while the motors are
        # off is invisible to the host, so after a disable the next jog tracks
        # from wherever the axis physically rests.
        self.offsets = {"front": 0.0, "rear": 0.0}

        # True only once Z_HOME_FOR_PRINT has fully completed: homed, lever
        # engaged (front raised), and bed coupled (rear raised). Cleared by
        # anything that returns Z to an uncoupled/unhomed state -- the scrape
        # home, a plain G28 Z, a force-move jog (which unhomes Z), or a motor
        # disable. Exposed via get_status as printer.vertigo_z_homing.bed_coupled.
        self.bed_coupled = False

        self.printer.register_event_handler(
            "klippy:mcu_identify", self._handle_mcu_identify
        )
        self.printer.register_event_handler(
            "stepper_enable:motor_off", self._handle_motor_off
        )
        self.printer.register_event_handler(
            "homing:home_rails_end", self._handle_home_rails_end
        )

        gcode = self.printer.lookup_object("gcode")
        gcode.register_command(
            "Z_HOME_FOR_SCRAPE", self.cmd_Z_HOME_FOR_SCRAPE,
            desc=self.cmd_Z_HOME_FOR_SCRAPE_help,
        )
        gcode.register_command(
            "Z_HOME_FOR_PRINT", self.cmd_Z_HOME_FOR_PRINT,
            desc=self.cmd_Z_HOME_FOR_PRINT_help,
        )
        gcode.register_command(
            "FORCE_MOVE_Z_FRONT", self.cmd_FORCE_MOVE_Z_FRONT,
            desc=self.cmd_FORCE_MOVE_Z_FRONT_help,
        )
        gcode.register_command(
            "FORCE_MOVE_Z_REAR", self.cmd_FORCE_MOVE_Z_REAR,
            desc=self.cmd_FORCE_MOVE_Z_REAR_help,
        )

    def _handle_mcu_identify(self):
        kin = self.printer.lookup_object("toolhead").get_kinematics()
        by_name = {s.get_name(): s for s in kin.get_steppers()}
        for names, bucket in ((self.front_names, self.front_steppers),
                              (self.rear_names, self.rear_steppers)):
            for stepper_name in names:
                stepper = by_name.get(stepper_name)
                if stepper is None:
                    raise self.printer.config_error(
                        "vertigo_z_homing: stepper '%s' not found in Z "
                        "kinematics" % (stepper_name,)
                    )
                bucket.append(stepper)

    def get_status(self, eventtime):
        return {
            "bed_coupled": self.bed_coupled,
            "front_offset": self.offsets["front"],
            "rear_offset": self.offsets["rear"],
        }

    def _reset_offsets(self):
        self.offsets["front"] = 0.0
        self.offsets["rear"] = 0.0

    def _handle_motor_off(self, print_time):
        # M84 / M18 / idle-timeout disabled the steppers -> back to nominal,
        # Z unhomed, bed no longer coupled.
        self._reset_offsets()
        self.bed_coupled = False

    def _handle_home_rails_end(self, homing_state, rails):
        # Reset when any tracked Z stepper gets homed (G28 Z, full G28, or the
        # force_sensor_probe rehome). Homing lands the pairs at the bottom with
        # the lever disengaged, so the bed is not coupled.
        tracked = set(self.front_names) | set(self.rear_names)
        for rail in rails:
            for s in rail.get_steppers():
                if s.get_name() in tracked:
                    self._reset_offsets()
                    self.bed_coupled = False
                    return

    # ------------------------------------------------------------------ #
    def _run(self, script):
        self.printer.lookup_object("gcode").run_script_from_command(script)

    def _energize_all(self):
        # Ensure all four Z motors are energized so the non-moving pair holds.
        # Needed for the force-jogs, which don't go through homing (homing
        # energizes the steppers on its own).
        toolhead = self.printer.lookup_object("toolhead")
        stepper_enable = self.printer.lookup_object("stepper_enable")
        pt = toolhead.get_last_move_time()
        changed = False
        for s in self.front_steppers + self.rear_steppers:
            en = stepper_enable.lookup_enable(s.get_name())
            if not en.is_motor_enabled():
                en.motor_enable(pt)
                changed = True
        if changed:
            toolhead.dwell(0.100)

    def _raise_subset(self, steppers, up_distance, speed):
        # up_distance is signed in user up-positive mm (+ = up); UP_SIGN turns
        # it into the kinematic direction.
        mover = _SubsetZMover(self.printer, steppers, self.lift_accel)
        mover.attach()
        try:
            mover.move(UP_SIGN * up_distance, speed)
        finally:
            mover.detach()

    # ---- Homing for the scrape pose ---------------------------------- #
    cmd_Z_HOME_FOR_SCRAPE_help = (
        "Home all four Z to the bottom endstops (lever disengaged); ready for "
        "SCRAPE_BED_PROBE."
    )

    def cmd_Z_HOME_FOR_SCRAPE(self, gcmd):
        self._run(HOME_Z_GCODE)
        self.bed_coupled = False  # lever disengaged at the bottom
        gcmd.respond_info(
            "Z_HOME_FOR_SCRAPE: homed to bottom; lever disengaged, ready to scrape"
        )

    # ---- Homing for the print pose ----------------------------------- #
    cmd_Z_HOME_FOR_PRINT_help = (
        "Home all four Z to the bottom, raise the front pair to engage the "
        "lever, then raise the rear pair to couple the bed (ready to print)."
    )

    def cmd_Z_HOME_FOR_PRINT(self, gcmd):
        front_d = gcmd.get_float("FRONT", self.front_engage, above=0.0)
        rear_d = gcmd.get_float("REAR", self.rear_couple, above=0.0)
        toolhead = self.printer.lookup_object("toolhead")

        # 1) Home all four Z to the bottom endstops (homing energizes them).
        self._run(HOME_Z_GCODE)
        toolhead.wait_moves()
        base = toolhead.get_position()  # Z here is the bottom (position_endstop)

        # 2) Raise the FRONT pair to engage the lever (rear holds, energized).
        gcmd.respond_info(
            "Z_HOME_FOR_PRINT: engaging lever (front +%.2f mm)" % (front_d,)
        )
        self._raise_subset(self.front_steppers, front_d, self.lift_speed)

        # 3) Raise the REAR pair to couple the bed (front holds, energized).
        gcmd.respond_info(
            "Z_HOME_FOR_PRINT: coupling bed (rear +%.2f mm)" % (rear_d,)
        )
        self._raise_subset(self.rear_steppers, rear_d, self.lift_speed)

        # 4) Record the ready-to-print kinematic Z and keep Z homed (the private
        #    trapq moves did not touch the toolhead's Z). X/Y are untouched.
        toolhead.flush_step_generation()
        if self.max_print_z is not None:
            ready_z = self.max_print_z
        else:
            ready_z = base[2] + UP_SIGN * (front_d + rear_d)
        cur = toolhead.get_position()
        toolhead.set_position(
            [cur[0], cur[1], ready_z, cur[3]], homing_axes=(2,)
        )
        # Both pairs have made their upward moves: lever engaged, bed coupled.
        self.bed_coupled = True
        gcmd.respond_info(
            "Z_HOME_FOR_PRINT: bed ready to print (Z = %.3f)" % (ready_z,)
        )

    # ---- Calibration jogs (per pair) --------------------------------- #
    def _force_move(self, steppers, label, pair, gcmd):
        distance = gcmd.get_float("DISTANCE")          # signed; + = up
        speed = gcmd.get_float("SPEED", self.lift_speed, above=0.0)

        # All four energized: the named pair moves, the other pair holds.
        self._energize_all()
        self._raise_subset(steppers, distance, speed)  # UP_SIGN applied inside
        self.offsets[pair] += distance                 # track jog from nominal

        # An independent pair move makes the kinematic Z meaningless, so force a
        # full re-home. The next G28 Z re-homes ALL four steppers regardless.
        self.printer.lookup_object("toolhead").get_kinematics().clear_homing_state([2])
        self.bed_coupled = False  # Z is now unhomed; bed no longer coupled
        gcmd.respond_info(
            "%s: moved %.3f mm (+ = up); %s pair now %+.3f mm from nominal "
            "(since last G28 Z / M84). Z is now UNHOMED -- run G28 Z before "
            "printing." % (label, distance, pair, self.offsets[pair])
        )

    cmd_FORCE_MOVE_Z_FRONT_help = (
        "Jog ONLY the front Z pair for calibration (rear holds). DISTANCE mm, "
        "+ = up. Leaves Z unhomed."
    )

    def cmd_FORCE_MOVE_Z_FRONT(self, gcmd):
        self._force_move(self.front_steppers, "FORCE_MOVE_Z_FRONT", "front", gcmd)

    cmd_FORCE_MOVE_Z_REAR_help = (
        "Jog ONLY the rear Z pair for calibration (front holds). DISTANCE mm, "
        "+ = up. Leaves Z unhomed."
    )

    def cmd_FORCE_MOVE_Z_REAR(self, gcmd):
        self._force_move(self.rear_steppers, "FORCE_MOVE_Z_REAR", "rear", gcmd)


def load_config(config):
    return VertigoZHoming(config)