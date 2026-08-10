#!/usr/bin/env python3
"""Configure and calibrate an MKS xDrive Mini (ODrive v0.5.1-based firmware,
confirmed via community docs -- see SIM2REAL.md sec 3) for one JAVIS wheel:
36V/350W direct-drive hub motor (pole_pairs=15, confirmed by counting rotor
magnets), onboard SPI absolute AS5047P encoder (CS on GPIO7), velocity
control mode.

IMPORTANT hardware fact (confirmed 2026-08-06, see SIM2REAL.md sec 3 for
sources): each MKS xDrive Mini board is SINGLE-axis -- only axis0 ever has
a real motor. axis1 exists only because the underlying firmware is the
dual-axis ODrive v3.6 base; it's a "ghost" axis with nothing connected. This
robot uses TWO separate boards (one per wheel), not one board's two axes.
Each board's ghost axis1 must be set to CAN node_id=63 (a documented
community fix) so it doesn't collide with the *other* board's real axis0 on
a shared CAN bus -- this script always does that, regardless of which wheel
you're configuring.

DO NOT run odrivetool's "upgrade"/DFU firmware update commands on this
board: it ships modified v0.5.1 firmware, and a standard upgrade will brick
it (recovery needs an ST-Link programmer -- see SIM2REAL.md sec 3).

Also unresolved, separate from anything this script controls: some MKS
xDrive Mini units have a hardware defect in the onboard SN65HVD230 CAN
transceiver (RS pin resistor value keeps it in listen-only mode, so it can
receive but never transmit on CAN) -- if the eventual CAN driver can read
this board but commands never arrive, check for that before assuming a
software bug (see SIM2REAL.md sec 3).

Verified end-to-end on real hardware 2026-08-06 on two separate boards
(right wheel, then left wheel): erase config, set board/motor/encoder/
controller config, save+reboot, motor calibration, encoder offset
calibration, closed-loop control, test spin. Values here (TORQUE_CONSTANT,
VEL_GAIN) were measured/tuned on the right wheel's board -- see
robot_constants.py's WHEEL_ACTUATOR_CFG comment and this file's VEL_GAIN
comment. Applied as-is to the left wheel's board as a starting point since
it's the same motor model; re-measure per-board if that precision matters.

Usage:
    venv/bin/python scripts/setup_odrive.py --wheel right
    venv/bin/python scripts/setup_odrive.py --wheel left --no-test-spin
    venv/bin/python scripts/setup_odrive.py --wheel right --startup-closed-loop

Only one board should be connected via USB at a time when running this --
odrive.find_any() doesn't let you pick a specific serial number, so with
two boards plugged in simultaneously there's no guarantee which one you'd
be configuring.

Safety: this physically spins the motor during encoder offset calibration
and the test spin. Before running, make sure the wheel is elevated/secured
(can't drive the robot into anything or pinch anyone) and someone is
present to power off if something looks wrong.
"""

import argparse
import time

import odrive
from odrive.enums import (
  AXIS_STATE_CLOSED_LOOP_CONTROL,
  AXIS_STATE_ENCODER_OFFSET_CALIBRATION,
  AXIS_STATE_IDLE,
  AXIS_STATE_MOTOR_CALIBRATION,
  CONTROL_MODE_VELOCITY_CONTROL,
  ENCODER_MODE_SPI_ABS_AMS,
  INPUT_MODE_PASSTHROUGH,
  MOTOR_TYPE_HIGH_CURRENT,
)

# --- Battery: 6S Li-ion (Lishen SK 21700), measured/rated 18.0-25.2 V ---
UNDERVOLTAGE_TRIP = 19.0  # small margin above the battery's 18.0 V floor
OVERVOLTAGE_TRIP = 28.0  # margin above 25.2 V full-charge for regen spikes
DC_MAX_POSITIVE_CURRENT = 20.0  # conservative bus current cap for bring-up
DC_MAX_NEGATIVE_CURRENT = -1.0  # limit regen; brake resistor use unconfirmed
BRAKE_RESISTANCE = 2.0
MAX_REGEN_CURRENT = 0

# --- Motor: 36V/350W direct-drive hub motor, 8" wheel ---
POLE_PAIRS = 15  # 30 rotor magnets / 2, counted 2026-08-06
CALIBRATION_CURRENT = 5
RESISTANCE_CALIB_MAX_VOLTAGE = 2
CURRENT_LIM = 15.0  # <= 16 A continuous rating from the motor's OEM datasheet
REQUESTED_CURRENT_RANGE = 20
TORQUE_CONSTANT = 0.207  # N*m/A, measured 2026-08-06 (see robot_constants.py
# WHEEL_ACTUATOR_CFG comment for the measurement method) -- replaces the
# firmware's untouched default of 0.04, which was never actually calibrated

# --- Encoder: onboard SPI absolute (AMS chip), confirmed CS pin 2026-08-06 ---
ENCODER_CS_GPIO_PIN = 7
ENCODER_CPR = 16384
ENCODER_BANDWIDTH = 3000
# Fraction-of-expected-response tolerance for the CPR/pole-pairs sanity check
# run at the end of AXIS_STATE_ENCODER_OFFSET_CALIBRATION (fails with
# ERROR_CPR_POLEPAIRS_MISMATCH if exceeded). This was 10 (1000% tolerance --
# 500x looser than the firmware's own default of 0.02), which silently
# accepted a calibration scan even when the encoder's response during the
# scan was wildly inconsistent with the commanded phase. Confirmed via the
# firmware source (Firmware/MotorControl/encoder.hpp) that 0.02 is the
# intended default; using 0.05 here for a little real-world noise margin.
ENCODER_CALIB_RANGE = 0.05

# --- Controller: velocity mode (matches WHEEL_ACTUATOR_CFG / velocity_task.py) ---
VEL_LIMIT = 15.0  # turns/s safety ceiling -- NOT the sample's 120 (~7200 RPM)
VEL_GAIN = 0.3  # verified 2026-08-06 on real hardware: clean 1.5-3 turn/s
# tracking, <2A typical current, 0 errors over a 5s run.
#
# Debugging note (don't repeat this the hard way): mid-session, testing
# with vel_gain=0.85 looked like a serious hardware fault -- current pinned
# near current_lim, velocity overshooting ~2x target, and an intermittent
# ERROR_ENCODER_FAILED (SPI comms fault). Chased it through motor phase
# wiring inspection, encoder recalibration, and a full current/speed sweep.
# Stale live config (vel_gain, current_lim changed interactively without a
# save_configuration()+reboot() cycle) turned out to reproduce it reliably,
# and a clean save+reboot+recalibration fixed it -- but NOT permanently:
# a later clean reboot (startup_encoder_offset_calibration auto-running,
# no interactive config changes at all) reproduced the exact same
# high-current/near-zero-velocity symptom again, no error flag either
# time. So the honest conclusion is: stale config *can* cause this, but
# isn't the only cause -- encoder offset calibration itself sometimes
# converges to a bad commutation angle, silently (no error raised),
# without a diagnosed root cause (best guess still EMI/magnet-alignment
# during the calibration routine itself, see SIM2REAL.md sec 5b). Never
# assume one successful-looking calibration (is_ready=True, no errors) is
# actually good -- verify_and_fix_calibration() below checks the resulting
# torque/speed relationship on a real test spin and redoes the offset
# calibration if it looks wrong, since ODrive itself won't tell you.
VEL_INTEGRATOR_GAIN = 0.2

CAN_BAUD_RATE = 500000
GHOST_AXIS_CAN_NODE_ID = 63  # community-documented fix, see module docstring
WHEEL_CAN_NODE_ID = {"right": 0, "left": 1}  # axis0's real CAN ID per board


def wait_for_idle(axis, timeout_s: float, what: str) -> None:
  """Poll until the axis returns to IDLE (calibration routines vary in how
  long they take -- e.g. encoder offset calibration took >8s but <11s in
  testing -- so poll rather than guessing a fixed sleep)."""
  start = time.monotonic()
  while axis.current_state != AXIS_STATE_IDLE:
    if time.monotonic() - start > timeout_s:
      raise TimeoutError(f"{what}: timed out after {timeout_s}s, still in state {axis.current_state}")
    time.sleep(0.5)
  if axis.error != 0:
    raise RuntimeError(f"{what} failed: axis.error={axis.error}")


def calibrate_encoder_with_retry(axis, max_attempts: int = 3) -> bool:
  """Run AXIS_STATE_ENCODER_OFFSET_CALIBRATION, retrying on a hard error
  (not just a silently-bad result -- see verify_and_fix_calibration for
  that) instead of raising on the first failure. Confirmed necessary on
  real hardware: a from-scratch calibration attempt failed outright with
  encoder.error=ERROR_ABS_SPI_COM_FAIL (128) on its first try, succeeded
  cleanly on retry -- see SIM2REAL.md sec 5b. Returns True if any attempt
  ends with encoder.is_ready and no error.
  """
  for attempt in range(1, max_attempts + 1):
    axis.error = 0
    axis.encoder.error = 0
    axis.requested_state = AXIS_STATE_ENCODER_OFFSET_CALIBRATION
    start = time.monotonic()
    while axis.current_state != AXIS_STATE_IDLE:
      if time.monotonic() - start > 20:
        print(f"  attempt {attempt}/{max_attempts}: timed out, still in state {axis.current_state}")
        break
      time.sleep(0.3)
    if axis.error == 0 and axis.encoder.is_ready:
      if attempt > 1:
        print(f"  succeeded on attempt {attempt}/{max_attempts}.")
      return True
    print(f"  attempt {attempt}/{max_attempts}: axis.error={axis.error}, encoder.error={axis.encoder.error}")
  return False


# A well-calibrated axis should reach a modest commanded speed on a small
# fraction of current_lim (the working run this was tuned against: 3 turn/s
# on <2A, against a 15A current_lim). A badly-calibrated one draws current
# steadily (integrator winding up against a wrong commutation angle) while
# barely moving. These thresholds are deliberately loose -- this is a sanity
# check for "clearly broken", not a precision measurement.
VERIFY_TEST_VEL = 2.5  # turn/s
VERIFY_MAX_CURRENT_A = 4.0
VERIFY_MIN_VEL_FRACTION = 0.6


def verify_and_fix_calibration(axis, max_attempts: int = 3) -> bool:
  """Command a brief test spin and check the resulting speed/current
  relationship looks like a properly-calibrated motor, not the "high
  current, near-zero velocity" pattern of a bad encoder offset calibration
  (see the VEL_GAIN comment above -- ODrive does not raise an error for
  this on its own). Redo the offset calibration and recheck if it looks
  wrong, up to max_attempts times. Leaves the axis in IDLE either way.
  """
  for attempt in range(1, max_attempts + 1):
    axis.error = 0
    axis.controller.input_vel = 0
    axis.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL
    time.sleep(0.3)
    axis.controller.input_vel = VERIFY_TEST_VEL
    time.sleep(1.0)
    vel = axis.encoder.vel_estimate
    iq = axis.motor.current_control.Iq_measured
    axis.controller.input_vel = 0
    time.sleep(0.2)
    axis.requested_state = AXIS_STATE_IDLE

    good = abs(vel) > VERIFY_MIN_VEL_FRACTION * VERIFY_TEST_VEL and abs(iq) < VERIFY_MAX_CURRENT_A
    print(
      f"  verify attempt {attempt}/{max_attempts}: commanded {VERIFY_TEST_VEL} turn/s, "
      f"measured vel={vel:.2f} turn/s, Iq={iq:.2f} A -> {'OK' if good else 'BAD'}"
    )
    if good:
      return True
    if attempt < max_attempts:
      print("  Re-running encoder offset calibration...")
      if not calibrate_encoder_with_retry(axis, max_attempts=2):
        return False
  return False


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument(
    "--wheel", required=True, choices=["left", "right"],
    help="which wheel's board this is -- sets axis0's CAN node_id "
    f"({WHEEL_CAN_NODE_ID}) so the two boards don't collide on the shared CAN bus",
  )
  parser.add_argument("--erase", action="store_true", default=True, help="erase existing config first (default on)")
  parser.add_argument("--no-erase", dest="erase", action="store_false")
  parser.add_argument("--test-spin", action="store_true", default=True, help="spin at 1 turn/s briefly to verify (default on)")
  parser.add_argument("--no-test-spin", dest="test_spin", action="store_false")
  parser.add_argument(
    "--startup-closed-loop",
    action="store_true",
    default=False,
    help="auto-enter closed-loop control on power-up (off by default -- leaves the "
    "board idle after this script so motion stays deliberate until you're ready)",
  )
  parser.add_argument("--find-timeout", type=float, default=15.0)
  args = parser.parse_args()

  print(f"Looking for ODrive (timeout {args.find_timeout}s)...")
  odrv = odrive.find_any(timeout=args.find_timeout)
  print(f"Connected. serial={hex(odrv.serial_number)} vbus={odrv.vbus_voltage:.2f} V")

  if args.erase:
    print("Erasing existing configuration...")
    try:
      odrv.erase_configuration()
    except Exception:
      pass  # board reboots on erase; connection drop is expected
    time.sleep(3)
    odrv = odrive.find_any(timeout=args.find_timeout)
    print("Reconnected after erase.")

  # axis0 is always the real motor on this single-axis board (see module
  # docstring); axis1 is the ghost that just needs a non-colliding CAN id.
  axis = odrv.axis0
  ghost = odrv.axis1

  print("Applying board-level config (battery voltage/current limits)...")
  odrv.config.brake_resistance = BRAKE_RESISTANCE
  odrv.config.dc_bus_undervoltage_trip_level = UNDERVOLTAGE_TRIP
  odrv.config.dc_bus_overvoltage_trip_level = OVERVOLTAGE_TRIP
  odrv.config.dc_max_positive_current = DC_MAX_POSITIVE_CURRENT
  odrv.config.dc_max_negative_current = DC_MAX_NEGATIVE_CURRENT
  odrv.config.max_regen_current = MAX_REGEN_CURRENT

  print(f"Applying motor/encoder/controller config for the {args.wheel} wheel (axis0)...")
  axis.motor.config.pole_pairs = POLE_PAIRS
  axis.motor.config.calibration_current = CALIBRATION_CURRENT
  axis.motor.config.resistance_calib_max_voltage = RESISTANCE_CALIB_MAX_VOLTAGE
  axis.motor.config.motor_type = MOTOR_TYPE_HIGH_CURRENT
  axis.motor.config.current_lim = CURRENT_LIM
  axis.motor.config.requested_current_range = REQUESTED_CURRENT_RANGE
  axis.motor.config.torque_constant = TORQUE_CONSTANT

  axis.encoder.config.mode = ENCODER_MODE_SPI_ABS_AMS
  axis.encoder.config.abs_spi_cs_gpio_pin = ENCODER_CS_GPIO_PIN
  axis.encoder.config.cpr = ENCODER_CPR
  axis.encoder.config.bandwidth = ENCODER_BANDWIDTH
  axis.encoder.config.calib_range = ENCODER_CALIB_RANGE

  axis.controller.config.control_mode = CONTROL_MODE_VELOCITY_CONTROL
  axis.controller.config.vel_limit = VEL_LIMIT
  axis.controller.config.vel_gain = VEL_GAIN
  axis.controller.config.vel_integrator_gain = VEL_INTEGRATOR_GAIN
  axis.controller.config.input_mode = INPUT_MODE_PASSTHROUGH

  axis.config.can_node_id = WHEEL_CAN_NODE_ID[args.wheel]
  ghost.config.can_node_id = GHOST_AXIS_CAN_NODE_ID
  odrv.can.set_baud_rate(CAN_BAUD_RATE)

  odrv.save_configuration()
  print("Saved. Rebooting (required -- GPIO pin assignment etc. only takes effect after reboot)...")
  try:
    odrv.reboot()
  except Exception:
    pass
  time.sleep(4)
  odrv = odrive.find_any(timeout=args.find_timeout)
  axis = odrv.axis0
  print("Reconnected after reboot.")

  print("Motor calibration...")
  axis.requested_state = AXIS_STATE_MOTOR_CALIBRATION
  wait_for_idle(axis, timeout_s=15, what="motor calibration")
  assert axis.motor.is_calibrated
  axis.motor.config.pre_calibrated = True
  odrv.save_configuration()
  print(
    f"  OK. phase_resistance={axis.motor.config.phase_resistance:.4f} ohm "
    f"phase_inductance={axis.motor.config.phase_inductance:.6f} H"
  )

  print("Encoder offset calibration (motor will move)...")
  if not calibrate_encoder_with_retry(axis):
    raise RuntimeError(
      "Encoder offset calibration failed repeatedly (see SIM2REAL.md sec 5b -- "
      "an intermittent SPI comms fault with the onboard encoder chip). Not "
      "something more retries fix; needs physical inspection."
    )
  print("  OK. encoder ready (is_ready=True, no error) -- but that alone isn't")
  print("  trustworthy, see the VEL_GAIN comment. Verifying with a test spin...")

  if not verify_and_fix_calibration(axis):
    raise RuntimeError(
      "Encoder offset calibration still looks bad (high current, little motion) "
      "after retrying. Not marking this axis ready for autonomous startup -- "
      "see SIM2REAL.md sec 5b. Physical inspection likely needed before trusting "
      "this axis on the installed robot."
    )
  print("  Verified good.")

  # Deliberately left False. encoder.is_ready is NOT persisted (unlike
  # motor.config.pre_calibrated) -- it resets on every power cycle, so the
  # tempting fix is startup_encoder_offset_calibration=True, auto-running
  # calibration on every boot. Tried that first; it's also what the
  # original "sample, not real robot" odriveconfig.txt did. But community
  # docs for this exact board (see module docstring / SIM2REAL.md sec 3)
  # report this board has noise-prone encoder init specifically during
  # automatic startup calibration, and recommend the opposite: leave this
  # False and have whatever software drives the board run the offset
  # calibration explicitly (calibrate_encoder_with_retry) with a short
  # pause before closed-loop control -- exactly what this script and
  # verify_and_fix_calibration already do. Whatever eventually drives this
  # robot in normal operation MUST run that same
  # calibrate+verify-with-retry sequence on its own startup, every time --
  # there is no config flag that makes this reliable on its own (see
  # SIM2REAL.md sec 5b/6).
  axis.config.startup_encoder_offset_calibration = False
  odrv.save_configuration()

  if args.startup_closed_loop:
    axis.config.startup_closed_loop_control = True
    odrv.save_configuration()

  print("Entering closed-loop control...")
  # The intermittent encoder fault (SIM2REAL.md sec 5b) has been observed
  # striking here too, after a clean calibration + passed verification --
  # it's not confined to the calibration step, so this needs the same
  # clear-and-retry treatment, not just a one-shot check.
  entered = False
  for attempt in range(1, 4):
    axis.error = 0
    axis.encoder.error = 0
    axis.controller.input_vel = 0
    axis.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL
    time.sleep(1.5)
    if axis.error == 0:
      entered = True
      break
    print(f"  attempt {attempt}/3: axis.error={axis.error}, encoder.error={axis.encoder.error}")
  if not entered:
    raise RuntimeError(
      "Repeatedly failed to enter closed-loop control (see SIM2REAL.md sec 5b -- "
      "the intermittent encoder fault can strike outside calibration too, not just "
      "during it). Needs physical inspection, not more retries."
    )
  print(f"  OK. state={axis.current_state}")

  if args.test_spin:
    print("Test spin: 1 turn/s for 2s...")
    axis.controller.input_vel = 1
    time.sleep(2)
    print(f"  commanded 1.0 turn/s, measured {axis.encoder.vel_estimate:.3f} turn/s, error={axis.error}")
    axis.controller.input_vel = 0
    time.sleep(1)

  axis.requested_state = AXIS_STATE_IDLE
  print(f"Done. {args.wheel} wheel back to IDLE (state={axis.current_state}).")


if __name__ == "__main__":
  main()
