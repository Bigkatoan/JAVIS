#!/usr/bin/env python3
"""Full ODrive setup for JAVIS wheels -- ported to the v0.5.6 firmware API.

This is scripts/setup_odrive.py adapted for the property renames introduced
in firmware v0.5.2 (carried through to v0.5.6, the version actually flashed):
  - encoder.config.offset       -> encoder.config.phase_offset
  - encoder.config.offset_float -> encoder.config.phase_offset_float
  - motor.config.direction      -> encoder.config.direction
  - GPIOs now require an explicit odrv0.config.gpioN_mode before use
    (the SPI encoder nCS pin needs GPIO_MODE_DIGITAL)

Also fixes a real bug carried over from the old script: ENCODER_CALIB_RANGE
was 10 (the fraction-of-expected-response tolerance for the CPR/pole-pairs
sanity check), when the firmware's own default is 0.02 -- 500x too loose,
which silently accepted a calibration scan even if the encoder response
during the scan was wildly wrong. Fixed to 0.05 here (defaults to a bit
looser than the firmware default to allow for real-world noise, but nowhere
near as permissive as the old 10).

Motor/board parameters below are the same values used throughout this
project's diagnostics session (measured on this exact hardware): pole_pairs
confirmed by both physically counting 30 rotor magnets AND by an open-loop
lockin_spin test that matched the theoretical encoder count rate to <1%.

Only run this against a board already reflashed to v0.5.6 -- on v0.5.1
firmware these property names don't exist and this script will fail with
AttributeError immediately (which is a safe failure mode, not a risk to the
board).
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
UNDERVOLTAGE_TRIP = 19.0
OVERVOLTAGE_TRIP = 28.0
DC_MAX_POSITIVE_CURRENT = 20.0
DC_MAX_NEGATIVE_CURRENT = -1.0
BRAKE_RESISTANCE = 2.0

# --- Motor: 36V/350W direct-drive hub motor, 8" wheel ---
POLE_PAIRS = 15  # confirmed by magnet count AND lockin_spin open-loop test
CALIBRATION_CURRENT = 10
RESISTANCE_CALIB_MAX_VOLTAGE = 2
CURRENT_LIM = 15.0
REQUESTED_CURRENT_RANGE = 20
TORQUE_CONSTANT = 0.207

# --- Encoder: onboard SPI absolute (AS5047P), CS on GPIO7 ---
ENCODER_CS_GPIO_PIN = 7
ENCODER_CPR = 16384
ENCODER_BANDWIDTH = 3000
ENCODER_CALIB_RANGE = 0.05  # was 10 in the old 0.5.1 script -- see module docstring

# --- Controller: velocity mode ---
VEL_LIMIT = 15.0
VEL_GAIN = 0.3
VEL_INTEGRATOR_GAIN = 0.2

WHEEL_CAN_NODE_ID = {"right": 0, "left": 1}

VERIFY_TEST_VEL = 2.5
VERIFY_MAX_CURRENT_A = 4.0
VERIFY_MIN_VEL_FRACTION = 0.6


def wait_for_idle(axis, timeout_s: float, what: str) -> None:
    start = time.monotonic()
    while axis.current_state != AXIS_STATE_IDLE:
        if time.monotonic() - start > timeout_s:
            raise TimeoutError(f"{what}: timed out after {timeout_s}s, still in state {axis.current_state}")
        time.sleep(0.3)
    if axis.error != 0:
        raise RuntimeError(f"{what}: axis.error={axis.error}")


def apply_config(odrv, axis, wheel: str) -> None:
    odrv.config.brake_resistance = BRAKE_RESISTANCE
    odrv.config.dc_bus_undervoltage_trip_level = UNDERVOLTAGE_TRIP
    odrv.config.dc_bus_overvoltage_trip_level = OVERVOLTAGE_TRIP
    odrv.config.dc_max_positive_current = DC_MAX_POSITIVE_CURRENT
    odrv.config.dc_max_negative_current = DC_MAX_NEGATIVE_CURRENT
    odrv.config.max_regen_current = 0

    # GPIOs now need an explicit mode (new in 0.5.2+) -- the encoder SPI nCS
    # pin must be GPIO_MODE_DIGITAL or the SPI transaction silently can't
    # drive chip-select.
    from odrive.enums import GPIO_MODE_DIGITAL
    setattr(odrv.config, f"gpio{ENCODER_CS_GPIO_PIN}_mode", GPIO_MODE_DIGITAL)

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

    axis.config.can.node_id = WHEEL_CAN_NODE_ID[wheel]


def calibrate_encoder_with_retry(axis, max_attempts: int = 3) -> bool:
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


def verify_and_fix_calibration(axis, max_attempts: int = 3) -> bool:
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
    parser.add_argument("--wheel", required=True, choices=["left", "right"])
    parser.add_argument("--erase", action="store_true", default=True)
    parser.add_argument("--no-erase", dest="erase", action="store_false")
    parser.add_argument("--test-spin", action="store_true", default=True)
    parser.add_argument("--no-test-spin", dest="test_spin", action="store_false")
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
            pass
        time.sleep(2.0)
        odrv = odrive.find_any(timeout=args.find_timeout)
        print("Reconnected after erase.")

    axis = odrv.axis0

    print("Applying board-level and motor/encoder/controller config...")
    apply_config(odrv, axis, args.wheel)
    odrv.save_configuration()
    time.sleep(2.0)
    odrv = odrive.find_any(timeout=args.find_timeout)
    axis = odrv.axis0
    print("Reconnected after reboot.")

    print("Motor calibration...")
    axis.requested_state = AXIS_STATE_MOTOR_CALIBRATION
    wait_for_idle(axis, 30, "motor calibration")
    print(f"  OK. phase_resistance={axis.motor.config.phase_resistance:.4f} ohm "
          f"phase_inductance={axis.motor.config.phase_inductance:.6f} H")

    print("Encoder offset calibration (motor will move)...")
    if not calibrate_encoder_with_retry(axis):
        raise RuntimeError("Encoder offset calibration failed outright after retries.")
    print("  OK. Verifying with a test spin...")

    if args.test_spin:
        if not verify_and_fix_calibration(axis):
            raise RuntimeError(
                "Encoder offset calibration still looks bad (high current, little motion) "
                "after retrying, even on v0.5.6 firmware."
            )

    axis.motor.config.pre_calibrated = True
    axis.encoder.config.pre_calibrated = True
    odrv.save_configuration()
    print(f"\n{args.wheel.upper()} wheel setup complete and saved.")


if __name__ == "__main__":
    main()
