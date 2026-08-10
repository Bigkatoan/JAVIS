#!/usr/bin/env python3
"""Automated ODrive velocity-loop PID tuning for JAVIS wheels.

Runs a step-response test (0 -> STEP_VEL turns/s) at several vel_gain /
vel_integrator_gain candidates on each board, scores each candidate on rise
time + overshoot + steady-state error + jitter, and applies + saves the
best-scoring gains.

SAFETY: wheels must be elevated and free-spinning (not touching the ground,
nothing that can catch fingers/cables) before running this -- it commands
real step velocity changes on both wheels repeatedly.

Usage:
    venv/bin/python scripts/tune_wheel_pid.py
    venv/bin/python scripts/tune_wheel_pid.py --step-vel 1.5 --out-dir logs/pid_tune
"""

import argparse
import csv
import os
import time

import numpy as np
import odrive
from odrive.enums import (
    AXIS_STATE_CLOSED_LOOP_CONTROL,
    INPUT_MODE_PASSTHROUGH,
)

LEFT_SERIAL = "318236823335"
RIGHT_SERIAL = "3676365D3335"

# (vel_gain, vel_integrator_gain) candidates to try. Includes the
# motor_web_test.py baseline (0.25, 0.15) plus a spread around it.
GAIN_CANDIDATES = [
    (0.15, 0.10),
    (0.20, 0.10),
    (0.25, 0.15),
    (0.30, 0.15),
    (0.35, 0.20),
    (0.40, 0.20),
]

SAMPLE_DT = 0.005  # ~200 Hz sampling during step test


def connect(serial, name):
    print(f"Connecting {name} (SN {serial})...")
    odrv = odrive.find_any(serial_number=serial, timeout=10)
    if odrv is None:
        raise RuntimeError(f"could not connect to {name} ({serial})")
    return odrv


def ensure_ready(odrv, name):
    odrv.axis0.clear_errors()
    if not odrv.axis0.motor.is_calibrated:
        raise RuntimeError(f"{name}: motor not calibrated, run scripts/setup_odrive.py first")
    odrv.axis0.controller.config.input_mode = INPUT_MODE_PASSTHROUGH
    odrv.axis0.controller.input_vel = 0.0
    odrv.axis0.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL
    time.sleep(0.3)
    if odrv.axis0.current_state != AXIS_STATE_CLOSED_LOOP_CONTROL:
        raise RuntimeError(f"{name}: failed to enter closed loop (state={odrv.axis0.current_state}, "
                            f"error=0x{odrv.axis0.error:x})")


def set_gains(odrv, vel_gain, vel_integrator_gain):
    odrv.axis0.controller.config.vel_gain = vel_gain
    odrv.axis0.controller.config.vel_integrator_gain = vel_integrator_gain


def step_test(odrv, step_vel, duration, dt):
    """Command a step 0 -> step_vel, log (t, cmd, meas) at fixed rate, return to 0."""
    odrv.axis0.controller.input_vel = 0.0
    time.sleep(0.3)  # settle at 0 before stepping

    t_log, meas_log = [], []
    t0 = time.monotonic()
    odrv.axis0.controller.input_vel = step_vel
    while True:
        t = time.monotonic() - t0
        if t > duration:
            break
        meas_log.append(odrv.axis0.encoder.vel_estimate)
        t_log.append(t)
        time.sleep(dt)
    odrv.axis0.controller.input_vel = 0.0
    time.sleep(0.5)  # let it settle back to 0 before next candidate
    return np.array(t_log), np.array(meas_log)


def score_response(t, meas, target):
    """Lower is better. Penalizes slow rise, overshoot, steady-state error, jitter."""
    above_90 = np.where(meas >= 0.9 * target)[0]
    rise_time = t[above_90[0]] if len(above_90) > 0 else t[-1]
    overshoot = max(0.0, (np.max(meas) - target) / target) if target != 0 else 0.0
    tail = meas[int(0.7 * len(meas)):]
    ss_error = np.mean(np.abs(tail - target)) / abs(target) if target != 0 else 0.0
    jitter = np.std(tail)
    score = rise_time * 2.0 + overshoot * 5.0 + ss_error * 10.0 + jitter * 1.0
    return score, dict(rise_time=rise_time, overshoot=overshoot, ss_error=ss_error, jitter=jitter)


def tune_wheel(odrv, name, candidates, step_vel, duration, dt, log_dir):
    print(f"\n=== Tuning {name} ===")
    results = []
    for vg, vig in candidates:
        set_gains(odrv, vg, vig)
        time.sleep(0.1)
        t, meas = step_test(odrv, step_vel, duration, dt)

        err = odrv.axis0.error
        if err != 0:
            print(f"  vel_gain={vg:.2f} vel_integrator_gain={vig:.2f}: "
                  f"ERROR 0x{err:x} during test, skipping candidate")
            odrv.axis0.clear_errors()
            continue

        score, metrics = score_response(t, meas, step_vel)
        print(f"  vel_gain={vg:.2f} vel_integrator_gain={vig:.2f}: "
              f"score={score:.3f} rise={metrics['rise_time']:.3f}s "
              f"overshoot={metrics['overshoot']*100:.1f}% "
              f"ss_err={metrics['ss_error']*100:.1f}% jitter={metrics['jitter']:.3f}")
        results.append((score, vg, vig, metrics))

        csv_path = os.path.join(log_dir, f"{name}_vg{vg}_vig{vig}.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "cmd_vel", "meas_vel"])
            for ti, mi in zip(t, meas):
                w.writerow([ti, step_vel, mi])

    results.sort(key=lambda r: r[0])
    return results


def reversal_check(odrv, name, step_vel):
    """Sanity check the winning gains survive a hard reversal (+v -> -v)."""
    print(f"  [{name}] reversal check: +{step_vel} -> -{step_vel} turns/s")
    odrv.axis0.controller.input_vel = step_vel
    time.sleep(1.0)
    odrv.axis0.controller.input_vel = -step_vel
    time.sleep(1.0)
    err = odrv.axis0.error
    meas = odrv.axis0.encoder.vel_estimate
    odrv.axis0.controller.input_vel = 0.0
    time.sleep(0.5)
    if err != 0:
        print(f"    !! ERROR 0x{err:x} during reversal")
        odrv.axis0.clear_errors()
        return False
    print(f"    ok, measured {meas:.2f} turns/s after reversal (target -{step_vel})")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--step-vel", type=float, default=2.0, help="step target, turns/s")
    parser.add_argument("--duration", type=float, default=1.5, help="seconds to hold each step")
    parser.add_argument("--out-dir", default="logs/pid_tune", help="directory for per-candidate CSV logs")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    left = connect(LEFT_SERIAL, "LEFT")
    right = connect(RIGHT_SERIAL, "RIGHT")

    try:
        ensure_ready(left, "LEFT")
        ensure_ready(right, "RIGHT")

        left_results = tune_wheel(left, "left", GAIN_CANDIDATES, args.step_vel, args.duration, SAMPLE_DT, args.out_dir)
        right_results = tune_wheel(right, "right", GAIN_CANDIDATES, args.step_vel, args.duration, SAMPLE_DT, args.out_dir)

        print("\n=== Results ===")
        for name, results in [("LEFT", left_results), ("RIGHT", right_results)]:
            if not results:
                print(f"{name}: no valid candidate (all errored) -- NOT applying anything")
                continue
            best_score, best_vg, best_vig, best_metrics = results[0]
            print(f"{name}: best vel_gain={best_vg:.2f} vel_integrator_gain={best_vig:.2f} "
                  f"(score={best_score:.3f}, rise={best_metrics['rise_time']:.3f}s, "
                  f"overshoot={best_metrics['overshoot']*100:.1f}%, "
                  f"ss_err={best_metrics['ss_error']*100:.1f}%)")

        if left_results:
            _, vg, vig, _ = left_results[0]
            set_gains(left, vg, vig)
            reversal_check(left, "LEFT", args.step_vel)
            left.save_configuration()
            print(f"LEFT: saved vel_gain={vg:.2f} vel_integrator_gain={vig:.2f}")

        if right_results:
            _, vg, vig, _ = right_results[0]
            set_gains(right, vg, vig)
            reversal_check(right, "RIGHT", args.step_vel)
            right.save_configuration()
            print(f"RIGHT: saved vel_gain={vg:.2f} vel_integrator_gain={vig:.2f}")

    finally:
        # Always leave both wheels stopped, even on error/exception.
        try:
            left.axis0.controller.input_vel = 0.0
        except Exception:
            pass
        try:
            right.axis0.controller.input_vel = 0.0
        except Exception:
            pass
        print("\nBoth wheels commanded to 0. Done.")


if __name__ == "__main__":
    main()
