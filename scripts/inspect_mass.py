#!/usr/bin/env python3
"""Print the JAVIS mass budget and the balance envelope it implies.

This is the first thing to run after changing javis/mass_model.py, and the
human checkpoint before trusting anything downstream: if the center of mass is
wrong here, every balance policy trained on it is wrong too.

    .venv/bin/python scripts/inspect_mass.py
    .venv/bin/python scripts/inspect_mass.py --rebuild      # ignore the moment cache
    .venv/bin/python scripts/inspect_mass.py --check-model  # compare vs compiled MuJoCo
"""

import argparse
import math

import numpy as np

from javis import mass_model
from javis.robot_constants import WHEEL_EFFORT_LIMIT_NM, WHEEL_RADIUS_M

GRAVITY = 9.81


def print_balance_envelope(total_mass: float, com_height: float) -> None:
    """The hardware ceiling on how far this robot can lean and recover.

    A wheeled inverted pendulum arrests a lean by accelerating its contact
    point out from under the CoM. The ground force available for that is capped
    by wheel torque: F = 2 * tau_max / r. Holding a static lean of theta needs
    F = M g tan(theta), so the largest sustainable lean is
    atan(2 tau_max / (r M g)) -- independent of CoM height.

    CoM height does set the *timescale*: the pendulum's fall time constant is
    sqrt(h / g), so a taller robot is slower and therefore easier to catch at a
    fixed control rate. Low and heavy is the hard case, not tall and heavy.
    """
    force_max = 2.0 * WHEEL_EFFORT_LIMIT_NM / WHEEL_RADIUS_M
    print("\n=== balance envelope ===")
    print(f"  wheel torque limit   = {WHEEL_EFFORT_LIMIT_NM:.2f} N*m x 2")
    print(f"  wheel radius         = {WHEEL_RADIUS_M:.4f} m")
    print(f"  max ground force     = {force_max:.1f} N")
    print(f"  fall time constant   = {math.sqrt(com_height / GRAVITY) * 1000:.0f} ms "
          f"(CoM height {com_height:.3f} m)")
    print(f"\n  {'total mass':>12}  {'max lean':>9}  {'max accel':>10}  {'max slope':>10}")
    for mass in [total_mass, 15.0, 20.0, 25.0, 30.0]:
        ratio = force_max / (mass * GRAVITY)
        lean = math.degrees(math.atan(ratio)) if ratio < 1e3 else 90.0
        accel = force_max / mass
        # On a slope, holding station already costs M g sin(alpha). The steepest
        # slope the robot can even stand still on is where that eats the budget.
        slope = math.degrees(math.asin(min(1.0, ratio)))
        tag = "  <- nominal" if abs(mass - total_mass) < 1e-6 else ""
        print(f"  {mass:10.2f} kg  {lean:8.1f}d  {accel:8.2f} m/s2  {slope:8.1f}d{tag}")


def check_against_model() -> None:
    """Compile the MjSpec and confirm MuJoCo agrees with the analytic model.

    Guards the hand-off in robot_constants.get_spec(): we compute mass/ipos/
    iquat/inertia ourselves and write them onto the spec with
    inertiafromgeom=FALSE, so a silent disagreement here would mean the sim is
    running different physics than the model this file reports.
    """
    import mujoco

    from javis.robot_constants import get_spec

    model = get_spec().compile()
    print("\n=== compiled MuJoCo model vs analytic model ===")
    worst = 0.0
    for body in [mass_model.CHASSIS_BODY, *mass_model.WHEEL_BODIES]:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
        assert bid >= 0, f"body '{body}' missing from compiled model"
        mass, com, iquat, inertia = mass_model.fuse_nominal(body)
        errs = {
            "mass": abs(model.body_mass[bid] - mass),
            "ipos": float(np.abs(model.body_ipos[bid] - com).max()),
            "inertia": float(np.abs(np.sort(model.body_inertia[bid]) - np.sort(inertia)).max()),
        }
        worst = max(worst, max(errs.values()))
        status = "OK " if max(errs.values()) < 1e-6 else "BAD"
        print(
            f"  {status} {body:8s} mass {model.body_mass[bid]:8.4f} "
            f"(err {errs['mass']:.2e})  ipos err {errs['ipos']:.2e}  "
            f"inertia err {errs['inertia']:.2e}"
        )
    print(f"  worst error = {worst:.2e}" + ("" if worst < 1e-6 else "   <-- MISMATCH"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true",
                        help="recompute moments from the STLs instead of using the cache")
    parser.add_argument("--check-model", action="store_true",
                        help="also compile the MjSpec and diff against the analytic model")
    args = parser.parse_args()

    if args.rebuild:
        mass_model.build_moments(verbose=True)
        mass_model.get_moments(rebuild=True)

    mass_model.report()

    chassis_mass, chassis_com, _, _ = mass_model.fuse_nominal(mass_model.CHASSIS_BODY)
    wheel_mass, _, _, _ = mass_model.fuse_nominal("wheel")
    total = chassis_mass + 2 * wheel_mass

    # The chassis link frame sits at the wheel axle; the ground is one wheel
    # radius below it, so CoM height above ground is com_z + r.
    com_height = float(chassis_com[2]) + WHEEL_RADIUS_M
    print_balance_envelope(total, max(com_height, 1e-3))

    print("\n=== cross-checks ===")
    print(f"  chassis + 2 wheels   = {total:.3f} kg")
    print("  previous guess       = 11.87 kg (CHASSIS_MASS_KG 6.0 + 2 x 2.936)")
    print("  measured so far      = battery 3.423 kg, wheel 2.936 kg each")
    print(f"  CoM above ground     = {com_height:.4f} m")
    print(f"  CoM fore/aft offset  = {chassis_com[0]:+.4f} m "
          f"(equilibrium lean {math.degrees(math.atan2(chassis_com[0], max(com_height, 1e-6))):+.1f} deg)")
    print(f"  CoM lateral offset   = {chassis_com[1]:+.4f} m "
          "(wheel half-track is 0.129 m)")

    if args.check_model:
        check_against_model()


if __name__ == "__main__":
    main()
