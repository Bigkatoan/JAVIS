#!/usr/bin/env python3
"""Find the stiffest ODrive velocity-loop gains the simulation can actually run.

Why this is bounded at all: the PI loop in `javis/mdp/actions.py` computes
torque explicitly once per physics step and hands it to MuJoCo. That is a
forward-Euler feedback path, so its stability depends on the physics timestep:

    a = kp * dt / J        stable for a < 2, well damped for a < ~1

`J` is the inertia the loop sees, and the *smallest* value it can ever see sets
the limit. That is the bare wheel, 0.0122 kg·m², which happens whenever a wheel
unloads or slips -- not the ~0.12 kg·m² it sees while actually pushing the
robot. Tuning against the driving case and ignoring the unloaded case is how
you get a loop that is fine until the first time the robot leans hard.

The real board has no such limit (its loop runs at 8 kHz), so this is purely a
simulation constraint. It still matters, because a policy trained against a
loop the hardware cannot reproduce -- in either direction -- will not transfer.

Sweeps (kp, ki), scores each with the same weighting
`scripts/tune_wheel_pid.py` uses on real hardware, and prints the winners in
both per-radian and ODrive-native units so the same numbers can go straight
onto the board.

    .venv/bin/python scripts/tune_sim_gains.py
    .venv/bin/python scripts/tune_sim_gains.py --timestep 0.0025   # 400 Hz physics
"""

import argparse
import math

import mujoco
import numpy as np

from javis import mass_model
from javis.robot_constants import WHEEL_EFFORT_LIMIT_NM, WHEEL_RADIUS_M
from javis.sim_config import DrivetrainCfg

TWO_PI = 2.0 * math.pi


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--timestep", type=float, default=0.0025,
                   help="physics timestep, s (must match SimulationCfg)")
    p.add_argument("--step-rad-s", type=float, default=10.0,
                   help="velocity step to command, rad/s")
    p.add_argument("--duration", type=float, default=1.0)
    p.add_argument("--kp", type=float, nargs="+", default=None,
                   help="per-radian kp values to try; default is a log sweep")
    p.add_argument("--ki-ratios", type=float, nargs="+",
                   default=[0.0, 2.0, 5.0, 10.0, 20.0, 40.0],
                   help="ki = ratio * kp, i.e. 1/ratio is the integral time in s")
    p.add_argument("--max-alpha", type=float, default=0.8,
                   help="reject gains whose kp*dt/J_free exceeds this")
    p.add_argument("--allow-zero-ki", action="store_true",
                   help="let ki=0 win. Off by default: a stiff enough P term "
                        "makes steady-state error vanish on its own, so the "
                        "score would happily pick ki=0 -- but the real board "
                        "always runs an integrator, and simulating one without "
                        "it is exactly the mismatch this loop exists to remove")
    return p.parse_args()


def wheel_inertia() -> float:
    """Spin-axis inertia of one wheel, from the mass model."""
    _, _, _, principal = mass_model.fuse_nominal("wheel")
    # The wheel body's frame is rotated so its spin axis is local z; the spin
    # moment is the distinct (largest) one, the other two being equal.
    return float(np.max(principal))


def build_wheel(inertia: float, dt: float) -> mujoco.MjModel:
    """Single hinge carrying `inertia`, driven by a plain torque actuator.

    Torque actuator, not MuJoCo's <velocity>: the point is to test the PI loop
    we actually ship, which produces torque itself.
    """
    spec = mujoco.MjSpec()
    spec.option.gravity = [0, 0, 0]
    body = spec.worldbody.add_body(name="wheel")
    body.mass = 1.0
    body.inertia = [inertia] * 3
    joint = body.add_joint(name="hinge", type=mujoco.mjtJoint.mjJNT_HINGE,
                           axis=[0, 0, 1])
    spec.add_actuator(name="motor", target=joint.name,
                      trntype=mujoco.mjtTrn.mjTRN_JOINT,
                      gear=[1, 0, 0, 0, 0, 0])
    model = spec.compile()
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    model.opt.timestep = dt
    model.actuator_forcerange[0] = [-WHEEL_EFFORT_LIMIT_NM, WHEEL_EFFORT_LIMIT_NM]
    model.actuator_forcelimited[0] = 1
    return model


def simulate(model, kp: float, ki: float, target: float, dt: float,
             duration: float, ramp: float) -> tuple[np.ndarray, np.ndarray]:
    """Run the same discrete PI loop `javis/mdp/actions.py` implements."""
    data = mujoco.MjData(model)
    n = int(round(duration / dt))
    vel = np.zeros(n)
    torque = np.zeros(n)

    integrator = 0.0
    ramped = 0.0
    limit = WHEEL_EFFORT_LIMIT_NM

    for i in range(n):
        ramped += float(np.clip(target - ramped, -ramp * dt, ramp * dt))
        error = ramped - data.qvel[0]

        candidate = integrator + ki * error * dt
        if abs(kp * error + candidate) <= limit or abs(candidate) < abs(integrator):
            integrator = candidate

        tau = float(np.clip(kp * error + integrator, -limit, limit))
        data.ctrl[0] = tau
        mujoco.mj_step(model, data)

        vel[i] = data.qvel[0]
        torque[i] = tau
    return vel, torque


def score(vel: np.ndarray, target: float, dt: float) -> dict:
    """Rise time, overshoot, steady-state error and jitter, as tune_wheel_pid.py."""
    settled = vel[int(len(vel) * 0.7):]
    ss = float(np.mean(settled))
    ss_error = abs(ss - target) / abs(target)
    overshoot = max(0.0, (float(np.max(vel)) - abs(target)) / abs(target))
    jitter = float(np.std(settled)) / abs(target)

    reached = np.argmax(vel >= 0.9 * target) if np.any(vel >= 0.9 * target) else -1
    rise = reached * dt if reached >= 0 else float("inf")

    diverged = (not np.all(np.isfinite(vel))) or float(np.max(np.abs(vel))) > 50 * abs(target)
    return {
        "rise_s": rise,
        "overshoot": overshoot,
        "ss_error": ss_error,
        "jitter": jitter,
        "diverged": diverged,
        # Same weighting the real-hardware tuner uses, so a gain that wins here
        # would win there too.
        "score": (float("inf") if diverged or not np.isfinite(rise)
                  else 2 * rise + 5 * overshoot + 10 * ss_error + 1 * jitter),
    }


def main() -> None:
    args = parse_args()
    dt = args.timestep
    j_free = wheel_inertia()
    robot_mass = sum(
        mass_model.fuse_nominal(b)[0]
        for b in [mass_model.CHASSIS_BODY, *mass_model.WHEEL_BODIES]
    )
    j_driving = j_free + robot_mass * WHEEL_RADIUS_M**2

    print(f"physics timestep      {dt * 1000:.2f} ms ({1 / dt:.0f} Hz)")
    print(f"wheel inertia         {j_free:.5f} kg m^2   (unloaded -- sets the limit)")
    print(f"effective inertia     {j_driving:.5f} kg m^2   (driving {robot_mass:.2f} kg robot)")
    print(f"stability bound       kp < 2*J/dt = {2 * j_free / dt:.2f} N m/(rad/s)")
    print(f"target alpha <= {args.max_alpha}   ->  kp <= "
          f"{args.max_alpha * j_free / dt:.2f} N m/(rad/s)\n")

    kp_values = args.kp or [
        v for v in (0.15, 0.3, 0.61, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0)
        if v * dt / j_free <= args.max_alpha * 1.5
    ]

    rows = []
    for kp in kp_values:
        alpha = kp * dt / j_free
        for ratio in args.ki_ratios:
            ki = ratio * kp
            ok = True
            per_load = {}
            for tag, inertia in (("free", j_free), ("drive", j_driving)):
                model = build_wheel(inertia, dt)
                vel, torque = simulate(model, kp, ki, args.step_rad_s, dt,
                                       args.duration, DrivetrainCfg().vel_ramp_rad_s2)
                m = score(vel, args.step_rad_s, dt)
                per_load[tag] = m
                ok = ok and not m["diverged"]
            rows.append({
                "kp": kp, "ki": ki, "ratio": ratio, "alpha": alpha,
                "ok": ok and alpha <= args.max_alpha,
                "free": per_load["free"], "drive": per_load["drive"],
                # Rank on the driving case -- that is the job -- but only among
                # gains that are also stable unloaded.
                "score": per_load["drive"]["score"] if ok else float("inf"),
            })

    header = (f"{'kp':>6} {'ki':>7} {'alpha':>6} | {'free rise':>9} {'over':>6} "
              f"{'sserr':>6} | {'drive rise':>10} {'over':>6} {'sserr':>6} "
              f"{'score':>7}  {'vel_gain':>9} {'vel_int':>8}")
    print(header)
    print("-" * len(header))
    for r in sorted(rows, key=lambda x: (not x["ok"], x["score"])):
        flag = "" if r["ok"] else "  <- unstable/over alpha"
        print(f"{r['kp']:6.2f} {r['ki']:7.1f} {r['alpha']:6.2f} | "
              f"{r['free']['rise_s'] * 1000:8.1f}ms {r['free']['overshoot']:6.3f} "
              f"{r['free']['ss_error']:6.3f} | "
              f"{r['drive']['rise_s'] * 1000:9.1f}ms {r['drive']['overshoot']:6.3f} "
              f"{r['drive']['ss_error']:6.3f} "
              f"{r['score']:7.3f}  {r['kp'] * TWO_PI:9.2f} {r['ki'] * TWO_PI:8.1f}{flag}")

    eligible = [r for r in rows if args.allow_zero_ki or r["ki"] > 0.0]
    best = min(eligible, key=lambda x: (not x["ok"], x["score"]))
    print(f"\nbest stable gain at dt = {dt * 1000:.2f} ms:")
    print(f"  kp = {best['kp']:.3f} N m/(rad/s)   ki = {best['ki']:.2f} N m/(rad/s)/s")
    print(f"  alpha = {best['alpha']:.2f} (stability margin "
          f"{2 / best['alpha']:.1f}x on the unloaded wheel)")
    print(f"  velocity-loop time constant: {j_free / best['kp'] * 1000:.1f} ms unloaded, "
          f"{j_driving / best['kp'] * 1000:.1f} ms driving")
    print(f"  robot fall time constant:    "
          f"{math.sqrt(0.2875 / 9.81) * 1000:.0f} ms")
    print("\nput these on the board (ODrive native units):")
    print(f"  axis0.controller.config.vel_gain            = {best['kp'] * TWO_PI:.2f}")
    print(f"  axis0.controller.config.vel_integrator_gain = {best['ki'] * TWO_PI:.2f}")
    print("\nand in javis/sim_config.py, DrivetrainCfg:")
    print(f"  vel_gain = {best['kp'] * TWO_PI:.2f}")
    print(f"  vel_integrator_gain = {best['ki'] * TWO_PI:.2f}")


if __name__ == "__main__":
    main()
