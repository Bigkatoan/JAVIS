#!/usr/bin/env python3
"""Derive WHEEL_ACTUATOR_CFG's damping/effort_limit (javis/robot_constants.py)
from real-world data, instead of the current guessed placeholders.

Two modes:

  datasheet   Compute directly from motor/gearbox datasheet numbers, using
              the same reflected-inertia math mjlab's own asset_zoo robots
              use (mjlab.utils.actuator). Good for a first-pass estimate
              before any physical robot exists to test.

  fit         Fit damping/effort_limit against a real step-response log
              recorded from the physical robot (command a wheel to spin at a
              fixed velocity via the ODrive and log the encoder feedback).
              This is the "calibrate against reality" mode: it simulates an
              isolated single wheel -- using the wheel's actual mass/inertia
              from javis/robot_constants.py -- under the same commanded
              velocity trace as the log, and uses least-squares to find the
              (damping, effort_limit) whose simulated response best matches
              the real one. Produces a comparison plot.

Examples:
    venv/bin/python scripts/calibrate_actuator.py datasheet \\
        --no-load-speed-rpm 200 --stall-torque-nm 1.2 --gear-ratio 10

    venv/bin/python scripts/calibrate_actuator.py fit --log wheel_step_test.csv

Log CSV format (fit mode), one row per sample, header required:
    t,cmd_vel,meas_vel
    0.000,0.0,0.0
    0.002,8.0,0.0
    0.004,8.0,1.9
    ...
  t         seconds, monotonically increasing, roughly evenly spaced
  cmd_vel   commanded WHEEL angular velocity, rad/s
  meas_vel  measured WHEEL angular velocity, rad/s

  Note: this robot's encoder doesn't sit directly on the wheel axle -- a
  two-stage gear train (gearbox.stl / spur_gear__20_teeth / __30_teeth in
  the CAD) relocates the encoder magnet up to the MKS ODrive Mini board. The
  two stages (30:20 then 20:30) cancel out, so raw ODrive turns/s already
  equal wheel rad/s -- no conversion needed, log ODrive's numbers directly.
"""

import argparse
import csv

import mujoco
import numpy as np

from javis.robot_constants import WHEEL_ACTUATOR_CFG, get_javis_robot_cfg

from mjlab.entity import Entity
from mjlab.utils.actuator import reflected_inertia


def cmd_datasheet(args: argparse.Namespace) -> None:
  no_load_speed = args.no_load_speed_rpm * 2 * np.pi / 60  # rad/s, at the motor
  velocity_limit = no_load_speed / args.gear_ratio  # rad/s, at the wheel
  effort_limit = args.stall_torque_nm * args.gear_ratio * args.efficiency  # N*m, at the wheel

  # A DC motor's linear torque-speed curve (torque falls off linearly from
  # stall_torque at zero speed to zero at no_load_speed) is exactly what a
  # MuJoCo <velocity> actuator produces: force = damping * (ctrl - joint_vel),
  # clipped to +-effort_limit. So the equivalent damping is just the slope
  # of that line, reflected through the gearbox.
  damping = effort_limit / velocity_limit

  print("# Datasheet-derived actuator config (placeholder until physically")
  print("# verified -- see the 'fit' mode once you have a real robot to test).")
  print(f"# no_load_speed = {args.no_load_speed_rpm} RPM motor -> {velocity_limit:.3f} rad/s wheel")
  print(f"# stall_torque   = {args.stall_torque_nm} N*m motor -> {effort_limit:.3f} N*m wheel "
        f"(efficiency={args.efficiency})")
  print()
  print("WHEEL_ACTUATOR_CFG = BuiltinVelocityActuatorCfg(")
  print("  target_names_expr=WHEEL_JOINTS,")
  print(f"  damping={damping:.4f},")
  print(f"  effort_limit={effort_limit:.4f},")
  if args.rotor_inertia_kgm2 is not None:
    armature = reflected_inertia(args.rotor_inertia_kgm2, args.gear_ratio)
    print(f"  armature={armature:.6f},  # reflected rotor inertia")
  print(")")


def _wheel_inertial_params(joint_name: str) -> tuple[float, float, np.ndarray]:
  """(mass, inertia-about-hinge-axis, hinge axis in world) for one wheel body,
  read from the currently configured robot (javis/robot_constants.py) so the
  fit uses whatever mass the robot is actually configured with."""
  robot = Entity(get_javis_robot_cfg())
  model = robot.spec.compile()

  jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
  if jid < 0:
    raise ValueError(f"No joint named {joint_name!r} in the compiled robot")
  body_id = model.jnt_bodyid[jid]

  mass = float(model.body_mass[body_id])
  idiag = model.body_inertia[body_id]
  iquat = model.body_iquat[body_id]  # principal-axis frame -> body frame

  R = np.zeros((3, 3))
  mujoco.mju_quat2Mat(R.reshape(-1), iquat)
  R = R.reshape(3, 3)

  axis_body = model.jnt_axis[jid]
  axis_principal = R.T @ axis_body
  inertia_about_axis = float(np.sum(idiag * axis_principal**2))
  return mass, inertia_about_axis, axis_body


def _build_single_wheel_model(mass: float, inertia_about_axis: float) -> mujoco.MjModel:
  spec = mujoco.MjSpec()
  spec.option.gravity = [0, 0, 0]  # isolated bench test, not resting under gravity
  body = spec.worldbody.add_body(name="wheel")
  body.mass = mass
  # Only the on-axis moment matters for a single-hinge system; the other two
  # principal components are unconstrained by this test, so set them equal.
  body.inertia = [inertia_about_axis] * 3
  joint = body.add_joint(
    name="wheel_hinge", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 0, 1]
  )
  spec.add_actuator(
    name="wheel_motor",
    target=joint.name,
    trntype=mujoco.mjtTrn.mjTRN_JOINT,
    gear=[1, 0, 0, 0, 0, 0],
  )
  return spec.compile()


def _simulate_velocity_response(
  model: mujoco.MjModel,
  t: np.ndarray,
  cmd_vel: np.ndarray,
  damping: float,
  effort_limit: float,
  phys_dt: float = 5e-4,
) -> np.ndarray:
  # Reproduce a plain MuJoCo <velocity kv="damping"> actuator (what
  # BuiltinVelocityActuatorCfg builds): force = damping * (ctrl - qvel),
  # clipped to +-effort_limit. gaintype/biastype must be set explicitly --
  # the actuator this model was compiled with defaults to a fixed-gain
  # "motor" (force = ctrl), which would apply cmd_vel as a raw torque.
  actuator_id = 0
  model.actuator_gaintype[actuator_id] = mujoco.mjtGain.mjGAIN_FIXED
  model.actuator_gainprm[actuator_id, 0] = damping
  model.actuator_biastype[actuator_id] = mujoco.mjtBias.mjBIAS_AFFINE
  # affine bias = biasprm[0] + biasprm[1]*length + biasprm[2]*velocity --
  # index 2 (velocity), not 1 (length/position): index 1 would add a
  # position-proportional term (an unstable anti-spring here) instead of
  # the velocity damping a <velocity> actuator actually applies.
  model.actuator_biasprm[actuator_id, 2] = -damping
  model.actuator_forcerange[actuator_id] = [-effort_limit, effort_limit]
  model.actuator_forcelimited[actuator_id] = 1

  # A tiny wheel inertia against a stiff velocity servo is a numerically
  # stiff system: integrating at the log's own sample interval (often
  # ~2 ms, i.e. a real control loop's rate) can be far coarser than the
  # servo's own time constant (inertia / damping) and diverge under
  # explicit Euler. Sub-step at a fixed, fine physics dt between log
  # samples instead (matches how mjlab itself decouples physics timestep
  # from control/observation rate via `decimation`), with implicitfast
  # integration for the same stiffness reason mjlab defaults to it.
  model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
  model.opt.timestep = phys_dt

  data = mujoco.MjData(model)
  sim_vel = np.empty_like(t)
  sim_vel[0] = 0.0
  for i in range(1, len(t)):
    dt = t[i] - t[i - 1]
    n_sub = max(1, round(dt / phys_dt))
    data.ctrl[actuator_id] = cmd_vel[i - 1]
    for _ in range(n_sub):
      mujoco.mj_step(model, data)
    sim_vel[i] = data.qvel[0]
  return sim_vel


def cmd_fit(args: argparse.Namespace) -> None:
  from scipy.optimize import least_squares

  rows = list(csv.DictReader(open(args.log)))
  t = np.array([float(r["t"]) for r in rows])
  cmd_vel = np.array([float(r["cmd_vel"]) for r in rows])
  meas_vel = np.array([float(r["meas_vel"]) for r in rows])

  mass, inertia, axis = _wheel_inertial_params(args.joint)
  print(f"Using wheel '{args.joint}': mass={mass:.4f} kg, "
        f"inertia about hinge axis={inertia:.6f} kg*m^2 (axis={axis})")
  model = _build_single_wheel_model(mass, inertia)

  x0 = np.array([WHEEL_ACTUATOR_CFG.damping, WHEEL_ACTUATOR_CFG.effort_limit])

  def residuals(x: np.ndarray) -> np.ndarray:
    damping, effort_limit = x
    sim_vel = _simulate_velocity_response(model, t, cmd_vel, damping, effort_limit)
    return sim_vel - meas_vel

  result = least_squares(residuals, x0, bounds=([1e-4, 1e-4], [np.inf, np.inf]))
  fit_damping, fit_effort_limit = result.x

  print()
  print(f"Fitted from {args.log} ({len(t)} samples, RMS error "
        f"{np.sqrt(np.mean(result.fun**2)):.4f} rad/s):")
  print()
  print("WHEEL_ACTUATOR_CFG = BuiltinVelocityActuatorCfg(")
  print("  target_names_expr=WHEEL_JOINTS,")
  print(f"  damping={fit_damping:.4f},")
  print(f"  effort_limit={fit_effort_limit:.4f},")
  print(")")

  if args.plot:
    import matplotlib.pyplot as plt

    sim_initial = _simulate_velocity_response(model, t, cmd_vel, *x0)
    sim_fitted = _simulate_velocity_response(model, t, cmd_vel, fit_damping, fit_effort_limit)
    fig, ax = plt.subplots()
    ax.plot(t, cmd_vel, "k--", label="commanded", alpha=0.5)
    ax.plot(t, meas_vel, "C0", label="real (measured)")
    ax.plot(t, sim_initial, "C1:", label="sim (initial guess)")
    ax.plot(t, sim_fitted, "C2", label="sim (fitted)")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("wheel velocity [rad/s]")
    ax.legend()
    fig.savefig(args.plot, dpi=150)
    print(f"\nWrote comparison plot to {args.plot}")


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  sub = parser.add_subparsers(dest="mode", required=True)

  p_datasheet = sub.add_parser("datasheet", help="derive config from motor/gearbox datasheet numbers")
  p_datasheet.add_argument("--no-load-speed-rpm", type=float, required=True, help="motor no-load speed, RPM")
  p_datasheet.add_argument("--stall-torque-nm", type=float, required=True, help="motor stall torque, N*m")
  p_datasheet.add_argument("--gear-ratio", type=float, required=True, help="motor:wheel gear reduction ratio")
  p_datasheet.add_argument("--efficiency", type=float, default=0.85, help="drivetrain efficiency (default 0.85)")
  p_datasheet.add_argument("--rotor-inertia-kgm2", type=float, default=None, help="motor rotor inertia, kg*m^2 (optional)")
  p_datasheet.set_defaults(func=cmd_datasheet)

  p_fit = sub.add_parser("fit", help="fit config against a real step-response log")
  p_fit.add_argument("--log", required=True, help="CSV log: t,cmd_vel,meas_vel (see module docstring)")
  p_fit.add_argument("--joint", default="left_wheel", choices=["left_wheel", "right_wheel"])
  p_fit.add_argument("--plot", default=None, help="path to save a sim-vs-real comparison PNG")
  p_fit.set_defaults(func=cmd_fit)

  args = parser.parse_args()
  args.func(args)


if __name__ == "__main__":
  main()
