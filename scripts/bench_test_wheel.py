#!/usr/bin/env python3
"""Simulate the JAVIS rover "on a bench": chassis welded motionless in place
(matching how the real robot was tested -- elevated, wheels free-spinning),
spin one wheel with a velocity command, and render a video so it can be
watched and (once a clean real log exists) compared against real hardware.

Renders to a video file rather than an interactive window, since headless
environments can't reliably pop up a GLFW window -- this works everywhere
and produces a shareable artifact.

Usage:
    # quick built-in step command (4 turn/s, 1.5s)
    venv/bin/python scripts/bench_test_wheel.py --out bench.mp4

    # replay a real command trace (same CSV format as calibrate_actuator.py:
    # t,cmd_vel,meas_vel -- only t and cmd_vel are used here)
    venv/bin/python scripts/bench_test_wheel.py --log real_log.csv --out bench.mp4

    # also overlay sim vs real velocity as a comparison plot
    venv/bin/python scripts/bench_test_wheel.py --log real_log.csv --out bench.mp4 --compare-plot fit.png

IMPORTANT: only pass --log a file recorded with encoder.error == 0 for the
whole capture (see SIM2REAL.md sec 5b -- a real encoder SPI comms fault was
found 2026-08-06; logs captured while that error is set contain garbage
velocity readings, not real wheel motion, and will produce a misleading
comparison).
"""

import argparse
import csv

import imageio
import mujoco
import numpy as np

from javis.robot_constants import WHEEL_JOINTS, get_javis_robot_cfg

from mjlab.entity import Entity

WHEEL_JOINT = "right_wheel"  # matches the physical axis0 test (right wheel)
FPS = 30


def _weld_chassis_to_world(spec: mujoco.MjSpec, chassis: mujoco.MjsBody) -> None:
  """Rigidly fix the chassis in place, like the robot mounted on a bench
  stand for the real test: only the wheels are free to rotate."""
  spec.add_equality(
    type=mujoco.mjtEq.mjEQ_WELD,
    objtype=mujoco.mjtObj.mjOBJ_BODY,
    name1=chassis.name,
    name2="world",
  )


def build_model() -> mujoco.MjModel:
  robot = Entity(get_javis_robot_cfg())
  spec = robot.spec
  chassis = next(b for b in spec.bodies if b.name == "body")
  _weld_chassis_to_world(spec, chassis)
  # A bare Entity has no lights (those normally come from mjlab's Scene/
  # scene.xml) -- add some so the render isn't near-black.
  for i, pos in enumerate([[0.6, 0.6, 1.2], [-0.6, -0.6, 1.0], [0, 0, 1.5]]):
    spec.worldbody.add_light(
      name=f"bench_light_{i}", pos=pos, type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
      diffuse=[0.8, 0.8, 0.8], ambient=[0.3, 0.3, 0.3],
    )
  return spec.compile()


def load_command_trace(log_path: str | None) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
  """Returns (t, cmd_vel, meas_vel_or_None) in seconds / rad/s."""
  if log_path is None:
    t = np.linspace(0, 1.5, int(1.5 * 500))
    cmd = np.where(t < 0.05, 0.0, 4 * 2 * np.pi)  # step to 4 turn/s
    return t, cmd, None
  rows = list(csv.DictReader(open(log_path)))
  t = np.array([float(r["t"]) for r in rows])
  cmd = np.array([float(r["cmd_vel"]) for r in rows])
  meas = np.array([float(r["meas_vel"]) for r in rows])
  return t, cmd, meas


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--log", default=None, help="real t,cmd_vel,meas_vel CSV to replay (default: built-in step)")
  parser.add_argument("--out", default="bench_test.mp4", help="output video path")
  parser.add_argument("--compare-plot", default=None, help="also save a sim-vs-real velocity comparison PNG")
  parser.add_argument("--width", type=int, default=640)
  parser.add_argument("--height", type=int, default=480)
  args = parser.parse_args()

  model = build_model()
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)

  actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, WHEEL_JOINT)
  joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, WHEEL_JOINT)
  dof_adr = model.jnt_dofadr[joint_id]

  t, cmd, meas = load_command_trace(args.log)
  duration = t[-1] - t[0]
  n_frames = max(1, int(duration * FPS))

  renderer = mujoco.Renderer(model, height=args.height, width=args.width)
  cam = mujoco.MjvCamera()
  cam.lookat = data.body("body").xpos.copy()
  cam.distance = 1.3
  cam.azimuth = 120
  cam.elevation = -20

  frames = []
  sim_t, sim_vel = [], []

  next_frame_t = 0.0
  frame_idx = 0
  for i in range(len(t)):
    data.ctrl[actuator_id] = cmd[i]
    mujoco.mj_step(model, data)
    sim_t.append(data.time)
    sim_vel.append(data.qvel[dof_adr])
    if data.time >= next_frame_t and frame_idx < n_frames:
      renderer.update_scene(data, camera=cam)
      frames.append(renderer.render().copy())
      frame_idx += 1
      next_frame_t = frame_idx / FPS

  imageio.mimwrite(args.out, frames, fps=FPS)
  print(f"Wrote {len(frames)} frames ({duration:.2f}s @ {FPS}fps) to {args.out}")

  if args.compare_plot:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(t, cmd, "k--", alpha=0.5, label="commanded")
    if meas is not None:
      ax.plot(t, meas, "C0", label="real (measured)")
    ax.plot(sim_t, sim_vel, "C2", label="sim")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("wheel velocity [rad/s]")
    ax.legend()
    fig.savefig(args.compare_plot, dpi=150)
    print(f"Wrote comparison plot to {args.compare_plot}")


if __name__ == "__main__":
  main()
