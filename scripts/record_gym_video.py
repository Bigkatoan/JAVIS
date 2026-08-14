#!/usr/bin/env python3
"""Record one clip per --algo, each showing that policy across several fixed
noise/terrain scenarios back to back -- the same "watch it, don't just read a
table" idea as scripts/record_payload_video.py, for the plain-MuJoCo/
Gymnasium pipeline (javis/gym_env.py, scripts/train_gym.py).

Each scenario pins the env's terrain (flat/ramp/rough -- real geometry, not
just the training-time gravity-tilt proxy) and a noise-scale multiplier, so
every algorithm is shown the SAME fixed conditions -- differences on screen
are the policy, not luck of the draw. Domain randomization elsewhere (mass,
payload, drivetrain gain, wheel friction) stays on, since "how it handles an
unknown load" is the whole point of this task.

Every frame carries:
  - a checkered ground (javis/gym_env.py's terrain_type materials) so
    translation/rotation actually reads on screen against a static camera,
  - a chase camera that follows the robot's heading (not just its position),
    so "forward" is consistently "into the screen" the way a follow-cam on a
    real RC car would look,
  - a small HUD: a blue arrow for the commanded forward speed and a curved
    wedge for the commanded turn rate, orange for what the robot is actually
    doing, plus the raw numbers underneath.

Usage:
    .venv/bin/python scripts/record_gym_video.py \\
        --runs ppo=logs/gym_compare/<ts>/ppo sac=logs/gym_compare/<ts>/sac \\
               td3=logs/gym_compare/<ts>/td3 ddpg=logs/gym_compare/<ts>/ddpg \\
        --out-dir logs/gym_compare/<ts>/videos
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw
from stable_baselines3 import DDPG, PPO, SAC, TD3

from javis.gym_env import CONTROL_HZ, JavisBalanceEnv

_LOADERS = {"ppo": PPO, "sac": SAC, "td3": TD3, "ddpg": DDPG}
_BLUE = (66, 133, 244)
_ORANGE = (251, 140, 0)

# (label, terrain_type, terrain_slope_deg, noise_scale). Same four
# conditions for every algorithm; terrain matches javis/sim_config.py's
# TerrainMixCfg (10 deg is that cfg's trained slope_range upper bound).
SCENARIOS = [
  ("flat, normal noise", "flat", 0.0, 1.0),
  ("flat, high noise (2x)", "flat", 0.0, 2.0),
  ("10 deg ramp, normal noise", "ramp", 10.0, 1.0),
  ("rough terrain, normal noise", "rough", 0.0, 1.0),
]


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--runs", nargs="+", required=True, help="name=log_dir pairs, one per algorithm")
  p.add_argument("--checkpoint-name", default="model_final", help="filename stem inside each log_dir")
  p.add_argument("--out-dir", type=str, required=True)
  p.add_argument("--seconds", type=float, default=6.0, help="per scenario segment")
  p.add_argument("--fps", type=int, default=30)
  p.add_argument("--width", type=int, default=640)
  p.add_argument("--height", type=int, default=480)
  p.add_argument("--seed", type=int, default=0)
  return p.parse_args()


def _parse_runs(items: list[str]) -> dict[str, Path]:
  out = {}
  for item in items:
    if "=" not in item:
      raise SystemExit(f"--runs entries must be name=path, got: {item}")
    name, path = item.split("=", 1)
    out[name] = Path(path)
  return out


def _heading(env: JavisBalanceEnv) -> float:
  """World-frame yaw of the chassis, radians. See module docstring's chase
  camera note -- azimuth == degrees(heading) puts the camera directly behind
  the robot looking the same direction it's facing (verified empirically:
  MuJoCo's free-camera azimuth places the eye at world angle
  azimuth-180 from lookat, so its viewing direction is azimuth itself)."""
  xmat = env.data.xmat[env._chassis_id].reshape(3, 3)
  return math.atan2(xmat[1, 0], xmat[0, 0])


def _draw_hud(frame: np.ndarray, cmd_vx: float, cmd_wz: float, act_vx: float, act_wz: float) -> np.ndarray:
  """Bottom-right HUD: forward-speed arrows (up=forward) and a turn-rate
  wedge (left/right), blue for commanded, orange for actual -- plus the raw
  numbers, so "which way is it going" has both a glance answer and an exact
  one."""
  img = Image.fromarray(frame)
  draw = ImageDraw.Draw(img)
  w, h = img.size
  cx, cy = w - 90, h - 90
  panel = [cx - 70, cy - 70, cx + 70, cy + 70]
  draw.rectangle(panel, fill=(0, 0, 0, 160), outline=(200, 200, 200))
  draw.line([cx, cy - 55, cx, cy + 15], fill=(90, 90, 90), width=1)  # forward axis
  draw.line([cx - 40, cy + 40, cx + 40, cy + 40], fill=(90, 90, 90), width=1)  # turn axis

  def arrow(x0, y0, x1, y1, color, width=3):
    draw.line([x0, y0, x1, y1], fill=color, width=width)
    ang = math.atan2(y1 - y0, x1 - x0)
    for d in (-1, 1):
      a = ang + d * 2.6
      draw.line([x1, y1, x1 + 8 * math.cos(a), y1 + 8 * math.sin(a)], fill=color, width=width)

  # Forward/back speed: vertical arrow, up = forward. Scale so 0.5 m/s (the
  # trained command range's max) reaches the panel edge.
  scale = 45.0 / 0.5
  arrow(cx, cy - 20, cx, cy - 20 - np.clip(cmd_vx, -0.5, 0.5) * scale, _BLUE)
  arrow(cx, cy - 20, cx, cy - 20 - np.clip(act_vx, -0.5, 0.5) * scale, _ORANGE)

  # Turn rate: horizontal arrow at the bottom, right = turning right (-wz in
  # this robot's convention matches world +z being "up", positive wz = CCW =
  # visually left when viewed from behind, so we flip sign for "right feels
  # right" on screen).
  wscale = 35.0 / 1.0
  arrow(cx, cy + 40, cx - np.clip(cmd_wz, -1.0, 1.0) * wscale, cy + 40, _BLUE)
  arrow(cx, cy + 40, cx - np.clip(act_wz, -1.0, 1.0) * wscale, cy + 40, _ORANGE)

  draw.text((panel[0] + 4, panel[1] - 16), "cmd", fill=_BLUE)
  draw.text((panel[0] + 34, panel[1] - 16), "actual", fill=_ORANGE)
  draw.text((panel[0], panel[3] + 4),
            f"vx {cmd_vx:+.2f}/{act_vx:+.2f} m/s  wz {cmd_wz:+.2f}/{act_wz:+.2f} rad/s",
            fill=(230, 230, 230))
  return np.array(img)


def _caption(frame: np.ndarray, text: str) -> np.ndarray:
  img = Image.fromarray(frame)
  draw = ImageDraw.Draw(img)
  pad = 6
  draw.rectangle([0, 0, img.width, 22], fill=(0, 0, 0))
  draw.text((pad, pad - 4), text, fill=(255, 255, 255))
  return np.array(img)


def record_algo(name: str, checkpoint: Path, args: argparse.Namespace, out_path: Path) -> None:
  if name not in _LOADERS:
    raise SystemExit(f"unknown algo '{name}' -- --runs names must be one of {tuple(_LOADERS)}")
  print(f"[record_gym_video] {name}: loading {checkpoint}")
  model = _LOADERS[name].load(str(checkpoint))

  writer = imageio.get_writer(str(out_path), fps=args.fps, macro_block_size=1)
  cam = mujoco.MjvCamera()
  cam.distance = 1.6
  cam.elevation = -18

  # Control runs at CONTROL_HZ (100), well above --fps (30 by default) --
  # capture every step_dt/(1/fps)'th step so playback lands at real
  # wall-clock speed instead of ~3x slow motion. Same convention as
  # scripts/record_payload_video.py's render_every.
  render_every = max(1, round(CONTROL_HZ / args.fps))

  for label, terrain_type, slope_deg, noise_scale in SCENARIOS:
    env = JavisBalanceEnv(
      seed=args.seed, render_mode="rgb_array", noise_scale=noise_scale,
      terrain_type=terrain_type, terrain_slope_deg=slope_deg, init_xy_jitter=0.0,
    )
    obs, _ = env.reset(seed=args.seed)
    n_steps = round(args.seconds * CONTROL_HZ)
    env._renderer = mujoco.Renderer(env.model, height=args.height, width=args.width)
    for step in range(n_steps):
      action, _ = model.predict(obs, deterministic=True)
      obs, reward, terminated, truncated, info = env.step(action)
      if step % render_every == 0:
        heading = _heading(env)
        cam.azimuth = math.degrees(heading)
        cam.lookat = env.data.xpos[env._chassis_id].copy()
        env._renderer.update_scene(env.data, camera=cam)
        frame = env._renderer.render()
        frame = _draw_hud(
          frame, cmd_vx=float(env._command[0]), cmd_wz=float(env._command[2]),
          act_vx=env._forward_speed(), act_wz=float(env.data.qvel[env._free_dof_adr + 5]),
        )
        frame = _caption(frame, f"{name.upper()} | {label}")
        writer.append_data(frame)
      if terminated or truncated:
        obs, _ = env.reset()
    env.close()

  writer.close()
  print(f"[record_gym_video] wrote {out_path}")


def main() -> None:
  args = parse_args()
  runs = _parse_runs(args.runs)
  out_dir = Path(args.out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  for name, log_dir in runs.items():
    checkpoint = log_dir / f"{args.checkpoint_name}.zip"
    if not checkpoint.exists():
      print(f"[record_gym_video] {name}: no checkpoint at {checkpoint}, skipping")
      continue
    out_path = out_dir / f"{name}_scenarios.mp4"
    record_algo(name, checkpoint, args, out_path)

  print("[record_gym_video] done")


if __name__ == "__main__":
  main()
