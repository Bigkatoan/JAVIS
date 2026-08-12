#!/usr/bin/env python3
"""Record one clip per --algo, each showing that policy across several fixed
noise/terrain scenarios back to back -- the same "watch it, don't just read a
table" idea as scripts/record_payload_video.py, for the plain-MuJoCo/
Gymnasium pipeline (javis/gym_env.py, scripts/train_gym.py).

Each scenario pins the env's terrain-proxy slope and a noise-scale multiplier
(JavisBalanceEnv's force_slope_rad / noise_scale, both otherwise randomized
per-episode during training) so every algorithm is shown the SAME fixed
conditions -- differences on screen are the policy, not luck of the draw.
Domain randomization elsewhere (mass, payload, drivetrain gain, wheel
friction) stays on, since "how it handles an unknown load" is the whole
point of this task.

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

# (label, force_slope_rad, noise_scale). Same four conditions for every
# algorithm. 0.3x/2.0x noise bracket the 1.0x the policy actually trained
# under, to show margin (or lack of it) on either side.
SCENARIOS = [
  ("flat, low noise (0.3x)", 0.0, 0.3),
  ("flat, normal noise (1.0x)", 0.0, 1.0),
  ("10 deg slope, normal noise (1.0x)", math.radians(10.0), 1.0),
  ("10 deg slope, high noise (2.0x)", math.radians(10.0), 2.0),
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
  cam.distance = 1.4
  cam.azimuth = 120
  cam.elevation = -20

  # Control runs at CONTROL_HZ (100), well above --fps (30 by default) --
  # capture every step_dt/(1/fps)'th step so playback lands at real
  # wall-clock speed instead of ~3x slow motion. Same convention as
  # scripts/record_payload_video.py's render_every.
  render_every = max(1, round(CONTROL_HZ / args.fps))

  for label, slope_rad, noise_scale in SCENARIOS:
    env = JavisBalanceEnv(
      seed=args.seed, render_mode="rgb_array",
      force_slope_rad=slope_rad, noise_scale=noise_scale,
    )
    obs, _ = env.reset(seed=args.seed)
    n_steps = round(args.seconds * CONTROL_HZ)
    env._renderer = mujoco.Renderer(env.model, height=args.height, width=args.width)
    for step in range(n_steps):
      action, _ = model.predict(obs, deterministic=True)
      obs, reward, terminated, truncated, info = env.step(action)
      if step % render_every == 0:
        cam.lookat = env.data.xpos[env._chassis_id].copy()
        env._renderer.update_scene(env.data, camera=cam)
        frame = env._renderer.render()
        caption = f"{name.upper()} | {label}"
        writer.append_data(_caption(frame, caption))
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
