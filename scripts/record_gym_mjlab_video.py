#!/usr/bin/env python3
"""Record a video of an SB3 checkpoint trained via train_gym_mjlab.py,
using the real mjlab task's own renderer (skybox/shadows/reflections) and
built-in target-vs-actual velocity debug_vis (javis/mdp/commands.py) --
no hand-drawn HUD needed, unlike scripts/record_gym_video.py's plain-MuJoCo
equivalent.

Shows several robots at once (--num-envs, small -- these are neighbors in
the SAME batched sim, each with its own randomized mass/CoM/payload/terrain
tile from the task's own domain randomization, not a scripted scenario
list): watching a handful side by side across one continuous rollout says as
much about generalization as a fixed scenario sweep would, and it's exactly
the mjlab task's own DR doing the work, not a hand-picked list.

Usage:
    .venv/bin/python scripts/record_gym_mjlab_video.py \\
        --checkpoint logs/gym_mjlab_sac/<ts>/model_final.zip --algo sac
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
from stable_baselines3 import DDPG, PPO, SAC, TD3

from javis.gym_env_mjlab import JavisMjlabVecEnv

_LOADERS = {"ppo": PPO, "sac": SAC, "td3": TD3, "ddpg": DDPG}


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--checkpoint", type=Path, required=True)
  p.add_argument("--algo", choices=tuple(_LOADERS), required=True)
  p.add_argument("--task", choices=("flat", "rough"), default="rough")
  p.add_argument("--num-envs", type=int, default=6, help="robots shown at once, side by side")
  p.add_argument("--seconds", type=float, default=20.0)
  p.add_argument("--fps", type=int, default=30)
  p.add_argument("--width", type=int, default=960)
  p.add_argument("--height", type=int, default=540)
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--out", type=Path, default=Path("logs/eval/gym_mjlab_scenarios.mp4"))
  return p.parse_args()


def main() -> None:
  args = parse_args()
  args.out.parent.mkdir(parents=True, exist_ok=True)

  print(f"[record_gym_mjlab_video] loading {args.checkpoint} ({args.algo})")
  model = _LOADERS[args.algo].load(str(args.checkpoint))

  env = JavisMjlabVecEnv(
    num_envs=args.num_envs, rough=(args.task == "rough"), device=args.device,
    render_mode="rgb_array", seed=args.seed,
    viewer_env_idx=0, viewer_max_extra_envs=args.num_envs - 1,
    viewer_width=args.width, viewer_height=args.height,
  )

  obs = env.reset()
  render_every = max(1, round((1.0 / env.env.step_dt) / args.fps))
  n_steps = round(args.seconds / env.env.step_dt)

  writer = imageio.get_writer(str(args.out), fps=args.fps, macro_block_size=1)
  for step in range(n_steps):
    action, _ = model.predict(obs, deterministic=True)
    env.step_async(action)
    obs, reward, done, infos = env.step_wait()
    if step % render_every == 0:
      frame = env.render()
      if frame is not None:
        writer.append_data(frame)
  writer.close()
  env.close()
  print(f"[record_gym_mjlab_video] wrote {args.out}")


if __name__ == "__main__":
  main()
