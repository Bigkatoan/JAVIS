#!/usr/bin/env python3
"""Train javis/balance_task.py's REAL mjlab task (terrain generator, built-in
target-vs-actual debug_vis, GPU-batched physics) through stable-baselines3,
via javis/gym_env_mjlab.py's SB3 VecEnv wrapper.

The companion to scripts/train_gym.py, which reimplements the sim by hand
against plain `mujoco` (no GPU batching, plainer rendering -- see
javis/gym_env.py's docstring). This script trades that independence for the
mjlab task's own validated physics/rendering: `--num-envs` here means
GPU-batched envs inside ONE process (mjlab-sized, hundreds to thousands),
not `train_gym.py`'s SubprocVecEnv-sized tens.

Same terminal + CSV logging as train_gym.py -- no wandb, no tensorboard.
Auto-plots and auto-records a scenario video when training finishes (same
opt-outs: --no-plot / --no-video).

Usage:
    .venv/bin/python scripts/train_gym_mjlab.py --algo ppo --num-envs 4096 --total-timesteps 20000000
    .venv/bin/python scripts/train_gym_mjlab.py --algo sac --num-envs 512  --total-timesteps 2000000
"""

from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import DDPG, PPO, SAC, TD3
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.logger import configure
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import VecMonitor

from javis.gym_env_mjlab import JavisMjlabVecEnv

ON_POLICY = ("ppo",)
OFF_POLICY = ("sac", "td3", "ddpg")


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--algo", choices=(*ON_POLICY, *OFF_POLICY), default="ppo")
  p.add_argument("--num-envs", type=int, default=1024, help="GPU-batched envs in this one process")
  p.add_argument("--task", choices=("flat", "rough"), default="rough")
  p.add_argument("--total-timesteps", type=int, default=20_000_000)
  p.add_argument("--n-steps", type=int, default=24,
                  help="PPO only: rollout length per env before each update")
  p.add_argument("--buffer-size", type=int, default=1_000_000,
                  help="SAC/TD3/DDPG only: replay buffer capacity")
  p.add_argument("--learning-starts", type=int, default=10_000,
                  help="SAC/TD3/DDPG only: random-action warmup steps before training starts")
  p.add_argument("--difficulty", type=float, default=1.0, help="DR level in [0, 1]; 1.0 = full hard envelope")
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--seed", type=int, default=42)
  p.add_argument("--log-dir", type=str, default=None, help="default: logs/gym_mjlab_<algo>/<timestamp>")
  p.add_argument("--save-interval-steps", type=int, default=2_000_000,
                  help="save a checkpoint every this many total env steps")
  p.add_argument("--no-plot", action="store_true", help="skip auto-plotting after training")
  p.add_argument("--no-video", action="store_true", help="skip auto-recording a scenario video after training")
  p.add_argument("--video-seconds", type=float, default=20.0)
  return p.parse_args()


def _build_model(algo: str, vec_env: VecMonitor, args: argparse.Namespace) -> BaseAlgorithm:
  action_dim = vec_env.action_space.shape[0]

  if algo == "ppo":
    buffer_size = args.n_steps * args.num_envs
    batch_size = max(256, buffer_size // 4)
    while buffer_size % batch_size != 0 and batch_size > 1:
      batch_size -= 1
    return PPO(
      "MlpPolicy",
      vec_env,
      n_steps=args.n_steps,
      batch_size=batch_size,
      n_epochs=5,
      gamma=0.99,
      gae_lambda=0.95,
      clip_range=0.2,
      ent_coef=0.005,
      learning_rate=1e-3,
      max_grad_norm=1.0,
      target_kl=0.02,
      policy_kwargs=dict(
        net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128]),
        activation_fn=torch.nn.ELU,
      ),
      seed=args.seed,
      device=args.device,
      verbose=1,
    )

  # Off-policy: see scripts/train_gym.py's _build_model for the
  # gradient_steps == num_envs reasoning (matches train_freq=1's
  # single-env update-to-data ratio under vectorized collection).
  common = dict(
    policy="MlpPolicy",
    env=vec_env,
    buffer_size=args.buffer_size,
    learning_starts=args.learning_starts,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    train_freq=1,
    gradient_steps=args.num_envs,
    seed=args.seed,
    device=args.device,
    verbose=1,
  )

  if algo == "sac":
    return SAC(
      **common,
      learning_rate=3e-4,
      ent_coef="auto",
      policy_kwargs=dict(
        net_arch=dict(pi=[256, 256, 128], qf=[256, 256, 128]),
        activation_fn=torch.nn.ELU,
      ),
    )

  action_noise = NormalActionNoise(mean=np.zeros(action_dim), sigma=0.1 * np.ones(action_dim))
  policy_kwargs = dict(
    net_arch=dict(pi=[256, 256, 128], qf=[256, 256, 128]),
    activation_fn=torch.nn.ELU,
  )
  if algo == "td3":
    return TD3(**common, learning_rate=1e-3, action_noise=action_noise, policy_kwargs=policy_kwargs)
  return DDPG(**common, learning_rate=1e-3, action_noise=action_noise, policy_kwargs=policy_kwargs)


def main() -> None:
  args = parse_args()

  log_dir = Path(args.log_dir) if args.log_dir else Path(
    f"logs/gym_mjlab_{args.algo}/{dt.datetime.now():%Y-%m-%d_%H-%M-%S}"
  )
  log_dir.mkdir(parents=True, exist_ok=True)
  print(f"[train_gym_mjlab] algo={args.algo} task={args.task} logging to {log_dir}")
  print(f"[train_gym_mjlab] num_envs={args.num_envs} total_timesteps={args.total_timesteps} "
        f"difficulty={args.difficulty} device={args.device}")

  raw_env = JavisMjlabVecEnv(
    num_envs=args.num_envs, rough=(args.task == "rough"), difficulty=args.difficulty,
    device=args.device, seed=args.seed,
  )
  vec_env = VecMonitor(raw_env, filename=str(log_dir / "monitor"))

  model = _build_model(args.algo, vec_env, args)
  model.set_logger(configure(str(log_dir), ["stdout", "csv"]))

  while model.num_timesteps < args.total_timesteps:
    chunk = min(args.save_interval_steps, args.total_timesteps - model.num_timesteps)
    model.learn(total_timesteps=chunk, reset_num_timesteps=False, log_interval=1)
    ckpt_path = log_dir / f"model_{model.num_timesteps}"
    model.save(str(ckpt_path))
    print(f"[train_gym_mjlab] saved checkpoint: {ckpt_path}.zip ({model.num_timesteps} steps)")

  final_path = log_dir / "model_final"
  model.save(str(final_path))
  print(f"[train_gym_mjlab] done. final model: {final_path}.zip")

  vec_env.close()

  if not args.no_plot:
    print(f"[train_gym_mjlab] plotting -> {log_dir}")
    result = subprocess.run([sys.executable, "scripts/plot_gym_results.py", "--log-dir", str(log_dir)])
    if result.returncode != 0:
      print(f"[train_gym_mjlab] plotting failed (exit {result.returncode}) -- re-run by hand: "
            f".venv/bin/python scripts/plot_gym_results.py --log-dir {log_dir}")

  if not args.no_video:
    print(f"[train_gym_mjlab] recording video -> {log_dir}")
    result = subprocess.run([
      sys.executable, "scripts/record_gym_mjlab_video.py",
      "--checkpoint", str(final_path) + ".zip",
      "--algo", args.algo,
      "--task", args.task,
      "--out", str(log_dir / "scenarios.mp4"),
      "--seconds", str(args.video_seconds),
      "--device", args.device,
    ])
    if result.returncode != 0:
      print(f"[train_gym_mjlab] video recording failed (exit {result.returncode}) -- re-run by hand: "
            f".venv/bin/python scripts/record_gym_mjlab_video.py --checkpoint {final_path}.zip "
            f"--algo {args.algo} --task {args.task} --out {log_dir}/scenarios.mp4")


if __name__ == "__main__":
  main()
