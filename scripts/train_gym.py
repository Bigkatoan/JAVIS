#!/usr/bin/env python3
"""Train the plain-MuJoCo/Gymnasium JAVIS balance env with stable-baselines3.

The companion to `javis/balance_task.py` (mjlab + rsl-rl + wandb/tensorboard):
same robot, same domain randomization, same reward -- see javis/gym_env.py's
docstring for exactly what's ported and what's simplified -- but no GPU batching,
no wandb, no tensorboard. Logging is stdout + a CSV (SB3's own progress.csv,
written by `stable_baselines3.common.logger`) plus one `monitor.csv` per parallel
env (SB3's `Monitor` wrapper, per-episode reward/length). Nothing else touches
the network or writes anywhere but `--log-dir`.

`javis/gym_env.py`'s `JavisBalanceEnv` is a plain Gymnasium env with a
continuous Box(-1, 1, (2,)) action space, so it needs no algorithm-specific
plumbing -- this script is what picks and configures one. `--algo` selects
between on-policy PPO and the off-policy SAC/TD3/DDPG, each with its own
sensible defaults (off-policy algorithms use a replay buffer and different
relevant knobs -- buffer_size/tau/train_freq instead of n_steps/gae_lambda/
clip_range -- see `_build_model` below for exactly what differs).

When training finishes it automatically plots the CSVs (scripts/plot_gym_results.py)
and records a scenario video of the trained checkpoint (scripts/record_gym_video.py,
flat/ramp/rough terrain x a couple of noise levels) into --log-dir -- pass --no-plot
/ --no-video to skip either (e.g. for a quick unattended sweep where you'll plot/
record afterward yourself, batched across runs).

Usage:
    .venv/bin/python scripts/train_gym.py --algo ppo --num-envs 16 --total-timesteps 2000000
    .venv/bin/python scripts/train_gym.py --algo sac --num-envs 8  --total-timesteps 500000
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import DDPG, PPO, SAC, TD3
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv

from javis.gym_env import JavisBalanceEnv
from javis.sim_config import JavisDomainCfg

ON_POLICY = ("ppo",)
OFF_POLICY = ("sac", "td3", "ddpg")


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--algo", choices=(*ON_POLICY, *OFF_POLICY), default="ppo")
  p.add_argument("--num-envs", type=int, default=max(4, min(24, (os.cpu_count() or 8) - 4)),
                  help="parallel subprocess envs (default: cpu_count - 4, clamped to [4, 24])")
  p.add_argument("--total-timesteps", type=int, default=2_000_000)
  p.add_argument("--n-steps", type=int, default=24,
                  help="PPO only: rollout length per env before each update")
  p.add_argument("--buffer-size", type=int, default=300_000,
                  help="SAC/TD3/DDPG only: replay buffer capacity")
  p.add_argument("--learning-starts", type=int, default=5_000,
                  help="SAC/TD3/DDPG only: random-action warmup steps before training starts")
  p.add_argument("--difficulty", type=float, default=1.0, help="DR level in [0, 1]; 1.0 = full hard envelope")
  p.add_argument("--seed", type=int, default=42)
  p.add_argument("--log-dir", type=str, default=None,
                  help="default: logs/gym_<algo>/<timestamp>")
  p.add_argument("--save-interval-steps", type=int, default=200_000,
                  help="save a checkpoint every this many total env steps")
  p.add_argument("--no-plot", action="store_true", help="skip auto-plotting after training")
  p.add_argument("--no-video", action="store_true", help="skip auto-recording a scenario video after training")
  p.add_argument("--video-seconds", type=float, default=6.0, help="per scenario, see record_gym_video.py")
  return p.parse_args()


def make_env(rank: int, seed: int, difficulty: float, log_dir: Path):
  def _init():
    env = JavisBalanceEnv(domain=JavisDomainCfg(), difficulty=difficulty, seed=seed + rank)
    env = Monitor(env, filename=str(log_dir / f"monitor_{rank}"))
    return env

  return _init


def _build_model(algo: str, vec_env: VecEnv, args: argparse.Namespace) -> BaseAlgorithm:
  action_dim = vec_env.action_space.shape[0]

  if algo == "ppo":
    buffer_size = args.n_steps * args.num_envs
    batch_size = max(64, buffer_size // 4)
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
      verbose=1,
    )

  # Off-policy: one shared replay buffer fed by every parallel env. SB3's own
  # recommendation for vectorized off-policy training is gradient_steps ==
  # num_envs at train_freq=(1, "step"), so one env-step collected across the
  # whole VecEnv triggers one gradient step per env -- keeps the update-to-
  # data ratio the same as running a single env at train_freq=1.
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

  # TD3/DDPG are deterministic-policy algorithms: unlike SAC's entropy-driven
  # stochastic policy, they need explicit exploration noise added to actions
  # during rollout collection or they never explore past the initial policy.
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
    f"logs/gym_{args.algo}/{dt.datetime.now():%Y-%m-%d_%H-%M-%S}"
  )
  log_dir.mkdir(parents=True, exist_ok=True)
  print(f"[train_gym] algo={args.algo} logging to {log_dir}")
  print(f"[train_gym] num_envs={args.num_envs} total_timesteps={args.total_timesteps} "
        f"difficulty={args.difficulty}")

  vec_env = SubprocVecEnv(
    [make_env(i, args.seed, args.difficulty, log_dir) for i in range(args.num_envs)]
  )

  model = _build_model(args.algo, vec_env, args)
  model.set_logger(configure(str(log_dir), ["stdout", "csv"]))

  while model.num_timesteps < args.total_timesteps:
    chunk = min(args.save_interval_steps, args.total_timesteps - model.num_timesteps)
    model.learn(
      total_timesteps=chunk,
      reset_num_timesteps=False,
      log_interval=1,
    )
    ckpt_path = log_dir / f"model_{model.num_timesteps}"
    model.save(str(ckpt_path))
    print(f"[train_gym] saved checkpoint: {ckpt_path}.zip ({model.num_timesteps} steps)")

  final_path = log_dir / "model_final"
  model.save(str(final_path))
  print(f"[train_gym] done. final model: {final_path}.zip")

  vec_env.close()

  # Auto-plot + auto-video, each a separate subprocess (reloads the saved
  # checkpoint fresh, same as running the two scripts by hand) so a failure
  # in either -- a rendering issue, a missing display lib -- can't take the
  # training run's own exit status down with it.
  if not args.no_plot:
    print(f"[train_gym] plotting -> {log_dir}")
    result = subprocess.run(
      [sys.executable, "scripts/plot_gym_results.py", "--log-dir", str(log_dir)]
    )
    if result.returncode != 0:
      print(f"[train_gym] plotting failed (exit {result.returncode}) -- "
            f"re-run by hand: .venv/bin/python scripts/plot_gym_results.py --log-dir {log_dir}")

  if not args.no_video:
    print(f"[train_gym] recording scenario video -> {log_dir}")
    result = subprocess.run([
      sys.executable, "scripts/record_gym_video.py",
      "--runs", f"{args.algo}={log_dir}",
      "--out-dir", str(log_dir),
      "--seconds", str(args.video_seconds),
    ])
    if result.returncode != 0:
      print(f"[train_gym] video recording failed (exit {result.returncode}) -- re-run by hand: "
            f".venv/bin/python scripts/record_gym_video.py --runs {args.algo}={log_dir} --out-dir {log_dir}")


if __name__ == "__main__":
  main()
