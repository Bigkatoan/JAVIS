#!/usr/bin/env python3
"""Plot PNG charts from a scripts/train_gym.py run -- no tensorboard, no wandb.

Reads the two kinds of CSV a run directory has:
  progress.csv          one row per PPO update (SB3's own logger.configure)
  monitor_<rank>.monitor.csv   one row per finished episode, per parallel env
                         (SB3's Monitor wrapper)

and writes PNGs next to them:
  reward.png             episode reward, from the monitor CSVs (ground truth,
                          not the training-time rollout average)
  episode_length.png     episode length, same source
  losses.png             value loss / policy gradient loss / entropy loss
  kl_clip.png             approx KL and clip fraction (PPO health signals)
  fps.png                 training throughput over time

Usage:
    .venv/bin/python scripts/plot_gym_results.py --log-dir logs/gym_ppo/2026-08-12_12-11-42
"""

from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# One hue per series, fixed order, distinguishable without relying on color
# alone (each is also a distinct line style where more than one series shares
# an axes). Kept local rather than pulled from a shared theme -- this script
# has exactly one consumer.
_BLUE = "#3b82f6"
_ORANGE = "#f97316"
_GREEN = "#22c55e"
_RED = "#ef4444"
_GRAY = "#6b7280"


def _read_progress(log_dir: Path) -> dict[str, np.ndarray]:
  path = log_dir / "progress.csv"
  if not path.exists():
    return {}
  with open(path, newline="") as f:
    rows = list(csv.DictReader(f))
  cols: dict[str, list[float]] = {}
  for row in rows:
    for k, v in row.items():
      if v == "" or v is None:
        continue
      try:
        cols.setdefault(k, []).append(float(v))
      except ValueError:
        continue
  return {k: np.array(v) for k, v in cols.items() if len(v) > 1}


def _read_monitors(log_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Concatenate every *.monitor.csv, sorted by wall-clock time.

  Two naming conventions, both matched by the same "*.monitor.csv" glob:
  scripts/train_gym.py's SubprocVecEnv writes one `monitor_<rank>.monitor.csv`
  per OS-process env; scripts/train_gym_mjlab.py's single-process VecMonitor
  (wrapping the GPU-batched mjlab env) writes just one `monitor.monitor.csv`.

  Returns (episode_time_s, reward, length), all sorted by time so the plot
  reads as "training progressed" even though episodes across parallel envs
  finish interleaved.
  """
  times, rewards, lengths = [], [], []
  for path in sorted(glob.glob(str(log_dir / "*.monitor.csv"))):
    with open(path, newline="") as f:
      next(f)  # SB3's Monitor header comment line, e.g. {"t_start": ...}
      for row in csv.DictReader(f):
        rewards.append(float(row["r"]))
        lengths.append(float(row["l"]))
        times.append(float(row["t"]))
  if not times:
    return np.array([]), np.array([]), np.array([])
  order = np.argsort(times)
  return np.array(times)[order], np.array(rewards)[order], np.array(lengths)[order]


def _smooth(y: np.ndarray, window: int) -> np.ndarray:
  if len(y) < window:
    return y
  kernel = np.ones(window) / window
  return np.convolve(y, kernel, mode="valid")


def _plot_series(x, y, xlabel, ylabel, title, out_path, color, window=None):
  fig, ax = plt.subplots(figsize=(8, 4.5))
  ax.plot(x, y, color=color, alpha=0.25 if window else 1.0, linewidth=1.2, label="raw")
  if window and len(y) >= window:
    y_smooth = _smooth(y, window)
    x_smooth = x[window - 1:]
    ax.plot(x_smooth, y_smooth, color=color, linewidth=2.0, label=f"moving avg ({window})")
    ax.legend(frameon=False)
  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)
  ax.set_title(title)
  ax.grid(True, alpha=0.25)
  ax.spines["top"].set_visible(False)
  ax.spines["right"].set_visible(False)
  fig.tight_layout()
  fig.savefig(out_path, dpi=150)
  plt.close(fig)
  print(f"  wrote {out_path}")


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--log-dir", type=str, required=True)
  args = p.parse_args()

  log_dir = Path(args.log_dir)
  if not log_dir.is_dir():
    raise SystemExit(f"not a directory: {log_dir}")

  print(f"[plot_gym_results] reading {log_dir}")
  progress = _read_progress(log_dir)
  ep_t, ep_r, ep_l = _read_monitors(log_dir)

  if ep_t.size:
    window = max(5, len(ep_r) // 50)
    _plot_series(ep_t, ep_r, "training time (s)", "episode reward",
                 "Episode reward", log_dir / "reward.png", _BLUE, window=window)
    _plot_series(ep_t, ep_l, "training time (s)", "episode length (control steps)",
                 "Episode length", log_dir / "episode_length.png", _GREEN, window=window)
  else:
    print("  no *.monitor.csv found (no episode finished yet?) -- skipping reward/length plots")

  # "Value loss" axis: PPO's train/value_loss, or SAC/TD3/DDPG's
  # train/critic_loss -- whichever the algorithm actually logged.
  # "Policy loss" axis: PPO's policy_gradient_loss/entropy_loss, or the
  # off-policy actor_loss/ent_coef_loss (SAC only; TD3/DDPG have no entropy
  # term). Plotting whichever subset exists keeps this one chart useful
  # across all four `--algo` choices instead of assuming PPO's column names.
  value_key = "train/value_loss" if "train/value_loss" in progress else "train/critic_loss"
  policy_keys = [
    k for k in ("train/policy_gradient_loss", "train/actor_loss", "train/entropy_loss", "train/ent_coef_loss")
    if k in progress
  ]
  if value_key in progress or policy_keys:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if value_key in progress:
      ax.plot(progress[value_key], color=_ORANGE, label=value_key.split("/")[1])
    ax2 = ax.twinx()
    for key, color in zip(policy_keys, (_BLUE, _GREEN, _GRAY, _RED)):
      ax2.plot(progress[key], color=color, label=key.split("/")[1], linestyle="--" if "entropy" in key else "-")
    ax.set_xlabel("gradient update")
    ax.set_ylabel(value_key.split("/")[1], color=_ORANGE)
    ax2.set_ylabel("policy-side loss")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="best")
    ax.set_title("Training losses")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(log_dir / "losses.png", dpi=150)
    plt.close(fig)
    print(f"  wrote {log_dir / 'losses.png'}")

  if "train/approx_kl" in progress or "train/clip_fraction" in progress:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if "train/approx_kl" in progress:
      ax.plot(progress["train/approx_kl"], color=_RED, label="approx KL")
    if "train/clip_fraction" in progress:
      ax.plot(progress["train/clip_fraction"], color=_BLUE, label="clip fraction")
    ax.set_xlabel("PPO update")
    ax.set_ylabel("value")
    ax.set_title("PPO health: KL divergence & clip fraction")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(log_dir / "kl_clip.png", dpi=150)
    plt.close(fig)
    print(f"  wrote {log_dir / 'kl_clip.png'}")

  if "time/fps" in progress:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(progress.get("time/total_timesteps", np.arange(len(progress["time/fps"]))),
            progress["time/fps"], color=_GREEN)
    ax.set_xlabel("total env steps")
    ax.set_ylabel("steps/sec")
    ax.set_title("Training throughput")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(log_dir / "fps.png", dpi=150)
    plt.close(fig)
    print(f"  wrote {log_dir / 'fps.png'}")

  print("[plot_gym_results] done")


if __name__ == "__main__":
  main()
