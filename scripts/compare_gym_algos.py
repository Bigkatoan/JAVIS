#!/usr/bin/env python3
"""Overlay scripts/train_gym.py runs from different --algo choices on one chart.

Reads each run's progress.csv (SB3's own logger, one row per gradient-update
batch) and plots `rollout/ep_rew_mean` / `rollout/ep_len_mean` against
`time/total_timesteps` -- env steps, not wall-clock time, since PPO and the
off-policy algorithms take very different wall-clock per step (PPO is
physics-bound here, SAC/TD3/DDPG are gradient-update-bound) and total env
steps is the fair, standard axis for a sample-efficiency comparison.

Usage:
    .venv/bin/python scripts/compare_gym_algos.py \
        --runs ppo=logs/gym_compare/ppo sac=logs/gym_compare/sac \
               td3=logs/gym_compare/td3 ddpg=logs/gym_compare/ddpg \
        --out-dir logs/gym_compare
"""

from __future__ import annotations

import argparse
import math
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_COLORS = {"ppo": "#3b82f6", "sac": "#f97316", "td3": "#22c55e", "ddpg": "#a855f7"}
_FALLBACK_COLORS = ["#3b82f6", "#f97316", "#22c55e", "#a855f7", "#ef4444", "#06b6d4"]


def _read_progress(log_dir: Path) -> dict[str, np.ndarray]:
  """Parse progress.csv into equal-length, row-aligned columns (NaN for gaps).

  SB3 doesn't log every column on every row -- e.g. `rollout/ep_rew_mean`
  is absent until the first episode finishes, while `time/total_timesteps`
  is on every row from the start. Keeping columns row-aligned (rather than
  dropping empty cells, which desyncs one column's length from another's) is
  what lets callers zip x/y from two different columns safely.
  """
  path = log_dir / "progress.csv"
  if not path.exists():
    return {}
  with open(path, newline="") as f:
    rows = list(csv.DictReader(f))
  fieldnames = rows[0].keys() if rows else []
  cols: dict[str, list[float]] = {k: [] for k in fieldnames}
  for row in rows:
    for k in fieldnames:
      v = row.get(k)
      try:
        cols[k].append(float(v) if v not in ("", None) else math.nan)
      except ValueError:
        cols[k].append(math.nan)
  return {k: np.array(v) for k, v in cols.items()}


def _smooth(y: np.ndarray, window: int) -> np.ndarray:
  if len(y) < window:
    return y
  kernel = np.ones(window) / window
  return np.convolve(y, kernel, mode="valid")


def _parse_runs(items: list[str]) -> dict[str, Path]:
  out = {}
  for item in items:
    if "=" not in item:
      raise SystemExit(f"--runs entries must be name=path, got: {item}")
    name, path = item.split("=", 1)
    out[name] = Path(path)
  return out


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--runs", nargs="+", required=True, help="name=log_dir pairs, one per algorithm")
  p.add_argument("--out-dir", type=str, required=True)
  p.add_argument("--smooth-window", type=int, default=5)
  args = p.parse_args()

  runs = _parse_runs(args.runs)
  out_dir = Path(args.out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  data = {}
  for name, log_dir in runs.items():
    progress = _read_progress(log_dir)
    if "time/total_timesteps" not in progress:
      print(f"[compare_gym_algos] {name}: no progress.csv / no total_timesteps column, skipping")
      continue
    data[name] = progress
    print(f"[compare_gym_algos] {name}: {log_dir} -- {len(progress['time/total_timesteps'])} logged updates, "
          f"final total_timesteps={progress['time/total_timesteps'][-1]:.0f}")

  if not data:
    raise SystemExit("no runs had usable progress.csv data")

  def _plot(metric_key: str, ylabel: str, title: str, out_name: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    any_plotted = False
    for i, (name, progress) in enumerate(data.items()):
      if metric_key not in progress:
        continue
      x = progress["time/total_timesteps"]
      y = progress[metric_key]
      valid = ~(np.isnan(x) | np.isnan(y))
      x, y = x[valid], y[valid]
      if len(y) == 0:
        continue
      color = _COLORS.get(name, _FALLBACK_COLORS[i % len(_FALLBACK_COLORS)])
      w = min(args.smooth_window, len(y))
      y_s = _smooth(y, w)
      x_s = x[w - 1:] if w > 1 else x
      ax.plot(x_s, y_s, color=color, linewidth=2.0, label=name)
      any_plotted = True
    if not any_plotted:
      plt.close(fig)
      return
    ax.set_xlabel("total env steps")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out_path = out_dir / out_name
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")

  _plot("rollout/ep_rew_mean", "episode reward (rollout mean)",
        "Algorithm comparison: episode reward vs. env steps", "comparison_reward.png")
  _plot("rollout/ep_len_mean", "episode length (control steps, rollout mean)",
        "Algorithm comparison: episode length vs. env steps", "comparison_episode_length.png")

  def _last_valid(progress: dict[str, np.ndarray], key: str) -> float:
    if key not in progress:
      return math.nan
    arr = progress[key][~np.isnan(progress[key])]
    return float(arr[-1]) if len(arr) else math.nan

  # Summary table: last logged rollout mean per algorithm, plus wall time.
  summary_path = out_dir / "comparison_summary.csv"
  rows = []
  for name, progress in data.items():
    rows.append((
      name,
      _last_valid(progress, "time/total_timesteps"),
      _last_valid(progress, "rollout/ep_rew_mean"),
      _last_valid(progress, "rollout/ep_len_mean"),
      _last_valid(progress, "time/fps"),
      _last_valid(progress, "time/time_elapsed"),
    ))

  with open(summary_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["algo", "total_timesteps", "final_ep_rew_mean", "final_ep_len_mean",
                "final_fps", "wall_clock_s"])
    for name, steps, rew, length, fps, wall in rows:
      w.writerow([name, f"{steps:.0f}", f"{rew:.2f}", f"{length:.2f}", f"{fps:.0f}", f"{wall:.0f}"])
  print(f"  wrote {summary_path}")

  print("\n=== summary ===")
  print(f"{'algo':<8}{'steps':>10}{'ep_rew_mean':>14}{'ep_len_mean':>14}{'fps':>8}{'wall_s':>10}")
  for name, steps, rew, length, fps, wall in rows:
    print(f"{name:<8}{steps:>10.0f}{rew:>14.2f}{length:>14.2f}{fps:>8.0f}{wall:>10.0f}")


if __name__ == "__main__":
  main()
