#!/usr/bin/env python3
"""Apply the project's rolling-mean convergence criterion to a
`scripts/scratch_ppo.py` / `scripts/scratch_sac.py` run's `eval/score_mean`
trajectory, and print the full trajectory for manual judgment alongside it.

Criterion (same one used throughout the SRL PPO/SAC investigation, see
README.md / configs/srl/*.yaml): trailing 5-eval-point rolling mean;
"converged" the first time 3 consecutive rolling-mean-to-rolling-mean
percent changes are each < 5% in magnitude. Known to false-positive during a
smooth, still-climbing or still-declining trend (three small steps in a row
can all be <5% of a slowly-moving base even while the direction is
unambiguous) -- this script flags that case heuristically (checks whether
the declared "converged" value sits within a longer monotonic run) but a
human should look at the printed trajectory, not just the verdict.

Usage:
    .venv/bin/python scripts/analyze_scratch_run.py runs/scratch_ppo/metrics.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_eval_scores(path: Path) -> list[tuple[int, float]]:
  points: list[tuple[int, float]] = []
  with open(path) as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      row = json.loads(line)
      if row.get("tag") == "eval/score_mean":
        points.append((int(row["step"]), float(row["value"])))
  points.sort(key=lambda p: p[0])
  return points


def rolling_mean(values: list[float], window: int) -> list[float | None]:
  out: list[float | None] = []
  for i in range(len(values)):
    if i + 1 < window:
      out.append(None)
    else:
      out.append(sum(values[i + 1 - window:i + 1]) / window)
  return out


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("metrics_jsonl", type=Path)
  ap.add_argument("--window", type=int, default=5)
  ap.add_argument("--threshold-pct", type=float, default=5.0)
  ap.add_argument("--consecutive", type=int, default=3)
  args = ap.parse_args()

  points = load_eval_scores(args.metrics_jsonl)
  if len(points) < args.window + args.consecutive:
    print(f"Only {len(points)} eval points found -- need at least "
          f"{args.window + args.consecutive} to evaluate the criterion. Printing what exists:")
    for step, score in points:
      print(f"  step={step:>12,}  eval/score_mean={score:.4f}")
    sys.exit(0)

  steps = [p[0] for p in points]
  scores = [p[1] for p in points]
  rmeans = rolling_mean(scores, args.window)

  print(f"=== {args.metrics_jsonl} ===")
  print(f"{len(points)} eval points, steps {steps[0]:,} .. {steps[-1]:,}")
  print(f"{'step':>14} {'score_mean':>12} {'roll_mean5':>12} {'%chg':>8}")
  pct_changes: list[float | None] = [None] * len(points)
  for i in range(len(points)):
    rm = rmeans[i]
    pct = None
    if i > 0 and rmeans[i - 1] is not None and rm is not None and rmeans[i - 1] != 0:
      pct = 100.0 * (rm - rmeans[i - 1]) / abs(rmeans[i - 1])
      pct_changes[i] = pct
    rm_str = f"{rm:.4f}" if rm is not None else "n/a"
    pct_str = f"{pct:+.2f}%" if pct is not None else ""
    print(f"{steps[i]:>14,} {scores[i]:>12.4f} {rm_str:>12} {pct_str:>8}")

  # Find first index where `consecutive` in-a-row rolling-mean pct changes
  # are all < threshold in magnitude.
  converged_at = None
  for i in range(len(points)):
    window_pcts = pct_changes[i - args.consecutive + 1:i + 1] if i - args.consecutive + 1 >= 0 else []
    if len(window_pcts) == args.consecutive and all(
      p is not None and abs(p) < args.threshold_pct for p in window_pcts
    ):
      converged_at = i
      break

  print()
  if converged_at is None:
    print(f"NOT converged by this criterion (no {args.consecutive} consecutive "
          f"rolling-mean changes all < {args.threshold_pct}% found).")
    print(f"Final rolling mean: {rmeans[-1]}, final raw score: {scores[-1]:.4f}")
  else:
    step, score, rm = steps[converged_at], scores[converged_at], rmeans[converged_at]
    print(f"CONVERGED (mechanically) at step={step:,}, eval/score_mean={score:.4f}, "
          f"rolling_mean={rm:.4f}")

    # Heuristic false-positive flag: is the declared convergence point part of
    # a longer monotonic run in the rolling mean (climbing or declining
    # smoothly through it), rather than a genuine flat plateau?
    lo = max(0, converged_at - args.consecutive - 2)
    hi = min(len(points), converged_at + 3)
    window_vals = [rmeans[j] for j in range(lo, hi) if rmeans[j] is not None]
    if len(window_vals) >= 4:
      diffs = [window_vals[k + 1] - window_vals[k] for k in range(len(window_vals) - 1)]
      same_sign = all(d > 0 for d in diffs) or all(d < 0 for d in diffs)
      if same_sign and max(abs(d) for d in diffs) > 1e-9:
        direction = "climbing" if diffs[0] > 0 else "declining"
        print(f"  ** LIKELY FALSE POSITIVE ** rolling mean is monotonically {direction} "
              f"through the surrounding points ({[round(v, 4) for v in window_vals]}) -- "
              "this reads as a still-moving smooth trend, not a real plateau. Inspect "
              "the full trajectory above before trusting this verdict.")

    # Report the post-convergence trend too (does it actually HOLD afterward?)
    post = scores[converged_at:]
    if len(post) >= 3:
      print(f"  post-convergence eval points (n={len(post)}): min={min(post):.4f} "
            f"max={max(post):.4f} mean={sum(post)/len(post):.4f} last={post[-1]:.4f}")


if __name__ == "__main__":
  main()
