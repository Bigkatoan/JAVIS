#!/usr/bin/env bash
# Watch a training run live, in the browser -- the web viewer you're
# remembering. mjlab already ships it: `play --viewer viser` opens a Viser
# (browser-based) 3D view of the robot with a checkpoint dropdown that lists
# every *.pt file in the run's log directory, newest first. Training keeps
# writing new checkpoints (every `save_interval` iterations) while this stays
# open -- pick a newer one from the dropdown and it hot-swaps the policy with
# no restart. There's also a live reward-term bar chart and (for this task)
# the target-vs-current velocity arrows from javis/tasks/velocity/mdp's
# debug_vis, since Javis-Payload-* leaves debug_vis=True on the twist command.
#
# This script just finds the latest run directory and its latest checkpoint
# so you don't have to hunt through logs/rsl_rl/ for the exact path, and
# forces the web viewer explicitly (mjlab's own "auto" mode already picks it
# whenever $DISPLAY isn't set, which is the common case when training runs
# over SSH -- but forcing it means this works the same either way).
#
# Usage:
#   scripts/watch_training.sh                 # rough, latest run, latest checkpoint
#   scripts/watch_training.sh flat             # flat task instead
#   scripts/watch_training.sh rough 2026-08-10_11-03-29   # a specific run
#
# The browser URL is printed by viser itself once it starts (default
# http://localhost:8080). Training on a remote box over SSH? Forward the port
# instead of opening it to the world:
#   ssh -L 8080:localhost:8080 <user>@<training-box>
# then open http://localhost:8080 on your own machine.
set -euo pipefail

TERRAIN="${1:-rough}"
RUN="${2:-}"

case "$TERRAIN" in
  flat)  TASK="Javis-Payload-Flat";  EXPERIMENT="javis_payload_flat" ;;
  rough) TASK="Javis-Payload-Rough"; EXPERIMENT="javis_payload_rough" ;;
  *)     echo "usage: $0 [flat|rough] [run-timestamp]" >&2; exit 1 ;;
esac

cd "$(dirname "${BASH_SOURCE[0]}")/.."
RUN_ROOT="logs/rsl_rl/$EXPERIMENT"

if [[ ! -d "$RUN_ROOT" ]]; then
  echo "[watch] no runs found under $RUN_ROOT -- has $TASK been trained yet?" >&2
  exit 1
fi

if [[ -z "$RUN" ]]; then
  RUN="$(ls -t "$RUN_ROOT" | head -1)"
fi
RUN_DIR="$RUN_ROOT/$RUN"
if [[ ! -d "$RUN_DIR" ]]; then
  echo "[watch] run directory not found: $RUN_DIR" >&2
  echo "        available runs:" >&2
  ls -t "$RUN_ROOT" >&2
  exit 1
fi

# Sort by the training step in the filename (model_<n>.pt), not by mtime --
# a resumed run can write an old step's file after a newer one on disk.
LATEST_CKPT="$(ls "$RUN_DIR"/model_*.pt 2>/dev/null \
  | sed -E 's/.*model_([0-9]+)\.pt/\1 &/' | sort -n | tail -1 | cut -d' ' -f2)"
if [[ -z "$LATEST_CKPT" ]]; then
  echo "[watch] no checkpoints yet in $RUN_DIR -- training may still be on iteration 0" >&2
  exit 1
fi

echo "[watch] task=$TASK run=$RUN checkpoint=$(basename "$LATEST_CKPT")"
echo "[watch] opening the web viewer -- once it's up, newer checkpoints saved by"
echo "[watch] the still-running training job show up in its checkpoint dropdown"
echo "[watch] with no restart needed."
exec .venv/bin/play "$TASK" \
  --checkpoint-file "$LATEST_CKPT" \
  --viewer viser \
  --num-envs 1
