#!/usr/bin/env bash
# Launch JAVIS payload/balance training with sane defaults.
#
# Usage:
#   scripts/train.sh                          # rough terrain, the real target
#   scripts/train.sh flat                     # flat only, fast iteration
#   scripts/train.sh rough 5000                # override iteration count
#   scripts/train.sh rough 3000 2048           # override envs too
#   NUM_ENVS=2048 scripts/train.sh rough        # same, via env var
#   scripts/train.sh rough --agent.resume       # resume the latest checkpoint
#
# Extra arguments after the positional ones are passed straight through to
# `train`, e.g. `--agent.resume` or `--agent.wandb-project foo`.
#
# Timing measured on an RTX 3090 (2026-08-10):
#   flat,  4096 envs   ~0.4s/iteration  -> 3000 iters ~ 20 min
#   rough, 4096 envs   ~2.5s/iteration  -> 3000 iters ~ 2 hours
# Rough is slower mainly from the terrain generator (raycasting/height lookups
# against the heightfield/mesh terrain), not from the robot itself. Drop
# NUM_ENVS if you're VRAM-constrained -- iteration time scales with envs, wall
# clock to a given curriculum level roughly does not.
#
# What "done" looks like: watch `Curriculum/payload_difficulty` in
# tensorboard climb toward 1.0. The flat smoke test in this branch reached
# 0.96 at 1500 iterations (4096 envs) with the retuned drivetrain gain
# (javis/sim_config.py DrivetrainCfg) -- rough terrain adds a second,
# independent ramp (`Curriculum/terrain_levels/*`) on top of that, so expect
# it to need more iterations to fully open up both at once.
set -euo pipefail

TERRAIN="${1:-rough}"
ITERATIONS="${2:-${ITERATIONS:-3000}}"
NUM_ENVS="${3:-${NUM_ENVS:-4096}}"
shift $(( $# < 3 ? $# : 3 )) || true

case "$TERRAIN" in
  flat)  TASK="Javis-Payload-Flat" ;;
  rough) TASK="Javis-Payload-Rough" ;;
  *)     echo "usage: $0 [flat|rough] [iterations] [num_envs] [-- extra train args]" >&2
         exit 1 ;;
esac

cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "[train.sh] task=$TASK iterations=$ITERATIONS num_envs=$NUM_ENVS extra_args=$*"
exec .venv/bin/train "$TASK" \
  --agent.max-iterations "$ITERATIONS" \
  --agent.logger tensorboard \
  --env.scene.num-envs "$NUM_ENVS" \
  "$@"
