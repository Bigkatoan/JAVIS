#!/usr/bin/env python3
"""Train JAVIS through github.com/Bigkatoan/SRL instead of stable-baselines3.

SRL's CLI (`srl-train`) resolves `--env` via plain `gymnasium.make(id)` for
the default env-type -- it has no notion of this repo. Importing
`javis.gym_env` registers `Javis-Balance-v0` (see that module's bottom) with
gymnasium's own registry, which is all `gym.make` needs; this script just
does that import and then hands off to SRL's own CLI `main()` unchanged, so
every `srl-train` flag documented at https://bigkatoan.github.io/SRL/cli
works exactly as it would for any other Gymnasium env.

Usage (identical to srl-train, plus the env is already registered):
    .venv/bin/python scripts/train_srl.py \\
        --config configs/srl/javis_balance_ppo.yaml \\
        --env Javis-Balance-v0 --algo ppo --device cuda
"""

from __future__ import annotations

import sys

import javis.gym_env  # noqa: F401  (side effect: registers Javis-Balance-v0)
from srl.cli.train import main

if __name__ == "__main__":
  raise SystemExit(main(sys.argv[1:]))
