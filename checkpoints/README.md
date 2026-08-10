# Checkpoints

> ⚠️ **`javis_payload_flat/model_1499.*` is now stale.** It was trained before
> the control-rate retune (50 Hz → 100 Hz, matching the real hardware's
> measured ceiling) and the added init-velocity randomization / IMU bias
> noise. The observation size changed with it (192 → 384 for the actor, 12 →
> 24 history frames), so this checkpoint's shapes no longer match
> `javis/balance_task.py` and it **cannot be loaded against current code** --
> `runner.load()` will fail on a shape mismatch. Kept here as a historical
> record of the mass-model + drivetrain-gain work it validated; train a fresh
> one (`scripts/train.sh`) before deploying anything.

Trained policies committed directly to the repo, small enough (a few MB) that
git LFS isn't needed. `logs/` (where training writes by default) stays
git-ignored — everything here was copied out deliberately.

## `javis_payload_flat/model_1499.*`

`Javis-Payload-Flat`, 1500 PPO iterations, 4096 envs, RTX 3090, ~13 min
wall-clock. Trained 2026-08-10 with the retuned ODrive drivetrain gain
(`javis/sim_config.py DrivetrainCfg`, `vel_gain=15.0`) at a 400 Hz physics
timestep — **not** the gain currently on the real board (see
`../SIM2REAL.md` sec 3 before flashing anything).

Curriculum reached difficulty **0.96/1.0**. Evaluated in
`../scripts/eval_payload_sweep.py`:

- Mass alone: **30/30** grid cells at 100% survival up to 30.9 kg total
  (2.8x the ~11 kg nominal robot).
- Centre-of-mass offset (the harder axis): **13/25** cells at >=90%
  survival. The failure boundary is a roughly constant ~7 degree
  equilibrium lean, not the offset distance itself — a large offset mounted
  high survives where a small one mounted low does not.

**This is a smoke-test checkpoint, not a finished policy.** It demonstrates
the pipeline and the retuned drivetrain both work end-to-end; it has not been
tuned to convergence, has never seen rough terrain, and has not touched real
hardware. Treat it as a starting point / regression baseline, not something
to flash and trust.

Files:

| File | What it's for |
|---|---|
| `model_1499.pt` | Full rsl-rl checkpoint (actor + critic + optimizer state). Resume training from it, or re-export ONNX after further training. |
| `model_1499.onnx` | Actor only, normalization baked in. What the Jetson driver actually loads. |
| `model_1499_contract.json` | The observation layout (term-major! see the note inside), action meaning/units, and control rate this ONNX graph expects. Read this before wiring up a driver — see `../DEPLOY.md`. |

Regenerate or replace with a better checkpoint:

```bash
scripts/train.sh flat            # or rough, once that's been trained
.venv/bin/python scripts/export_onnx.py --checkpoint logs/rsl_rl/javis_payload_flat/<run>/model_<n>.pt
cp logs/rsl_rl/javis_payload_flat/<run>/model_<n>.{pt,onnx} checkpoints/javis_payload_flat/
cp logs/rsl_rl/javis_payload_flat/<run>/model_<n>_contract.json checkpoints/javis_payload_flat/
```
