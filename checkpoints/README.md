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

## `javis_payload_rough/model_400.*` -- current, mid-training

`Javis-Payload-Rough`, iteration **400/1000, still training** at the time this
was copied out (2026-08-10) -- pushed early specifically to try keyboard
teleop (`../scripts/teleop_keyboard.py`) against it, not as a finished
deliverable. Trained at **12288 envs**, `CURRICULUM_ENABLED = False`
(`javis/balance_task.py` -- full hard DR envelope from iteration 0, no ramp)
and the fixed CoM-aware feasibility filter (previous commit).

At iteration 413 (closest logged point to this checkpoint):
`Train/mean_reward` ~55, `Train/mean_episode_length` ~1688/2000 (84%),
`Curriculum/total_mass_kg` ~17.5 (sampling the full 3-15 kg chassis + 0-10 kg
payload range every reset, per `CURRICULUM_ENABLED = False`). Iteration-to-
iteration reward is genuinely noisy (the adaptive-KL learning rate schedule
swings ~1e-5 to ~3e-3 between consecutive iterations) on top of a clear rising
trend -- read a smoothed curve, not single points, if judging progress from
`Train/mean_reward` directly.

Terrain: always at maximum configured slope/roughness (`difficulty_range =
(1.0, 1.0)`), not ramped either -- see `_terrain_cfg` in `balance_task.py`.

**Replace this file once training finishes (or once it visibly plateaus) --
it is a live snapshot, not a result.**

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
scripts/train.sh flat            # or: scripts/train.sh rough
.venv/bin/python scripts/export_onnx.py --task flat \
    --checkpoint logs/rsl_rl/javis_payload_flat/<run>/model_<n>.pt
cp logs/rsl_rl/javis_payload_flat/<run>/model_<n>.{pt,onnx} checkpoints/javis_payload_flat/
cp logs/rsl_rl/javis_payload_flat/<run>/model_<n>_contract.json checkpoints/javis_payload_flat/
```

Try a checkpoint by hand before trusting the numbers:

```bash
scripts/teleop_keyboard.py --checkpoint checkpoints/javis_payload_rough/model_400.pt --task rough
scripts/watch_training.sh rough   # or: watch it live in the browser instead
```
