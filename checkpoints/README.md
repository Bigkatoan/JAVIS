# Checkpoints

> ⚠️ **`javis_payload_flat/model_1499.*` is now stale, on two counts.** It
> was trained before the control-rate retune (50 Hz → 100 Hz) and the added
> init-velocity/IMU-bias randomization -- the observation size changed with
> it (192 → 384), so this checkpoint's shapes no longer match
> `javis/balance_task.py` and `runner.load()` will fail on a shape mismatch.
> It also predates the 2026-08-12 CAD update (redesigned drivetrain,
> `body`/`wheel`/`wheel_2` → `base_link`/`wheel_left`/`wheel_right`, new mass
> budget), which changes the physics it was trained against even where shapes
> still happen to match. Kept as a historical record; train a fresh one
> (`scripts/train.sh`) before deploying anything.

Trained policies committed directly to the repo, small enough (a few MB) that
git LFS isn't needed. `logs/` (where training writes by default) stays
git-ignored — everything here was copied out deliberately.

## `javis_payload_rough/model_399.*` -- current

`Javis-Payload-Rough`, **400 iterations, 12288 envs**, trained 2026-08-12
against the redesigned CAD (`javis/robot_constants.py`: `base_link`/
`wheel_left`/`wheel_right`, direct-drive encoder mount, printed-part mass
re-measured at 1332.26 g). `CURRICULUM_ENABLED = False` -- full hard DR
envelope (chassis 3-15 kg, payload 0-10 kg, CoM offsets to the configured
extremes) from iteration 0, no ramp -- and the CoM-aware feasibility filter.
Wall-clock: ~17 min.

Final logged iteration: `Train/mean_reward` ~51, `Train/mean_episode_length`
~1561/2000 (78%), `Curriculum/total_mass_kg` ~17.3 (full-envelope sampling
every reset). Terminations were overwhelmingly `time_out` over `chassis_down`
(2.5:1 in the final batch) -- most episodes are surviving to the length cap,
not falling. Reward term breakdown at that point: `track_linear_velocity`
0.91/1.5, `track_angular_velocity` 0.56/1.0, `upright` 0.74/1.0, `is_alive`
0.39/0.5. Terrain always at maximum configured slope/roughness
(`difficulty_range = (1.0, 1.0)`), not ramped either -- see `_terrain_cfg` in
`balance_task.py`.

This supersedes the previous `model_400.*` (same iteration count, but trained
against the pre-2026-08-12 CAD -- physically stale, not just superseded by a
better score). Evaluate with `scripts/eval_payload_sweep.py` before trusting
it beyond a smoke test; it has not touched real hardware.

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
scripts/teleop_keyboard.py --checkpoint checkpoints/javis_payload_rough/model_399.pt --task rough
scripts/watch_training.sh rough   # or: watch it live in the browser instead
```
