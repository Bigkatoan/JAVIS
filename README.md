# JAVIS

A 2-wheel differential-drive rover: Onshape CAD -> URDF -> [mjlab](https://github.com/mujocolab/mjlab)
(MuJoCo-Warp) simulation, targeting a Jetson-based robot with an Intel RealSense
D435 camera, an IMU, and MKS ODrive Mini wheel motor controllers.

## Layout

- `javis/config.json` — Onshape assembly URL + export settings for `onshape-to-robot`.
- `javis/robot.urdf` — generated from the Onshape assembly. Regenerate with
  `onshape-to-robot javis` after editing the CAD. Do not hand-edit.
- `javis/assets/` — meshes referenced by `robot.urdf` (STL for visuals/collision,
  `.part` for onshape-to-robot's own caching).
- `javis/robot_constants.py` — mjlab `EntityCfg` for the rover: loads
  `robot.urdf`, adds a free-floating root joint, wheel actuators, a chassis
  collision hull, a payload box, an IMU site + sensors, and a D435 camera.
- `javis/mass_model.py` — per-component-group mass and inertia model. Groups
  the 343 chassis parts by what they physically are (battery / printed PLA /
  Jetson / camera / ODrive boards / IMU / fasteners) and precomputes each
  group's unit-mass moments, which makes the fused body inertial exactly linear
  in the group masses. See **Mass model** below.
- `javis/sim_config.py` — every domain-randomization range in one editable
  place, each with an "easy" and a "hard" value that the curriculum
  interpolates between.
- `javis/mdp/` — task-specific MDP terms: payload/mass randomization
  (`events.py`), the simulated ODrive PI velocity loop (`actions.py`), balance
  rewards that tolerate a leaning equilibrium (`rewards.py`), privileged critic
  observations (`observations.py`), and the difficulty ramp (`curriculums.py`).
- `javis/balance_task.py` — the payload/balance RL tasks. See **Training a
  policy that survives a changing payload** below.
- `scripts/view_robot.py` — builds an mjlab `Scene` (flat ground plane + the
  rover) and opens it in MuJoCo's interactive viewer.
- `scripts/calibrate_actuator.py` — derives `WHEEL_ACTUATOR_CFG`'s
  damping/effort_limit either from motor datasheet numbers or by fitting
  against a real step-response log recorded from the physical robot. See
  **Calibrating the wheel actuators** below.
- `javis/velocity_task.py` — the RL task: a manager-based mjlab env that
  trains a policy to track a commanded (linear, angular) velocity while
  staying upright. See **Training a balance/velocity policy** below.
- `javis/tasks.py` — registers `Javis-Velocity-Flat` with mjlab so it shows
  up in `list-envs`/`train`/`play`, auto-loaded via the `mjlab.tasks` entry
  point in `pyproject.toml` (see **Setup**).
- `setup.bash` / `.env` — Onshape API credentials, loaded as environment
  variables (see **Setup**).
- `SIM2REAL.md` — detailed checklist of everything to measure/characterize
  on the physical robot for the simulation to transfer with minimal manual
  tuning.

## Setup

```bash
cp .env.example .env      # fill in your Onshape API key/secret
source setup.bash         # exports ONSHAPE_* into the shell
source venv/bin/activate  # or call venv/bin/python directly
pip install -e . --no-deps  # registers the javis package + its mjlab task
```

`.env` holds real credentials and is git-ignored — never commit it.
`requirements.txt` is a full freeze of the working `venv` (includes ROS2 /
Isaac ROS packages, which also require the underlying ROS2 system install on
a Jetson — pip alone won't reproduce those on a fresh machine).

### Workstation (training / evaluation)

Training runs on an x86_64 + CUDA machine, not the Jetson. `requirements.txt`
is the aarch64 freeze and will not install there; use `requirements-sim.txt`:

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements-sim.txt
.venv/bin/pip install -e . --no-deps
```

`python3.11 -m venv .venv` failing with an `ensurepip` error (`Command
'[...python3.11", "-m", "ensurepip"...]' returned non-zero exit status 1`)?
Debian/Ubuntu's `python3.11` package ships without `ensurepip` built in --
install the missing piece with `sudo apt install python3.11-venv` and re-run
the `venv` command above.

No `python3.11` on `PATH` at all, or no `apt`/root access to fix the above?
`python3 -m venv .venv` works too, as long as that `python3` is Python 3.10
or 3.11 (`requires-python = ">=3.10"` in `pyproject.toml`) -- but
`requirements-sim.txt`'s exact pins don't all ship Python 3.10 wheels (e.g.
`contourpy==1.3.3`), so on a 3.10 interpreter use `pip install -e '.[sim]'`
for unpinned versions instead of `-r requirements-sim.txt`. Last resort with
no `apt` access and no other Python 3.11 available: `pip install --user
virtualenv && python3.11 -m virtualenv .venv` creates the same `.venv/bin/`
layout every command on this page assumes, without needing `ensurepip`.

Equivalently `pip install -e '.[sim]'` for unpinned versions. Verified on an
RTX 3090 with mjlab 1.5.3 / mujoco 3.10.0 / torch 2.13.0+cu130.

The editable install (`pip install -e .`) is what makes `javis` a real,
importable package (`from javis.robot_constants import ...`) and, via the
`mjlab.tasks` entry point in `pyproject.toml`, what makes mjlab auto-discover
`javis/tasks.py` and register `Javis-Velocity-Flat` every time `mjlab` is
imported — run it once after cloning, or whenever `pyproject.toml` changes.

## Pulling CAD changes

```bash
venv/bin/onshape-to-robot javis
```

Re-exports `javis/robot.urdf` and `javis/assets/` from the Onshape assembly
in `javis/config.json`. `robot_constants.py` re-derives the mjlab scene from
`robot.urdf` on every import, so nothing else needs regenerating by hand.

## Viewing the robot

```bash
venv/bin/python scripts/view_robot.py
```

Opens the rover standing on a flat ground plane in MuJoCo's viewer. This is
the fastest way to confirm the CAD -> URDF -> mjlab pipeline still works
after a CAD change.

For headless use (no display), see mjlab's own `viz-nan`/viser-based viewer,
or render offscreen with `mujoco.Renderer`.

## Calibrating the wheel actuators

```bash
# from motor/gearbox datasheet numbers
venv/bin/python scripts/calibrate_actuator.py datasheet \
    --no-load-speed-rpm 200 --stall-torque-nm 1.2 --gear-ratio 10

# fit against a real step-response log (command a wheel via the ODrive,
# log t,cmd_vel,meas_vel -- see the script's module docstring for the CSV format)
venv/bin/python scripts/calibrate_actuator.py fit --log wheel_step_test.csv --plot fit.png
```

Both print a ready-to-paste `WHEEL_ACTUATOR_CFG` snippet. `fit` mode
simulates an isolated single wheel — using the wheel's actual mass/inertia
from `robot_constants.py` — under the logged command trace, and least-squares
fits `damping`/`effort_limit` to match the logged response. For
`effort_limit` to be identifiable, the log needs to include the motor
actually saturating (a large step); a log that never saturates only
constrains `damping`.

## Training a balance/velocity policy

```bash
venv/bin/list-envs                       # confirm "Javis-Velocity-Flat" is registered
venv/bin/train Javis-Velocity-Flat --agent.logger tensorboard
venv/bin/play  Javis-Velocity-Flat --checkpoint-file logs/rsl_rl/javis_velocity/<run>/model_<n>.pt
```

`javis/velocity_task.py` frames the control problem the same way mjlab/Isaac
Lab frame legged locomotion: a command-conditioned policy tracking
`(lin_vel_x, ang_vel_z)` while an `upright` reward keeps it balanced —
structurally closer to mjlab's own `tasks/cartpole` (single balance problem,
no feet/contact sensors) than to `tasks/velocity` (written for legged
robots), but it reuses `tasks/velocity`'s robot-agnostic pieces directly
(`UniformVelocityCommandCfg`, `track_linear_velocity`,
`track_angular_velocity`, `upright`) rather than duplicating them. See the
module docstring for the full reasoning, including why actions are wheel
*velocity* targets (matching the real ODrive's velocity-mode control) rather
than the torque actions a sim-only balance task would normally use.

Kept as the fixed-mass baseline to compare the payload tasks against. Verified
end-to-end (builds, steps, trains without NaNs) but never trained to
convergence or run on hardware.

## Mass model

`robot.urdf` has no usable mass data — no materials are assigned in the Onshape
assembly, so every link exports `mass="1e-9"`. `javis/mass_model.py` supplies it
instead, and does so per component group rather than as one averaged number,
because the battery is 39% of chassis mesh volume at ~5x the density of the
printed structure around it. Averaging puts the centre of mass in the wrong
place, and for a two-wheel balancing robot the centre of mass *is* the problem.

| group | volume | mass | source |
|---|---|---|---|
| `battery` | 1683 cm³ | 3.423 kg | weighed 2026-08-06 |
| `printed` | 2439 cm³ | 1.000 kg | one 1 kg PLA spool at ~15% infill → 0.41 g/cm³ |
| `jetson` | 70 cm³ | 0.176 kg | Orin Nano devkit catalog weight |
| `camera` | 36 cm³ | 0.072 kg | D435 datasheet |
| `odrive` | 22 cm³ | 0.070 kg | 2 × MKS xDrive Mini, estimated — not weighed |
| `imu` | 1 cm³ | 0.005 kg | BMX160 + BMP388 breakout |
| `hardware` | 19 cm³ | 0.148 kg | steel at 7.85 g/cm³ (checks out against a 6808-2RS catalog weight) |
| `electronics_misc` | 24 cm³ | 0.073 kg | connectors/capacitors at an assumed 3.0 g/cm³ |
| `wiring_misc` | — | 0.300 kg | harness and oddments, not in CAD, randomized 0–1 kg |
| each wheel | 1416 cm³ | 2.936 kg | weighed 2026-08-06 |

Chassis 5.27 kg, whole robot **11.14 kg**, CoM 0.2875 m above the floor and
1.4 mm off the axle fore/aft. Two things corroborate it: 1 kg of PLA over
2439 cm³ implies 0.41 g/cm³, which is what ~15% infill plus perimeters gives;
and the model reproduces the 0.01205 kg·m² wheel spin inertia that was measured
independently for the actuator calibration.

```bash
.venv/bin/python scripts/inspect_mass.py --check-model   # budget + balance envelope
.venv/bin/python scripts/verify_mass_model.py            # 32 independent checks
.venv/bin/python scripts/view_robot.py --color-by-group  # confirm mesh→group by eye
```

Because mass properties are linear in the group masses, `mass_model.fuse()`
produces an exact `(mass, ipos, iquat, inertia)` for any configuration with a
couple of batched matmuls — which is what lets every environment carry its own
physically consistent build with no MjSpec recompile.

## Training a policy that survives a changing payload

```bash
.venv/bin/train Javis-Payload-Flat  --agent.logger tensorboard
.venv/bin/train Javis-Payload-Rough --agent.logger tensorboard
.venv/bin/play  Javis-Payload-Flat --checkpoint-file logs/rsl_rl/javis_payload_flat/<run>/model_<n>.pt
```

`javis/balance_task.py` keeps the twist-tracking objective but randomizes what
the robot *is*: chassis 3–15 kg, payload 0–10 kg mounted anywhere from 5 to
60 cm up and ±12 cm off-axis, per-wheel mass and friction drawn separately, and
the payload swapped mid-episode. The policy is never told any of it — the actor
sees 24 frames (0.24 s at the 100 Hz control rate) of the same IMU / encoder /
command data the real robot publishes and has to infer the load from how the
robot responds. The critic is told, which costs nothing at deployment and
keeps value estimation sane.

Episodes also start harder than "just tilted": `reset_base` now draws an
initial linear + angular velocity (reusing `push_robot`'s own magnitudes,
since "just got shoved" and "starts having just been shoved" are the same
physical event) rather than always starting from exactly zero velocity, and
roll/pitch range widened from ±0.2 to ±0.35 rad. IMU-sourced observations
(`base_lin_vel`, `base_ang_vel`, `projected_gravity`) carry a per-episode bias
on top of the existing per-step noise (`NoiseModelWithAdditiveBiasCfg`) — a
persistent offset a policy can't average away, unlike i.i.d. per-step noise,
closer to how a real MEMS IMU's zero-rate offset behaves within one power
cycle. Magnitudes are order-of-magnitude placeholders, not a datasheet figure.

A curriculum widens the DR ranges (mass/payload/terrain, not the reset/noise
changes above, which are fixed) as mean episode length improves, because
starting at the full envelope produces no learnable episodes at all.

Control runs at **100 Hz** (`CONTROL_HZ` in `javis/balance_task.py`), not the
50 Hz originally assumed — real USB/ROS2 round trip tops out around 100–150 Hz
by recollection (not yet a rigorous benchmark, SIM2REAL.md sec 6), and 100 is
the conservative end: a policy trained at the *slower* rate generalizes more
safely to hardware running faster than it trained at than the reverse would.
Physics still runs at 400 Hz underneath (`decimation` follows from
`CONTROL_HZ` automatically) — that finer timestep isn't for contact accuracy,
it's because the ODrive PI loop is integrated explicitly, so how stiff a
velocity gain the sim can represent is capped by the physics timestep
(`DrivetrainCfg.stability_alpha`). `OdriveVelocityAction` raises at import if
`CONTROL_HZ`/timestep/gain ever drift out of the well-damped region, rather
than degrading training silently.

The checkpoint that validated all this (1500 iterations, curriculum 0.96) was
trained before this retune and is now shape-incompatible with current code —
see `checkpoints/README.md`. Train a fresh one with `scripts/train.sh`.

Evaluate and export:

```bash
.venv/bin/python scripts/tune_sim_gains.py                            # gain sweep
.venv/bin/python scripts/eval_payload_sweep.py   --checkpoint <ckpt>  # CSV + heatmaps
.venv/bin/python scripts/record_payload_video.py --checkpoint <ckpt>  # mp4
.venv/bin/python scripts/export_onnx.py          --checkpoint <ckpt>  # ONNX + I/O contract
```

## Alternative: plain MuJoCo + Gymnasium (no mjlab/wandb)

`javis/gym_env.py` is a second, independent path to the same balance +
twist-tracking problem, for when the mjlab/rsl-rl/wandb stack above isn't
available or wanted: one robot per OS process (plain `mujoco`, no
MuJoCo-Warp GPU batching), a standard `gymnasium.Env` interface, and
`stable-baselines3` for training. Logging is stdout + CSV only — no
tensorboard, no wandb.

```bash
.venv/bin/pip install -e ".[gym]"

.venv/bin/python scripts/train_gym.py --algo ppo --num-envs 16 --total-timesteps 2000000
.venv/bin/python scripts/train_gym.py --algo sac --num-envs 8  --total-timesteps 500000   # or td3 / ddpg

.venv/bin/python scripts/plot_gym_results.py --log-dir logs/gym_ppo/<run>   # writes PNGs next to the CSVs
```

It reuses `javis.robot_constants.get_spec()` and `javis.mass_model` directly
(both are plain MuJoCo/numpy/torch, nothing mjlab-specific), so it's built
from the same URDF, mass model and DR ranges (`javis/sim_config.py`) as
`javis/balance_task.py` — same reward terms, same ODrive PI velocity loop
(`javis/mdp/actions.py`), same feasibility filter, same 100 Hz/400 Hz control/
physics split, same 24-frame observation history. What's simplified, and why,
is documented in `javis/gym_env.py`'s module docstring; the one worth knowing
up front is that rough terrain is approximated by tilting gravity by the
slope angle each episode instead of building a heightfield — physically
equivalent from the robot's own dynamics, and it sidesteps the mjlab rough
task's heightfield-resolution contact warnings entirely.

Any continuous-action `stable-baselines3` algorithm works against
`JavisBalanceEnv` unchanged (it's a standard `Box(-1, 1, (2,))` action
space); `scripts/train_gym.py --algo` currently wires up PPO (on-policy) and
SAC/TD3/DDPG (off-policy, replay buffer, `gradient_steps=num_envs` so the
update-to-data ratio matches running one env at `train_freq=1`).

## Alternative #2: the real mjlab task through Gymnasium/stable-baselines3

`javis/gym_env_mjlab.py` splits the difference between the two options
above: `stable-baselines3` + terminal/CSV-only logging like
`javis/gym_env.py`, but running the ACTUAL `javis/balance_task.py` task
(GPU-batched physics, the real terrain generator, mjlab's own renderer --
skybox, shadows, reflections, the built-in target-vs-actual velocity
`debug_vis`) instead of the plain-`mujoco` reimplementation. Reach for this
one when the plain-MuJoCo pipeline's flatter rendering isn't good enough
(video review, presentations) or when its CPU-only throughput is the
bottleneck and a GPU is available.

```bash
.venv/bin/pip install -e ".[sim,gym]"   # needs BOTH extras -- mjlab (sim) + stable-baselines3 (gym)

.venv/bin/python scripts/train_gym_mjlab.py --algo ppo --num-envs 4096 --total-timesteps 20000000
.venv/bin/python scripts/train_gym_mjlab.py --algo sac --num-envs 512  --total-timesteps 2000000   # or td3 / ddpg

.venv/bin/python scripts/record_gym_mjlab_video.py --checkpoint <run>/model_final.zip --algo sac
```

The engineering problem this solves: mjlab's `ManagerBasedRlEnv` is already
GPU-batched over `num_envs` (torch tensors, one CUDA context), while SB3's
`VecEnv` API is numpy-based with the same shape convention but assumes the
parallelism comes from N separate env instances (normally N OS processes via
`SubprocVecEnv`). `JavisMjlabVecEnv` is a thin numpy&harr;torch translation
layer over the ALREADY-batched env, not N processes -- so `--num-envs` here
means mjlab-sized (hundreds to thousands, all in one process), not
`train_gym.py`'s SubprocVecEnv-sized tens. It also sets `auto_reset=False`
and drives resets itself, which is required for correctness: mjlab's default
auto-reset silently swaps in the next episode's first observation with no
way to recover the true terminal one, and SB3 needs that true terminal
observation (`infos[i]["terminal_observation"]`) to bootstrap value
estimates correctly across an episode boundary.

`train_gym_mjlab.py` auto-plots and auto-records a video when training
finishes, same as `train_gym.py` (`--no-plot` / `--no-video` to skip).
`record_gym_mjlab_video.py` shows several robots at once (`--num-envs`,
small) -- neighbors in the same batched sim, each with its own randomized
mass/CoM/payload/terrain tile from the task's own domain randomization, not
a scripted scenario list.

## Alternative #3: training through github.com/Bigkatoan/SRL

[SRL](https://github.com/Bigkatoan/SRL) is a separate YAML-first RL library
(PPO/SAC/DDPG/TD3/A2C/A3C, its own CLI). It resolves environments through
plain `gymnasium.make(id)`, so hooking it up to `javis/gym_env.py`'s
`JavisBalanceEnv` needed no changes on SRL's side: importing
`javis.gym_env` registers `Javis-Balance-v0` with gymnasium's own registry
(see that module's bottom), and `scripts/train_srl.py` is a two-line
launcher that does that import and then hands off to SRL's own CLI `main()`
unchanged.

```bash
.venv/bin/pip install -e ".[srl]"   # pulls srl-rl from GitHub, not PyPI

.venv/bin/python scripts/train_srl.py --config configs/srl/javis_balance_ppo.yaml \
    --env Javis-Balance-v0 --algo ppo --device cpu
.venv/bin/python scripts/train_srl.py --config configs/srl/javis_balance_sac.yaml \
    --env Javis-Balance-v0 --algo sac --device cpu
```

`configs/srl/javis_balance_{ppo,sac}.yaml` both use a single shared encoder
feeding both actor and critic, not SRL's own example configs' pattern of two
separate encoders — that pattern hits a real bug in SRL itself when the env
has one observation key and the config uses `>1` encoders with explicit
`input_name`: `srl/cli/train.py`'s rollout loop remaps the obs dict to
encoder names *before* calling `agent.predict()`, and `agent_model.py`'s
`forward()` remaps it *again* internally expecting the original key, which
no longer exists after the first pass → `KeyError`. A single shared encoder
sidesteps it: the first remap renames the obs key to the encoder's name, and
the second pass's "key already matches an encoder name" rule is then a
no-op. Verified both configs train end-to-end (score climbing, no errors)
before committing them.

### The real task, through SRL: `Javis-Payload-Rough`

The configs above train the toy plain-MuJoCo `Javis-Balance-v0` env (CPU,
`gymnasium.make`). To train the actual mjlab task this repo is really about
(GPU-batched, randomized payload/terrain, the one used for sim-to-real):

```bash
.venv/bin/python scripts/train_srl.py --config configs/srl/javis_mjlab_ppo.yaml \
    --env Javis-Payload-Rough --device cuda --n-envs 4096

.venv/bin/python scripts/train_srl.py --config configs/srl/javis_mjlab_sac.yaml \
    --env Javis-Payload-Rough --device cuda --n-envs 512
```

Setup/task-registration mechanics are on SRL's own docs, not repeated here: the
[JAVIS walkthrough](https://bigkatoan.github.io/SRL/source/integrations/mjlab.html#real-world-example-javis)
on `bigkatoan.github.io/SRL`.

**Which one to actually use** -- corrected after a real, *full-length* 20M-step
PPO run exposed a problem the earlier short comparison below missed entirely.
Neither algorithm currently has a run that's actually verified to hold a good
policy for the long haul -- **do not treat either config as "done, just
train it" for a real robot deployment yet.**

- **PPO (`javis_mjlab_ppo.yaml`) was previously (wrongly) documented here as
  "the reliable default... holds it, no further tuning needed."** A full
  20M-step run disproved that: `eval/score_mean` peaks around 1.87 at step
  ~3.5M, then **declines continuously for the remaining ~16M steps**, ending
  at 1.06 -- worse than the very first eval point. Root cause under active
  investigation: PPO's policy entropy (`ppo/entropy` in the console/
  TensorBoard, logged as *negative* real entropy -- see
  `srl/losses/rl_losses.py`) collapses from ~2.75 down to near-zero over the
  run, timed right around the peak, and `GaussianActorHead`'s `log_std_min`
  defaults to an essentially unbounded `-20.0` with nothing in this config
  overriding it -- i.e. nothing structurally stops the policy from
  collapsing into a narrow, brittle, near-deterministic strategy under
  continuous domain randomization. A higher `entropy_coef` measurably helps
  (raised the peak to 2.04 in one test) but does not fully stop the later
  decline by itself -- this is not yet a solved problem, just a diagnosed
  one. There is also currently **no best-checkpoint mechanism** -- training
  only saves the *final* checkpoint, so a run like this one loses its actual
  peak policy with no way to recover it. Don't trust a finished PPO run's
  final checkpoint without checking the full `eval/score_mean` trajectory
  first (`runs/ppo_javis_mjlab_ppo/metrics.jsonl`) -- the last checkpoint
  saved may be well past the best point in training.
- **SAC (`javis_mjlab_sac.yaml`) has the same underlying failure class**
  (entropy/temperature collapse -- SAC's `alpha` here, not PPO's policy std)
  and was investigated first. The default config already has SRL's async
  runner + GPU replay buffer + a batch-size restructuring for ~9.5x
  wall-clock throughput over the SAC textbook defaults (see that file's own
  comments), but at `lr_alpha: 3e-4` (unchanged) it's still prone to
  premature entropy collapse on a long run.
  `configs/srl/javis_mjlab_sac_flashsac.yaml` is an experimental variant
  (lower `lr_alpha` + FlashSAC-style weight normalization/BatchNorm, see
  [Bigkatoan/SRL#34](https://github.com/Bigkatoan/SRL/pull/34)) that fixes
  *that* instability -- two independent 2M-step runs plateaued consistently
  around `eval/score_mean` ≈ 1.2–1.8, no decline -- but that plateau is
  below PPO's *peak* (1.87-2.04), so it trades quality for reliability, not
  a strict win either.
- **Bottom line right now**: PPO can reach a higher score than SAC's
  stabilized plateau, but only briefly, before degrading past even SAC's
  level -- and nothing currently catches/saves that peak automatically. SAC
  is lower but (with the FlashSAC variant) actually holds. Until PPO's
  entropy-collapse fix and a best-checkpoint mechanism both land, treat any
  single finished training run's final checkpoint with suspicion -- check
  the eval trajectory, don't assume "training finished" means "training
  succeeded."
- A real, unrelated bug this surfaced: `javis/mdp/rewards.py`'s
  `pitch_rate_l2` term squares angular velocity with no clamp. SAC's
  optimized config trains an actor capable enough to occasionally find an
  action sequence that pushes mjlab's physics integrator into a divergent
  state, and squaring a huge-but-finite angular velocity there produces an
  astronomical (not NaN) reward value a few times per 2M-step run. Harmless
  to training itself (the huge value is finite, doesn't propagate into
  gradients that matter), but worth a defensive clamp — not yet fixed here.

## Simulation notes / known gaps

- **The ODrive velocity gains on the hardware cannot balance this robot.**
  Found in simulation, but it is a hardware conclusion, not a sim artifact.
  `vel_gain = 0.25 N·m/(turn/s)` against a 0.0122 kg·m² wheel is a velocity-loop
  time constant of 307 ms — 3 s once it also has to accelerate 11 kg of robot —
  while the robot's own fall time constant is `sqrt(h/g)` = 171 ms. The inner
  loop is slower than the thing it is supposed to stabilize, and no balance
  controller of any kind works through that.

  `javis/sim_config.py` therefore targets `vel_gain = 15.0` and
  `vel_integrator_gain = 75.0` — **60× and 500×** what is configured — chosen
  with `scripts/tune_sim_gains.py`, which sweeps candidates against both the
  loaded and unloaded wheel and scores them the same way
  `scripts/tune_wheel_pid.py` scores real hardware. That gives a 5.1 ms loop
  unloaded and 49.9 ms driving.

  Gains are stored in ODrive's own units, so the sim value goes straight onto
  the board. Do not jump there in one step: at that gain the 15 A limit is
  reached at 1.3 rad/s of velocity error, so encoder noise is amplified ~60×
  too. `SIM2REAL.md` sec 3 has a step-by-step ladder to walk it up on a stand.
  Note `scripts/tune_wheel_pid.py` sweeps only 0.15–0.40, on a free-spinning
  wheel where soft gains always score best — it cannot find this.
- **Component masses are known to varying degrees.** Battery and wheels are
  weighed; Jetson and camera are catalog figures; the MKS boards, IMU and
  fastener totals are estimates. Each group's domain-randomization range in
  `javis/sim_config.py` is set to reflect how well it is actually known, so
  weighing something later means narrowing one range rather than discovering a
  systematic error. `SIM2REAL.md` sec 1 lists what is still worth weighing.
- **Two drivetrain models exist; the payload tasks use the faithful one.**
  `Javis-Velocity-Flat` still uses mjlab's `BuiltinVelocityActuatorCfg`, which
  is proportional only. The real ODrive runs P **plus I**, and that integral
  term is exactly what absorbs an unknown load — omit it and the policy learns
  to do that job itself, then double-compensates against the real board and
  oscillates. `javis/mdp/actions.py` therefore simulates the board's PI loop
  explicitly (torque actuator + anti-windup + the board's velocity ramp), with
  `DrivetrainCfg.use_pi_actuator = False` to A/B against the builtin.
  `effort_limit` is grounded (Kt 0.207 N·m/A × 15 A); the fitted `damping`
  0.028 was a control gain, not physical drag, so physical wheel drag is now a
  separate randomized `joint_damping` term spanning 0 to that figure.
- **Chassis collision is a convex hull, not the exact shape.** The chassis
  carries 343 raw, non-convex CAD meshes (screws, gears, connectors...),
  visual-only and unfit for realtime contact, so `robot_constants.py` wraps
  them in one invisible convex-hull collision geom (`chassis_collision`) — a
  tipped-over rover now rests on that hull instead of clipping through the
  floor. It follows the true outer silhouette much more closely than a
  bounding box, but still fills in concave features (the vented headcover
  slots, the gap between the wide base and the narrower camera/Jetson
  tower). Replace with a multi-part convex decomposition (e.g. CoACD/V-HACD)
  if that precision matters.
- **The balance envelope is a hard limit, and heavy builds sit close to it.**
  Ground force is capped at `2·τ_max/r` = 63.4 N, so the steepest lean the
  motors can hold is `atan(63.4/(M·g))`: 30° at the nominal 11.1 kg, 18° at
  20 kg, 12° at 30 kg. On a 10° slope a 30 kg robot spends almost its entire
  budget just holding station. The requested randomization envelope reaches
  past that, so `sim_config.FeasibilityCfg` resamples configurations that are
  provably impossible rather than spending gradient on them, and
  `scripts/eval_payload_sweep.py` draws the boundary on its heatmaps. Set
  `enabled=False` to train on the raw envelope instead.
- **The rover is only stable on its own 2 wheels with a correct center of
  mass.** Two coaxial wheels and nothing else touching the ground is a
  Segway-style balancing configuration, not a tripod-stable base. Dropping the
  rover in sim from resting height pitches it over ~90 deg onto its face
  instead of standing — verified with `scripts/view_robot.py`. This is a real
  finding, not a mass artifact: this robot needs an active balance controller
  (`javis/balance_task.py`) rather than passive stability, unless the physical
  robot has a kickstand/tail contact point not yet in the URDF
  (`SIM2REAL.md` sec 2).
- **No ROS2 package / motor driver code.** Confirmed by inspecting the actual
  onboard Jetson Orin Nano (2026-08-06): ROS2 Humble + Isaac ROS are
  installed and already drive the sensor stack — `bmx160_bmp388_driver`
  (real IMU: Bosch BMX160 + BMP388, not the "DFRobot" name in old CAD
  comments, see `SIM2REAL.md` sec 5) and `orin_vslam_bringup` (RealSense
  D435 -> Isaac ROS cuVSLAM + nvblox) both run as systemd services/launch
  files — but there is no package or script anywhere on that machine talking
  to the MKS ODrive Mini controllers yet.
  **Decided 2026-08-07: this driver will talk USB, not CAN.** `can0` is up
  on the Jetson (`can0-up.service`) but after a long hardware debugging
  session found the right wheel's onboard CAN transceiver chip physically
  dead (transmit stage doesn't drive the bus at all — confirmed by process
  of elimination against a known-good left board and direct multimeter
  probing at the chip, see `MKS_XDRIVE_MINI.md`), the project dropped CAN as
  the primary interface: with only 2 motors, CAN's main advantages (shared
  bus for many nodes, hard real-time determinism) don't outweigh the
  hardware risk already hit. USB (native Fibre protocol via the `odrive`
  Python package, same approach `scripts/setup_odrive.py` already uses) is
  now the plan — Jetson has enough USB ports for both boards. This Jetson
  also runs an unrelated voice-assistant stack (wakeword, whisper.cpp,
  piper, ollama) sharing its 6-core/7.4GB budget — worth accounting for once
  a real-time policy needs to run here too.
- **`velocity_task.py`'s own numbers are still untuned placeholders.** Action
  `scale=5.0` (rad/s per unit policy output), the 50 Hz control rate
  (`decimation=4` @ 5 ms physics timestep), and all reward weights were picked
  by analogy to mjlab's legged-robot tasks, not tuned for this robot —
  deliberately left as-is (see **Training a balance/velocity policy** above:
  it's the fixed baseline the payload tasks are compared against, so it
  doesn't inherit their fixes). `balance_task.py` has since moved to a
  measured-ish 100 Hz control rate (see **Training a policy that survives a
  changing payload**); port that change here too if this baseline is ever
  used for more than comparison.
- **IMU/camera are geometrically placed but not characterized.** The IMU
  site and D435 camera in `robot_constants.py` use the CAD-mounted pose
  (real, from `robot.urdf`); the camera still has a generic 42 deg vertical
  FOV guess rather than the D435's factory intrinsics. The payload tasks do
  add IMU noise and a randomized 1–3 control-step observation delay, but those
  bounds are guesses too — the real BMX160 noise density and the real
  sensor-to-policy latency are both still unmeasured (`SIM2REAL.md` sec 5, 6).
- **`base_lin_vel` assumes the VSLAM stack is up.** The actor observes linear
  velocity, which the BMX160 cannot provide directly — on hardware it has to
  come from the Isaac ROS cuVSLAM odometry already running on the Jetson. If
  that pipeline drops out, the policy loses an observation it was trained on.
  Worth deciding whether to train a variant without it before deployment.

## Sim-to-real

`SIM2REAL.md` is a detailed, fillable checklist of what to measure/log on
the physical robot (mass distribution, drivetrain characterization, sensor
calibration, control-loop timing, ...) so a policy or controller tuned in
this simulation needs minimal changes on real hardware.
