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
  collision hull, an IMU site + sensors, a D435 camera, and approximate mass
  (see **Known gaps** below).
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

Verified end-to-end (env builds, steps, and trains without NaNs/crashes;
reward increases within the first few PPO iterations of a smoke test) but
**not yet trained to convergence or tried on hardware** — every numeric gap
in the section below (mass, actuator gains, friction) feeds directly into
this task's dynamics, so revisit `--agent.max-iterations` results after
closing those gaps rather than trusting an early checkpoint.

## Simulation notes / known gaps

- **Wheel mass is real, chassis mass is still a placeholder.** No material
  densities are assigned in the Onshape assembly, so `robot.urdf` exports
  `mass="1e-9"` for every link. `robot_constants.py` works around this with
  `_set_mass_from_density`, which scales each body's geom density so
  MuJoCo's own inertiafromgeom-computed mass hits a per-body target — not a
  single whole-robot total, so a known wheel mass can't skew the chassis or
  vice versa. `WHEEL_MASS_KG = 2.936` kg is a real measurement (per wheel).
  `CHASSIS_MASS_KG = 6.0` kg is still a guess (known lower bound: the
  battery alone, one part among ~343 fused into that link, is a confirmed
  3.423 kg). See `SIM2REAL.md` sec 1 for what's still needed (full robot
  total, remaining component masses) to replace it.
- **Wheel actuator gains are placeholders** until calibrated with
  `scripts/calibrate_actuator.py` (see above) against real datasheet numbers
  or a logged step response. For a more faithful drivetrain model than the
  plain `BuiltinVelocityActuatorCfg` used now, see
  `mjlab.actuator.BuiltinDcMotorActuatorCfg`.
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
- **The rover is only stable on its own 2 wheels with a correct center of
  mass.** Two coaxial wheels and nothing else touching the ground is a
  Segway-style balancing configuration, not a tripod-stable base. Even with
  real wheel mass now in place (2.936 kg each, well above the still-guessed
  6 kg chassis), dropping the rover in sim from resting height still pitches
  it over ~90 deg onto its face instead of standing — verified with
  `scripts/view_robot.py`. This is no longer just a "mass is fake" artifact,
  so treat it as a real finding: this robot needs an active balance
  controller (a natural first `mjlab` RL task, see `javis/velocity_task.py`)
  rather than passive stability, unless the physical robot has a
  kickstand/tail contact point not yet in the URDF (`SIM2REAL.md` sec 2).
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
- **The RL task's own numbers are untuned placeholders too.** Action
  `scale=5.0` (rad/s per unit policy output), the 50 Hz control rate
  (`decimation=4` @ 5 ms physics timestep), and all reward weights in
  `velocity_task.py` were picked by analogy to mjlab's legged-robot tasks,
  not tuned for this robot. The 50 Hz figure in particular should be
  replaced with the real control loop rate once known (`SIM2REAL.md` sec 6)
  rather than left as a guess.
- **IMU/camera are geometrically placed but not characterized.** The IMU
  site and D435 camera in `robot_constants.py` use the CAD-mounted pose
  (real, from `robot.urdf`) but noiseless, zero-latency sensor models and a
  generic 42 deg vertical FOV guess for the camera — not the real D435's
  factory-calibrated intrinsics or the IMU's real noise/bias. See
  `SIM2REAL.md`.

## Sim-to-real

`SIM2REAL.md` is a detailed, fillable checklist of what to measure/log on
the physical robot (mass distribution, drivetrain characterization, sensor
calibration, control-loop timing, ...) so a policy or controller tuned in
this simulation needs minimal changes on real hardware.
