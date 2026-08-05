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
  `robot.urdf`, adds a free-floating root joint, wheel actuators, wheel
  collision geoms, and approximate mass (see **Known gaps** below).
- `scripts/view_robot.py` — builds an mjlab `Scene` (flat ground plane + the
  rover) and opens it in MuJoCo's interactive viewer.
- `setup.bash` / `.env` — Onshape API credentials, loaded as environment
  variables (see **Setup**).

## Setup

```bash
cp .env.example .env      # fill in your Onshape API key/secret
source setup.bash         # exports ONSHAPE_* into the shell
source venv/bin/activate  # or call venv/bin/python directly
```

`.env` holds real credentials and is git-ignored — never commit it.
`requirements.txt` is a full freeze of the working `venv` (includes ROS2 /
Isaac ROS packages, which also require the underlying ROS2 system install on
a Jetson — pip alone won't reproduce those on a fresh machine).

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

## Simulation notes / known gaps

- **Mass is fake.** No material densities are assigned in the Onshape
  assembly, so `robot.urdf` exports `mass="1e-9"` for every link.
  `robot_constants.py` works around this by recomputing inertia from
  collision geometry and rescaling the whole robot to `TOTAL_MASS_KG` (a
  placeholder, currently 3.0 kg). Replace once real masses are known —
  weigh the assembled robot and each wheel separately, then update
  `TOTAL_MASS_KG` (and consider per-body mass once wheel mass is known
  precisely, since `settotalmass` currently distributes proportionally to
  geometry volume across the whole robot).
- **Wheel actuator gains are placeholders.** `WHEEL_ACTUATOR_CFG` uses a
  generic `BuiltinVelocityActuatorCfg` with guessed damping/effort limits,
  not real MKS ODrive Mini + motor datasheet numbers. For a more faithful
  model, see `mjlab.actuator.BuiltinDcMotorActuatorCfg`.
- **Only the wheels have collision geometry.** The chassis carries 343 raw,
  non-convex CAD meshes (screws, gears, connectors...) which are visual-only
  — not simplified for realtime contact. A tipped-over rover will currently
  clip through the floor instead of colliding with it. Add a simplified
  chassis collision hull (a box or a decimated convex mesh) before relying on
  chassis-ground or chassis-obstacle contact.
- **No RL task defined yet.** `venv` has `rsl-rl-lib`, `torch`, `wandb`, and
  mjlab's manager-based RL stack installed, but no `mjlab.tasks` entry for
  this robot. `mjlab/tasks/cartpole` is the simplest structural template for
  a first velocity-tracking task; `mjlab/tasks/velocity` is closer in intent
  (twist command tracking) but written for legged robots — its `foot_*`/
  `pose` reward and observation terms don't apply to a 2-wheel rover and
  would need to be stripped.
- **No ROS2 package / motor driver code.** ROS2 and Isaac ROS are installed
  (camera, IMU, general middleware) but there's no `package.xml`, launch
  file, or code talking to the MKS ODrive Mini controllers yet — nothing here
  currently drives the physical robot.
