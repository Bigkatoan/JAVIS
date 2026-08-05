"""JAVIS rover constants, mjlab EntityCfg definition.

Mirrors the pattern used by mjlab's own asset_zoo robots (see
mjlab/asset_zoo/robots/unitree_go1/go1_constants.py): a get_spec() that builds
the mujoco.MjSpec, actuator/collision configs, an initial state, and a
get_javis_robot_cfg() factory returning the EntityCfg mjlab consumes.

Source of truth for geometry is javis/robot.urdf (exported from Onshape via
onshape-to-robot, see javis/config.json). Re-run this whenever robot.urdf
changes -- no separate MJCF is checked in, the URDF is converted on the fly.
"""

import re
from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinVelocityActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

##
# URDF source.
##

JAVIS_DIR: Path = Path(__file__).resolve().parent
JAVIS_URDF: Path = JAVIS_DIR / "robot.urdf"
assert JAVIS_URDF.exists()

WHEEL_JOINTS = ("left_wheel", "right_wheel")

# robot.urdf has no real mass/inertia (onshape-to-robot exported 1e-9
# placeholders because no material densities are assigned in the Onshape
# assembly yet). Recompute inertia from collision geometry and rescale the
# whole robot to this total instead of simulating an effectively massless
# rover. Replace with the real value once the CAD has materials assigned and
# robot.urdf is re-pulled.
TOTAL_MASS_KG = 3.0


def get_spec() -> mujoco.MjSpec:
  """Load robot.urdf as an MjSpec, resolving package:// mesh URIs, naming the
  wheel collision geoms (URDF import otherwise leaves all geoms unnamed,
  which breaks mjlab's regex-based CollisionCfg/ActuatorCfg matching), adding
  a free joint on the chassis (URDF always assumes a fixed base), and
  patching in approximate mass/inertia (see TOTAL_MASS_KG above)."""
  urdf_text = JAVIS_URDF.read_text()
  fixed_urdf = re.sub(r"package://assets/", "assets/", urdf_text)

  tmp_urdf = JAVIS_DIR / "_javis_spec_tmp.urdf"
  tmp_urdf.write_text(fixed_urdf)
  try:
    spec = mujoco.MjSpec.from_file(str(tmp_urdf))
  finally:
    tmp_urdf.unlink()

  # URDF import leaves every geom unnamed (empty string). mjlab's
  # CollisionCfg(disable_other_geoms=True) deduplicates by name when turning
  # off collision for non-matching geoms, so a shared "" name would only
  # disable one of the 342 chassis geoms instead of all of them. Give every
  # geom a unique name first, then name the two wheels distinctly.
  for i, geom in enumerate(spec.geoms):
    if not geom.name:
      geom.name = f"geom_{i}"

  for body in spec.bodies:
    if body.name in ("wheel", "wheel_2"):
      body.geoms[0].name = f"{body.name}_collision"

  chassis = spec.worldbody.bodies[0]
  chassis.add_freejoint()

  spec.compiler.inertiafromgeom = mujoco.mjtInertiaFromGeom.mjINERTIAFROMGEOM_TRUE
  spec.compiler.settotalmass = TOTAL_MASS_KG

  return spec


##
# Actuator config.
##

# Placeholder gains: robot.urdf only carries onshape-to-robot's generic joint
# limit (effort=10 N*m, velocity=10 rad/s), not real MKS ODrive Mini + motor
# datasheet numbers. Replace once the drivetrain is characterized -- see
# mjlab.actuator.BuiltinDcMotorActuatorCfg for modeling the ODrive's velocity
# loop more faithfully than this plain <velocity> actuator.
WHEEL_ACTUATOR_CFG = BuiltinVelocityActuatorCfg(
  target_names_expr=WHEEL_JOINTS,
  damping=5.0,
  effort_limit=10.0,
)

JAVIS_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(WHEEL_ACTUATOR_CFG,),
)

##
# Collision config.
##

# Only the two wheels get contact geoms; disable_other_geoms=True (default)
# turns off collision for the chassis's 343 raw, non-convex CAD meshes, which
# are visual-only meshes as exported and not fit for realtime contact.
# TODO: once the chassis has a simplified collision hull, add a second
# CollisionCfg for it so a tipped-over rover doesn't clip through the floor.
WHEEL_COLLISION = CollisionCfg(
  geom_names_expr=("wheel_collision", "wheel_2_collision"),
  condim=3,
  priority=1,
  friction=(1.0, 0.005, 0.0001),
)

##
# Initial state.
##

# z offset ~= wheel radius (0.098 m, from assets/wheel.stl bounds) so the
# rover spawns resting on the ground plane instead of embedded in it.
INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.12),
)


def get_javis_robot_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(WHEEL_COLLISION,),
    spec_fn=get_spec,
    articulation=JAVIS_ARTICULATION,
  )


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_javis_robot_cfg())
  viewer.launch(robot.spec.compile())
