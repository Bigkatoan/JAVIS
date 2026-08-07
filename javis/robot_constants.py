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
import numpy as np
import trimesh

from mjlab.actuator import BuiltinVelocityActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.sensor import CameraSensorCfg
from mjlab.utils.spec_config import CameraCfg, CollisionCfg

##
# URDF source.
##

JAVIS_DIR: Path = Path(__file__).resolve().parent
JAVIS_URDF: Path = JAVIS_DIR / "robot.urdf"
assert JAVIS_URDF.exists()

WHEEL_JOINTS = ("left_wheel", "right_wheel")
WHEEL_COLLISION_GEOMS = ("wheel_collision", "wheel_2_collision")
CHASSIS_BODY_NAME = "body"
CHASSIS_COLLISION_GEOM = "chassis_collision"


def _quat_to_mat(quat: tuple[float, float, float, float]) -> np.ndarray:
  """MuJoCo wxyz quaternion -> 3x3 rotation matrix."""
  w, x, y, z = quat
  return np.array(
    [
      [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
  )


def _load_mesh_cached(
  spec: mujoco.MjSpec, meshname: str, cache: dict[str, trimesh.Trimesh]
) -> trimesh.Trimesh:
  if meshname not in cache:
    mesh_spec = next(m for m in spec.meshes if m.name == meshname)
    mesh_path = Path(mesh_spec.file)
    if not mesh_path.is_absolute():
      mesh_path = JAVIS_DIR / mesh_path
    cache[meshname] = trimesh.load(mesh_path)
  return cache[meshname]


def _chassis_point_cloud(
  spec: mujoco.MjSpec, chassis: mujoco.MjsBody, mesh_cache: dict[str, trimesh.Trimesh]
) -> np.ndarray:
  """Every chassis mesh vertex, transformed into the chassis body frame."""
  points = []
  for geom in chassis.geoms:
    if geom.type != mujoco.mjtGeom.mjGEOM_MESH:
      continue
    mesh = _load_mesh_cached(spec, geom.meshname, mesh_cache)
    world_verts = mesh.vertices @ _quat_to_mat(tuple(geom.quat)).T + np.array(geom.pos)
    points.append(world_verts)
  return np.concatenate(points, axis=0)


_CALIBRATION_DENSITY = 1000.0


def _default_density_body_masses() -> dict[str, float]:
  """What would body_mass be for "body"/"wheel"/"wheel_2" if every mesh geom
  had a uniform density of _CALIBRATION_DENSITY? Used to scale each body's
  real geom density to hit a real target mass (see _set_mass_from_density).

  Computed from a throwaway MjSpec parsed fresh from robot.urdf, independent
  of the main spec get_spec() builds. Needed because get_spec()'s spec has
  a freejoint added to "body" (chassis) already, and a fresh raw URDF parse
  doesn't -- and a body with no joint at all is, by default, silently fused
  into "world" at compile time (`compiler.fusestatic`), which drops it from
  the compiled model as a separate named body. mj_name2id() then can't find
  "body", returns -1, and model.body_mass[-1] silently wraps around to
  whatever the *last* body happens to be (wheel_2) instead of raising --
  disabling fusestatic here avoids that trap.
  """
  urdf_text = JAVIS_URDF.read_text()
  fixed_urdf = re.sub(r"package://assets/", "assets/", urdf_text)
  tmp_urdf = JAVIS_DIR / "_javis_calib_tmp.urdf"
  tmp_urdf.write_text(fixed_urdf)
  try:
    calib_spec = mujoco.MjSpec.from_file(str(tmp_urdf))
  finally:
    tmp_urdf.unlink()

  for g in calib_spec.geoms:
    if g.type == mujoco.mjtGeom.mjGEOM_MESH:
      g.density = _CALIBRATION_DENSITY
  calib_spec.compiler.inertiafromgeom = mujoco.mjtInertiaFromGeom.mjINERTIAFROMGEOM_TRUE
  calib_spec.compiler.fusestatic = False
  model = calib_spec.compile()

  return {
    name: model.body_mass[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)]
    for name in ("body", "wheel", "wheel_2")
  }


def _set_mass_from_density(
  body: mujoco.MjsBody,
  target_mass_kg: float,
  mass_at_calibration_density: float,
  exclude_geom_names: tuple[str, ...] = (),
) -> None:
  """Set a uniform geom density on `body`'s mesh geoms so that
  mjINERTIAFROMGEOM_TRUE-computed mass comes out to exactly target_mass_kg,
  while still deriving the inertia *shape* from the real geometry rather
  than treating the body as a point mass. Geoms in exclude_geom_names (e.g.
  a density=0 collision hull) are left untouched.

  `mass_at_calibration_density` comes from _default_density_body_masses()
  -- see that function's docstring for why this doesn't compile `body`'s
  own spec itself to get it.
  """
  scale = target_mass_kg / mass_at_calibration_density
  for g in body.geoms:
    if g.type == mujoco.mjtGeom.mjGEOM_MESH and g.name not in exclude_geom_names:
      g.density = _CALIBRATION_DENSITY * scale


def _add_chassis_collision_hull(
  spec: mujoco.MjSpec, chassis: mujoco.MjsBody, mesh_cache: dict[str, trimesh.Trimesh]
) -> None:
  """Add one convex-hull collision geom wrapping the chassis's visual
  meshes. robot.urdf's 343 chassis part meshes are raw, non-convex CAD
  output (screws, gears, connectors...) unfit for realtime contact, and by
  default the chassis has no collision at all (see ROBOT_COLLISION below),
  so a tipped-over rover would otherwise clip through the floor instead of
  hitting it.

  A single convex hull follows the true outer silhouette (tapered/rounded
  edges) far more closely than one bounding box, at the same runtime cost
  (MuJoCo treats it as one convex collision mesh). It's still not exact --
  concave features like the vented headcover slots or the gap between the
  wide base and the narrower camera/Jetson tower get filled in. Replace with
  a multi-part convex decomposition (e.g. via CoACD/V-HACD) if that matters.

  Passing only vertices (no faces) to MjSpec.add_mesh makes MuJoCo compute
  the convex hull itself at compile time.

  density=0 on the resulting geom so it contributes no mass/volume to the
  inertiafromgeom computation in get_spec() -- it would otherwise double up
  with the visual meshes it wraps and skew the chassis/wheel mass split.
  """
  points = _chassis_point_cloud(spec, chassis, mesh_cache)
  hull_verts = trimesh.Trimesh(vertices=points).convex_hull.vertices

  spec.add_mesh(name=CHASSIS_COLLISION_GEOM, uservert=hull_verts.flatten().tolist())
  chassis.add_geom(
    name=CHASSIS_COLLISION_GEOM,
    type=mujoco.mjtGeom.mjGEOM_MESH,
    meshname=CHASSIS_COLLISION_GEOM,
    density=0,
    rgba=[1, 0, 0, 0],  # invisible; the detailed chassis meshes are the visual layer
    contype=0,
    conaffinity=0,  # enabled by ROBOT_COLLISION below; disabled here as a safe default
  )


def _rpy_to_quat(rpy: tuple[float, float, float]) -> tuple[float, float, float, float]:
  """URDF <origin rpy="..."/> (extrinsic X-Y-Z, i.e. R = Rz(yaw)@Ry(pitch)@Rx(roll))
  -> MuJoCo (w, x, y, z) quaternion."""
  roll, pitch, yaw = rpy
  cr, sr = np.cos(roll / 2), np.sin(roll / 2)
  cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
  cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
  return (
    cr * cp * cy + sr * sp * sy,
    sr * cp * cy - cr * sp * sy,
    cr * sp * cy + sr * cp * sy,
    cr * cp * sy - sr * sp * cy,
  )


# IMU mount pose, copied from robot.urdf's "dfbot_imu_sensor" part <origin>,
# relative to the chassis ("body") link frame. Note: that CAD part name is a
# placeholder from the Onshape library -- the real sensor on the Jetson is a
# Bosch BMX160 (+ BMP388 baro), confirmed via the bmx160_bmp388_driver ROS2
# package running on-robot (see SIM2REAL.md sec 5).
IMU_SITE_NAME = "imu"
IMU_POS = (-0.0675, 0.0, 0.26695)
IMU_RPY = (0.0, 0.0, 1.5708)


def _add_imu(spec: mujoco.MjSpec, chassis: mujoco.MjsBody) -> None:
  """Add a site at the IMU's CAD-mounted pose plus gyro/velocimeter/
  accelerometer sensors on it, mirroring mjlab's own robots (e.g.
  asset_zoo/robots/unitree_go1/xmls/go1.xml's "imu" site + sensor block) so
  the same mjlab.sensor.BuiltinSensor / observation terms that read
  "<entity>/imu_ang_vel" etc. work for this robot too."""
  chassis.add_site(name=IMU_SITE_NAME, pos=IMU_POS, quat=_rpy_to_quat(IMU_RPY))
  spec.add_sensor(
    name="imu_ang_vel",
    type=mujoco.mjtSensor.mjSENS_GYRO,
    objtype=mujoco.mjtObj.mjOBJ_SITE,
    objname=IMU_SITE_NAME,
  )
  spec.add_sensor(
    name="imu_lin_vel",
    type=mujoco.mjtSensor.mjSENS_VELOCIMETER,
    objtype=mujoco.mjtObj.mjOBJ_SITE,
    objname=IMU_SITE_NAME,
  )
  spec.add_sensor(
    name="imu_lin_acc",
    type=mujoco.mjtSensor.mjSENS_ACCELEROMETER,
    objtype=mujoco.mjtObj.mjOBJ_SITE,
    objname=IMU_SITE_NAME,
  )


# Intel RealSense D435 mount pose, copied from robot.urdf's
# "intelrealsensed435" part <origin>, relative to the chassis link frame.
# Published D435 RGB module specs: ~69 x 42 deg FOV (H x V); the depth
# (stereo) module has a wider ~87 x 58 deg FOV and a small baseline offset
# from the RGB lens that this single camera does not model.
D435_CAMERA_NAME = "d435"
D435_POS = (0.12105, 0.0, 0.30795)
D435_RPY = (1.5708, 0.0, 1.5708)
D435_FOVY_DEG = 42.0

# robot.urdf has no real mass/inertia (onshape-to-robot exported 1e-9
# placeholders because no material densities are assigned in the Onshape
# assembly yet). get_spec() recomputes inertia from geometry and scales each
# body's geom density to hit these per-body mass targets -- see
# _set_mass_from_density -- instead of one global total (which would let an
# unknown chassis mass distort the now-known wheel mass, or vice versa).

# Measured 2026-08-06 (see SIM2REAL.md sec 1): each wheel (hub motor +
# tire assembly) weighs 2936 g. Assumed equal for both wheels -- flag if
# the real robot's two wheels differ.
WHEEL_MASS_KG = 2.936

# Still an unmeasured placeholder. Known lower bound: the battery alone
# (one of ~343 parts fused into this single "body" link) is confirmed at
# 3423 g (SIM2REAL.md sec 1) -- everything else (frame, Jetson, camera,
# ODrive boards, screws...) adds more on top of that, so this number is
# almost certainly too low. Replace once the remaining components in
# SIM2REAL.md sec 1 are weighed (or once Onshape has real material
# densities assigned, making this whole density-scaling workaround
# unnecessary).
CHASSIS_MASS_KG = 6.0


def get_spec() -> mujoco.MjSpec:
  """Load robot.urdf as an MjSpec, resolving package:// mesh URIs, naming the
  wheel collision geoms (URDF import otherwise leaves all geoms unnamed,
  which breaks mjlab's regex-based CollisionCfg/ActuatorCfg matching), adding
  a free joint on the chassis (URDF always assumes a fixed base), and
  patching in mass/inertia per body (see WHEEL_MASS_KG/CHASSIS_MASS_KG
  above)."""
  mesh_cache: dict[str, trimesh.Trimesh] = {}
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
  _add_chassis_collision_hull(spec, chassis, mesh_cache)
  _add_imu(spec, chassis)
  chassis.add_freejoint()

  # MuJoCo's URDF importer sets compiler.discardvisual=True (a URDF-workflow
  # default: visual and collision geoms are usually separate/redundant in a
  # well-formed URDF, so the compiler drops the non-colliding "visual"
  # copies at compile time). We rely on the opposite here: ROBOT_COLLISION
  # below disables collision (contype=conaffinity=0) on the 343 detailed
  # chassis meshes precisely so they stay purely visual while a separate,
  # simplified hull handles contact -- but that makes them exactly what
  # discardvisual silently deletes. Cost 2+ hours to track down: geom count
  # (and mass, and visually the whole chassis) looked fine right up until
  # the *final* compile, since discardvisual only fires then. Must override
  # before any collision-disabling happens, though setting it here (before
  # compile) is sufficient regardless of exact order.
  spec.compiler.discardvisual = False

  spec.compiler.inertiafromgeom = mujoco.mjtInertiaFromGeom.mjINERTIAFROMGEOM_TRUE
  default_masses = _default_density_body_masses()
  _set_mass_from_density(chassis, CHASSIS_MASS_KG, default_masses["body"], (CHASSIS_COLLISION_GEOM,))
  for wheel_body_name in ("wheel", "wheel_2"):
    wheel_body = next(b for b in spec.bodies if b.name == wheel_body_name)
    _set_mass_from_density(wheel_body, WHEEL_MASS_KG, default_masses[wheel_body_name])

  return spec


##
# Actuator config.
##

# Real motor identified: 36V/350W direct-drive hub motor built into an
# 8-inch wheel -- gear_ratio = 1:1 for the POWER path (motor torque directly
# drives the wheel, no reduction gearbox). Wheel radius from that spec
# (0.1016 m) matches assets/wheel.stl's measured ~0.098 m well.
#
# gearbox.stl/spur_gear__20_teeth/__30_teeth in the CAD are unrelated to the
# above: confirmed with the user they're a separate SENSING-path gear train
# (two stages, 30:20 then 20:30 -- they cancel out) that relocates the
# encoder magnet up to the MKS ODrive Mini board, not a power-transmission
# gearbox. Net 1:1, so raw ODrive turns/s already equal wheel rad/s; doesn't
# affect this sim either way since WHEEL_JOINTS here is driven directly.
#
# Measured on real hardware 2026-08-06 (right wheel, axis0 on the MKS
# xDrive Mini, see scripts/setup_odrive.py): with the wheel free-spinning
# (known inertia 0.01205 kg*m^2 from real mass + CAD geometry) and
# current_lim=15A configured, commanded a step to 5 turn/s in velocity
# mode and logged (t, vel_estimate, Iq_measured) at ~1.5kHz.
#
#   torque_constant (Kt) = 0.207 N*m/A, from I*delta_omega / integral(Iq dt)
#     over several window lengths of the initial ramp (mean of 6 estimates,
#     ~16% spread -- a quick single-shot bench measurement, not lab-grade,
#     but light-years better than the ODrive firmware's untouched default
#     torque_constant=0.04 N*m/A, which was never actually calibrated).
#     Cross-check: implied KV = 8.27/Kt = 40 rpm/V -> no-load speed at the
#     measured 23.5V bus = 939 RPM, plausibly close to the OEM datasheet's
#     800 RPM figure (different voltage/loading, so not an exact match).
#   effort_limit = Kt * current_lim = 0.207 * 15 = 3.1 N*m
#     (more grounded than dividing rated power by an ambiguous datasheet
#     RPM figure, previously used here -- see git history)
#   damping = 0.028 N*m/(rad/s), from scripts/calibrate_actuator.py fit
#     against that same logged step (see /tmp/right_wheel_step.csv in that
#     session -- not committed). RMS fit error 1.86 rad/s was fairly large
#     relative to the ~12 rad/s reached in the 0.2s capture: it never got
#     close to the commanded 31.4 rad/s, so this constrains damping's
#     initial-slope behavior reasonably but says little about steady state.
# Both current_lim=15A and this damping are still just the sample script's
# untuned starting values carried onto real hardware, not tuned against the
# loaded robot (free-spinning wheel only) -- expect to retune once the
# wheel is actually driving the assembled robot. Re-run
# scripts/calibrate_actuator.py fit with a longer, cleaner log (multiple
# velocity steps, each held to steady state) for a confident final value.
WHEEL_ACTUATOR_CFG = BuiltinVelocityActuatorCfg(
  target_names_expr=WHEEL_JOINTS,
  damping=0.028,
  effort_limit=3.1,
)

JAVIS_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(WHEEL_ACTUATOR_CFG,),
)

##
# Collision config.
##

# A single CollisionCfg for every geom that should actually collide: the two
# wheels plus the chassis bounding-box hull added in _add_chassis_collision_box.
# disable_other_geoms=True (default) turns off collision for the chassis's
# 343 raw, non-convex CAD meshes, which are visual-only meshes as exported
# and not fit for realtime contact.
#
# Must stay a *single* CollisionCfg: two separate CollisionCfg entries would
# each re-disable the other's geoms via their own disable_other_geoms pass
# (mjlab applies them in order), turning collision back off.
ROBOT_COLLISION = CollisionCfg(
  geom_names_expr=(*WHEEL_COLLISION_GEOMS, CHASSIS_COLLISION_GEOM),
  condim=3,
  priority={g: 1 for g in WHEEL_COLLISION_GEOMS} | {CHASSIS_COLLISION_GEOM: 0},
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


D435_CAMERA_CFG = CameraCfg(
  name=D435_CAMERA_NAME,
  body=CHASSIS_BODY_NAME,
  pos=D435_POS,
  quat=_rpy_to_quat(D435_RPY),
  fovy=D435_FOVY_DEG,
)


def get_javis_robot_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(ROBOT_COLLISION,),
    spec_fn=get_spec,
    articulation=JAVIS_ARTICULATION,
    cameras=(D435_CAMERA_CFG,),
  )


def get_javis_camera_sensor_cfg(
  entity_name: str = "robot", **overrides
) -> CameraSensorCfg:
  """Scene-level sensor wrapping the D435 camera added by get_javis_robot_cfg,
  for tasks/scripts that need actual RGB/depth tensors (not just the camera
  frustum visible in the viewer). `entity_name` must match the key this
  robot is registered under in SceneCfg.entities."""
  return CameraSensorCfg(
    camera_name=f"{entity_name}/{D435_CAMERA_NAME}",
    width=640,
    height=480,
    data_types=("rgb", "depth"),
    **overrides,
  )


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_javis_robot_cfg())
  viewer.launch(robot.spec.compile())
