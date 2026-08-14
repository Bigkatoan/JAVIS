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
import tempfile
from pathlib import Path

import mujoco
import numpy as np
import trimesh

from mjlab.actuator import BuiltinMotorActuatorCfg, BuiltinVelocityActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.sensor import CameraSensorCfg
from mjlab.utils.spec_config import CameraCfg, CollisionCfg

from . import mass_model

##
# URDF source.
##

JAVIS_DIR: Path = Path(__file__).resolve().parent
JAVIS_URDF: Path = JAVIS_DIR / "robot.urdf"
assert JAVIS_URDF.exists()

WHEEL_JOINTS = ("left_wheel", "right_wheel")
WHEEL_COLLISION_GEOMS = ("wheel_left_collision", "wheel_right_collision")
WHEEL_BODIES = ("wheel_left", "wheel_right")
CHASSIS_BODY_NAME = "base_link"
CHASSIS_COLLISION_GEOM = "chassis_collision"
PAYLOAD_GEOM = "payload"

# CAD re-pulled 2026-08-12 (new Onshape element, wheel/mount hardware
# redesigned -- link names changed body/wheel/wheel_2 -> base_link/wheel_left/
# wheel_right, but the wheel mesh itself (radius, width) is unchanged from the
# previous pull; only its mounting position on the chassis moved).

# Measured from assets/wheel_left.stl bounds. Used for the analytic balance
# envelope (scripts/inspect_mass.py) and for converting wheel torque to
# ground force.
WHEEL_RADIUS_M = 0.098
# Half the wheel track, from robot.urdf's left/right wheel joint origins
# (y = -+0.1, narrower than the previous CAD's 0.129). The lateral limit for
# the CoM before one wheel unloads.
WHEEL_HALF_TRACK_M = 0.1


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


def _apply_mass_model(spec: mujoco.MjSpec) -> None:
  """Write explicit inertial properties onto each body from javis.mass_model.

  Supersedes the old uniform-density approach (one density across all 343
  chassis parts, scaled to hit a guessed total). That produced the right total
  but the wrong center of mass, because the battery is 39% of chassis mesh
  volume at ~5x the density of the printed structure around it -- and CoM is
  precisely what a two-wheel balancing robot is sensitive to. See
  javis/mass_model.py for the per-component-group model that replaces it.

  Every mesh geom gets density=0 and the bodies carry explicit inertials, so
  MuJoCo does no inertia computation of its own here. mass_model's numbers are
  the single source of truth, in sim and in the per-environment randomization
  in javis/mdp/events.py alike.
  """
  # AUTO (not FALSE): bodies we don't set explicitly still fall back to the
  # geom-derived path instead of erroring out at compile time.
  spec.compiler.inertiafromgeom = mujoco.mjtInertiaFromGeom.mjINERTIAFROMGEOM_AUTO

  for geom in spec.geoms:
    if geom.type == mujoco.mjtGeom.mjGEOM_MESH:
      geom.density = 0.0

  for body_name in (CHASSIS_BODY_NAME, *WHEEL_BODIES):
    body = next(b for b in spec.bodies if b.name == body_name)
    mass, com, iquat, inertia = mass_model.fuse_nominal(body_name)
    body.mass = mass
    body.ipos = com.tolist()
    # The URDF importer fills fullinertia from the <inertial> block, and MuJoCo
    # rejects a body that specifies both that and a diagonal inertia. NaN in
    # slot 0 is how MjSpec marks fullinertia as unset (its own default), so
    # restoring that hands the diagonal + iquat pair below sole ownership.
    body.fullinertia = [float("nan"), 0.0, 0.0, 0.0, 0.0, 0.0]
    body.iquat = iquat.tolist()
    body.inertia = inertia.tolist()
    body.explicitinertial = True


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

  density=0 on the resulting geom, harmless now that _apply_mass_model writes
  explicit inertials (geom density no longer feeds inertia at all), but kept so
  the geom stays inert if that ever changes back.
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


# Payload defaults. Both are placeholders overwritten per-environment by
# javis/mdp/events.py (dr.geom_pos / dr.geom_size); they only decide what a
# plain, un-randomized scene shows.
PAYLOAD_DEFAULT_POS = (0.0, 0.0, 0.30)
PAYLOAD_DEFAULT_HALF_EXTENTS = (0.06, 0.06, 0.06)


def _add_payload_geom(chassis: mujoco.MjsBody) -> None:
  """Add the visible box standing in for whatever the robot is carrying.

  The payload's *mass* does not come from this geom -- it is a term in
  javis.mass_model's group vector, folded into the chassis's explicit inertial
  along with everything else, so mass, CoM and inertia stay mutually consistent
  when it changes. The geom is here purely so the load is (a) visible in the
  viewer and recorded videos and (b) has a pose that `dr.geom_pos` can
  randomize, which is then read back as the point-mass position.

  Non-colliding on purpose (contype/conaffinity 0): it is a mass proxy, not an
  obstacle. Giving it contact would make it collide with the chassis hull that
  already wraps the same volume.
  """
  chassis.add_geom(
    name=PAYLOAD_GEOM,
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=list(PAYLOAD_DEFAULT_HALF_EXTENTS),
    pos=list(PAYLOAD_DEFAULT_POS),
    density=0,
    rgba=[0.9, 0.45, 0.1, 0.65],
    contype=0,
    conaffinity=0,
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
# relative to the chassis link frame. Note: that CAD part name is a
# placeholder from the Onshape library -- the real sensor on the Jetson is a
# Bosch BMX160 (+ BMP388 baro), confirmed via the bmx160_bmp388_driver ROS2
# package running on-robot (see SIM2REAL.md sec 5).
# z updated 2026-08-12 for the redesigned CAD (0.26695 -> 0.159); x/rpy
# unchanged, so only mount height moved.
IMU_SITE_NAME = "imu"
IMU_POS = (-0.0675, 0.0, 0.159)
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
# z updated 2026-08-12 for the redesigned CAD (0.30795 -> 0.2).
D435_POS = (0.12105, 0.0, 0.2)
D435_RPY = (1.5708, 0.0, 1.5708)
D435_FOVY_DEG = 42.0

# robot.urdf has no real mass/inertia (onshape-to-robot exported 1e-9
# placeholders because no material densities are assigned in the Onshape
# assembly yet). get_spec() therefore writes explicit inertials from
# javis/mass_model.py, which models the chassis as named component groups
# (battery / printed PLA / Jetson / camera / ODrive / IMU / fasteners) rather
# than one averaged blob -- see _apply_mass_model above and mass_model's module
# docstring for why the averaged version put the CoM in the wrong place.


def get_spec() -> mujoco.MjSpec:
  """Load robot.urdf as an MjSpec, resolving package:// mesh URIs, naming the
  wheel collision geoms (URDF import otherwise leaves all geoms unnamed,
  which breaks mjlab's regex-based CollisionCfg/ActuatorCfg matching), adding
  a free joint on the chassis (URDF always assumes a fixed base), and writing
  explicit mass/inertia per body (see _apply_mass_model)."""
  mesh_cache: dict[str, trimesh.Trimesh] = {}
  urdf_text = JAVIS_URDF.read_text()
  # Absolute, not relative: MuJoCo resolves mesh paths against the URDF file's
  # own directory, so relative paths would force the temp copy to live inside
  # the package directory (which is what an earlier version did, leaving stray
  # files there on a crash). Absolute paths let it live in a real temp dir.
  assets_dir = JAVIS_DIR / "assets"
  fixed_urdf = re.sub(r"package://assets/", f"{assets_dir}/", urdf_text)

  with tempfile.TemporaryDirectory(prefix="javis_spec_") as tmp_dir:
    tmp_urdf = Path(tmp_dir) / "robot.urdf"
    tmp_urdf.write_text(fixed_urdf)
    spec = mujoco.MjSpec.from_file(str(tmp_urdf))

  # URDF import leaves every geom unnamed (empty string). mjlab's
  # CollisionCfg(disable_other_geoms=True) deduplicates by name when turning
  # off collision for non-matching geoms, so a shared "" name would only
  # disable one of the 342 chassis geoms instead of all of them. Give every
  # geom a unique name first, then name the two wheels distinctly.
  for i, geom in enumerate(spec.geoms):
    if not geom.name:
      geom.name = f"geom_{i}"

  for body in spec.bodies:
    if body.name in WHEEL_BODIES:
      body.geoms[0].name = f"{body.name}_collision"

  chassis = spec.worldbody.bodies[0]
  _add_chassis_collision_hull(spec, chassis, mesh_cache)
  _add_payload_geom(chassis)
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

  _apply_mass_model(spec)

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
WHEEL_TORQUE_CONSTANT_NM_PER_A = 0.207
WHEEL_CURRENT_LIMIT_A = 15.0
WHEEL_EFFORT_LIMIT_NM = WHEEL_TORQUE_CONSTANT_NM_PER_A * WHEEL_CURRENT_LIMIT_A  # 3.105
WHEEL_DAMPING_NM_PER_RAD_S = 0.028

WHEEL_ACTUATOR_CFG = BuiltinVelocityActuatorCfg(
  target_names_expr=WHEEL_JOINTS,
  damping=WHEEL_DAMPING_NM_PER_RAD_S,
  effort_limit=WHEEL_EFFORT_LIMIT_NM,
)

# Plain torque actuator, used when the ODrive PI velocity loop is simulated
# explicitly in javis/mdp/actions.py instead of leaning on MuJoCo's <velocity>
# actuator (which is proportional-only -- see that module's docstring for why
# the missing integral term matters so much under a varying payload).
WHEEL_MOTOR_ACTUATOR_CFG = BuiltinMotorActuatorCfg(
  target_names_expr=WHEEL_JOINTS,
  effort_limit=WHEEL_EFFORT_LIMIT_NM,
)

JAVIS_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(WHEEL_ACTUATOR_CFG,),
)

JAVIS_MOTOR_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(WHEEL_MOTOR_ACTUATOR_CFG,),
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


def get_javis_robot_cfg(torque_actuators: bool = False) -> EntityCfg:
  """The rover as an mjlab entity.

  Args:
    torque_actuators: use plain <motor> actuators instead of MuJoCo's built-in
      <velocity> ones. Required by javis/mdp/actions.py's OdriveVelocityAction,
      which runs the board's PI loop itself and commands torque directly.
  """
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(ROBOT_COLLISION,),
    spec_fn=get_spec,
    articulation=JAVIS_MOTOR_ARTICULATION if torque_actuators else JAVIS_ARTICULATION,
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
