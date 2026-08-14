"""JAVIS balance/velocity-tracking task as a plain MuJoCo + Gymnasium env.

A second, independent path to the same problem `javis/balance_task.py` solves
(mjlab/MuJoCo-Warp, rsl-rl, GPU-vectorized envs, wandb+tensorboard logging).
This one trades that throughput for a much shorter dependency chain: plain
`mujoco`, one robot per OS process, `stable-baselines3` PPO, and nothing but
stdout + CSV for logging (see `scripts/train_gym.py`,
`scripts/plot_gym_results.py`). Useful when mjlab/wandb aren't available, or
when the simpler stack is worth more than the GPU speedup.

It reuses the two framework-agnostic pieces of the main pipeline directly:
  - `javis.robot_constants.get_spec()` -- builds the plain `mujoco.MjSpec`
    (URDF import, mass model, IMU site+sensors, collision hull). Nothing in
    that function is mjlab-specific.
  - `javis.mass_model` -- the per-group mass/CoM/inertia fusion. Pure
    numpy/torch, no mjlab dependency either.

Everything mjlab supplied around those two pieces (the manager-based env,
GPU-batched domain randomization, the terrain generator, the rsl-rl action/
observation delay machinery) is reimplemented here directly against one
`mujoco.MjModel`/`MjData` pair, deliberately simplified in a few places
documented inline -- most notably: rough/noisy terrain is approximated by
tilting gravity by the slope angle instead of building a heightfield (which
is physically equivalent for a robot's own dynamics, and sidesteps the
heightfield-resolution contact warnings the mjlab task has seen), and there
is no observation/command latency or curriculum ramp (the main pipeline
already trains at its hardest fixed setting -- CURRICULUM_ENABLED=False in
javis/balance_task.py -- so this mirrors that rather than losing anything).

Physics/DR/reward numbers are ported from, and should be kept in sync with:
  javis/sim_config.py    -- every randomization range
  javis/mdp/actions.py   -- the ODrive PI velocity loop
  javis/mdp/rewards.py   -- pitch_rate_l2 / wheel_slip / torque_saturation
  javis/mdp/events.py    -- the mass-sampling + feasibility-filter algorithm
  javis/balance_task.py  -- control rate, episode length, reward weights
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import gymnasium as gym
import mujoco
import numpy as np
import torch
from gymnasium import spaces

from . import mass_model
from .robot_constants import (
  CHASSIS_BODY_NAME,
  CHASSIS_COLLISION_GEOM,
  PAYLOAD_GEOM,
  WHEEL_BODIES,
  WHEEL_COLLISION_GEOMS,
  WHEEL_EFFORT_LIMIT_NM,
  WHEEL_JOINTS,
  WHEEL_RADIUS_M,
  get_spec,
)
from .sim_config import JavisDomainCfg, lerp_range

GRAVITY = 9.81

# Same numbers as javis/balance_task.py: 100 Hz control, 400 Hz physics.
CONTROL_HZ = 100.0
PHYSICS_TIMESTEP_S = 0.0025
DECIMATION = round(1.0 / (CONTROL_HZ * PHYSICS_TIMESTEP_S))
EPISODE_LENGTH_S = 20.0
MAX_CONTROL_STEPS = round(EPISODE_LENGTH_S * CONTROL_HZ)

OBS_HISTORY_SECONDS = 0.24
OBS_HISTORY_LENGTH = round(OBS_HISTORY_SECONDS * CONTROL_HZ)
OBS_FRAME_DIM = 16  # base_lin_vel(3) + base_ang_vel(3) + proj_gravity(3) + wheel_vel(2) + last_action(2) + command(3)

ACTION_SCALE = 5.0  # rad/s per unit of policy output, matches balance_task.py

_TWO_PI = 2.0 * math.pi


def _euler_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
  cr, sr = math.cos(roll / 2), math.sin(roll / 2)
  cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
  cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
  return np.array([
    cr * cp * cy + sr * sp * sy,
    sr * cp * cy - cr * sp * sy,
    cr * sp * cy + sr * cp * sy,
    cr * cp * sy - sr * sp * cy,
  ])


def _uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
  return float(rng.uniform(lo, hi))


@dataclass
class _WheelPI:
  """Per-wheel ODrive PI velocity-loop state. See javis/mdp/actions.py."""

  kp: np.ndarray  # (2,) N*m/(rad/s)
  ki: np.ndarray  # (2,) N*m/rad
  integrator: np.ndarray
  ramped_target: np.ndarray
  last_torque: np.ndarray


class JavisBalanceEnv(gym.Env):
  """Single-robot balance + twist-tracking env under randomized load.

  Observation: a flattened `OBS_HISTORY_LENGTH`-frame stack of
  [base_lin_vel(3), base_ang_vel(3), projected_gravity(3), wheel_vel(2),
  last_action(2), command(3)] -- exactly javis/balance_task.py's actor
  observation group, so a policy trained here and one trained on the mjlab
  task see the same inputs.

  Action: 2 values in [-1, 1], scaled by ACTION_SCALE to a wheel angular
  velocity target in rad/s, tracked by a simulated ODrive PI loop.
  """

  metadata = {"render_modes": ["rgb_array"]}

  def __init__(
    self,
    domain: JavisDomainCfg | None = None,
    difficulty: float = 1.0,
    render_mode: str | None = None,
    seed: int | None = None,
    force_slope_rad: float | None = None,
    noise_scale: float = 1.0,
    terrain_type: str = "flat",
    terrain_slope_deg: float = 12.0,
    init_xy_jitter: float = 0.5,
  ):
    """
    force_slope_rad: pin every episode's terrain-proxy slope to this exact
      angle (heading fixed at 0, i.e. "downhill" is always +x) instead of
      randomly drawing flat/slope each reset. For scripted eval/video
      scenarios, not training -- leave None there. Ignored (treated as 0)
      when terrain_type != "flat", since the ground itself now provides the
      tilt -- see terrain_type below.
    noise_scale: multiplies every noise/bias term in `_compute_frame`
      (IMU white noise, per-episode IMU bias, encoder noise). 1.0 matches
      training; scripted scenarios can dial this to compare a policy's
      behavior under lighter/heavier sensor noise than it trained on.
    terrain_type: "flat" (default, matches training -- an infinite flat
      plane, slope simulated via force_slope_rad's gravity tilt if set),
      "ramp" (a real tilted-plane geom at terrain_slope_deg -- an actual
      sloped floor, for scripted video/eval where a visually flat floor
      during a "slope" scenario would be misleading), or "rough" (a
      randomly-bumpy heightfield, peak-to-peak matching
      JavisDomainCfg.terrain.rough_noise_range's upper bound). Fixed at
      construction time (MuJoCo geometry can't change shape after compile),
      so switching terrain means building a new env, not a mid-run swap.
      All three share the same checkered ground material, purely so motion
      reads clearly on video -- has no physical effect.
    init_xy_jitter: half-width (m) of the uniform x/y spawn range in
      reset(). Training uses the full 0.5 m (matches javis/balance_task.py's
      reset_base); scripted video on a "ramp"/"rough" terrain built around
      the origin passes a smaller value so the robot reliably spawns ON the
      terrain feature instead of near its edge.
    """
    super().__init__()
    self.domain = domain or JavisDomainCfg()
    self.difficulty = difficulty
    self.render_mode = render_mode
    self._np_random = np.random.default_rng(seed)
    self._force_slope_rad = force_slope_rad
    self._noise_scale = noise_scale
    self._terrain_type = terrain_type
    self._terrain_slope_deg = terrain_slope_deg
    self._init_xy_jitter = init_xy_jitter

    self._build_model()

    self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
    obs_dim = OBS_HISTORY_LENGTH * OBS_FRAME_DIM
    self.observation_space = spaces.Box(
      -np.inf, np.inf, shape=(obs_dim,), dtype=np.float32
    )

    self._renderer: mujoco.Renderer | None = None

  # -- model construction --------------------------------------------------

  def _build_model(self) -> None:
    spec = get_spec()

    keep_colliding = {*WHEEL_COLLISION_GEOMS, CHASSIS_COLLISION_GEOM}
    for geom in spec.geoms:
      if geom.name in keep_colliding:
        geom.contype = 1
        geom.conaffinity = 1
        geom.condim = 3
        geom.friction = [1.0, 0.005, 0.0001]
      else:
        geom.contype = 0
        geom.conaffinity = 0
    # Priority: wheels dig in over the chassis hull, same as ROBOT_COLLISION.
    for name in WHEEL_COLLISION_GEOMS:
      next(g for g in spec.geoms if g.name == name).priority = 1

    # Checkered material on every terrain variant purely so translation/
    # rotation reads clearly on video against an otherwise-featureless plane
    # -- no physical effect (it's a texture, not geometry).
    spec.add_texture(
      name="ground_checker",
      type=mujoco.mjtTexture.mjTEXTURE_2D,
      builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
      rgb1=[0.35, 0.4, 0.42],
      rgb2=[0.55, 0.6, 0.62],
      width=300,
      height=300,
    )
    spec.add_material(
      name="ground_mat", textures=["", "ground_checker"], texrepeat=[6, 6], texuniform=True
    )

    if self._terrain_type == "rough":
      # A fixed random bump field (not resampled per-episode -- geometry is
      # baked in at construction, see __init__'s terrain_type docstring).
      # Peak-to-peak matches JavisDomainCfg.terrain.rough_noise_range's
      # upper bound (2 cm), same magnitude the mjlab task's rough terrain
      # uses, just a single static realization instead of one per episode.
      nrow = ncol = 60
      terrain_rng = np.random.default_rng(0)
      heights = terrain_rng.uniform(0.0, 1.0, size=(nrow, ncol))
      spec.add_hfield(
        name="rough_hfield", size=[6.0, 6.0, 0.02, 0.05], nrow=nrow, ncol=ncol,
        userdata=heights.flatten().tolist(),
      )
      spec.worldbody.add_geom(
        name="ground", type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname="rough_hfield",
        contype=1, conaffinity=1, material="ground_mat",
      )
    elif self._terrain_type == "ramp":
      # A real tilted-plane geom (finite box, not mjGEOM_PLANE which MuJoCo
      # always treats as infinite/untilted for its collision shortcut) --
      # rotated about y by terrain_slope_deg, positioned so its top surface
      # still passes through world (0, 0, 0), matching the flat case's
      # convention that INIT_STATE's spawn height is measured from z=0.
      slope_rad = math.radians(self._terrain_slope_deg)
      half_thick = 0.1
      quat = _euler_to_quat(0.0, -slope_rad, 0.0)
      # Box center placed so its top face (local +z, offset half_thick) still
      # passes through world (0, 0, 0) after rotation -- i.e. solve
      # pos + R @ [0, 0, half_thick] == 0 for pos, with R = Ry(-slope_rad).
      pos = [half_thick * math.sin(slope_rad), 0.0, -half_thick * math.cos(slope_rad)]
      spec.worldbody.add_geom(
        name="ground", type=mujoco.mjtGeom.mjGEOM_BOX, size=[6.0, 3.0, half_thick],
        pos=pos, quat=quat.tolist(),
        contype=1, conaffinity=1, material="ground_mat",
      )
    else:
      spec.worldbody.add_geom(
        name="ground", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[20, 20, 0.1],
        contype=1, conaffinity=1, material="ground_mat",
      )

    # Plain torque actuators, one per wheel joint -- the ODrive PI loop below
    # computes the torque itself and writes it straight into data.ctrl, the
    # same "motor actuator, PI loop owns the math" split as
    # javis/mdp/actions.py's OdriveVelocityAction.
    for joint in WHEEL_JOINTS:
      spec.add_actuator(
        name=f"{joint}_motor",
        target=joint,
        trntype=mujoco.mjtTrn.mjTRN_JOINT,
        gear=[1, 0, 0, 0, 0, 0],
        ctrlrange=[-WHEEL_EFFORT_LIMIT_NM, WHEEL_EFFORT_LIMIT_NM],
        forcerange=[-WHEEL_EFFORT_LIMIT_NM, WHEEL_EFFORT_LIMIT_NM],
      )

    spec.option.timestep = PHYSICS_TIMESTEP_S

    self.model = spec.compile()
    self.data = mujoco.MjData(self.model)

    self._chassis_id = self.model.body(CHASSIS_BODY_NAME).id
    self._chassis_geom_id = self.model.geom(CHASSIS_COLLISION_GEOM).id
    self._payload_geom_id = self.model.geom(PAYLOAD_GEOM).id
    self._wheel_body_ids = [self.model.body(b).id for b in WHEEL_BODIES]
    self._wheel_geom_ids = [self.model.geom(g).id for g in WHEEL_COLLISION_GEOMS]
    self._wheel_dof_adr = [self.model.joint(j).dofadr[0] for j in WHEEL_JOINTS]
    self._wheel_qpos_adr = [self.model.joint(j).qposadr[0] for j in WHEEL_JOINTS]
    self._wheel_actuator_ids = [
      self.model.actuator(f"{j}_motor").id for j in WHEEL_JOINTS
    ]
    free_joints = np.where(self.model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)[0]
    assert free_joints.size == 1, "expected exactly one free joint on the chassis"
    self._free_qpos_adr = self.model.jnt_qposadr[free_joints[0]]
    self._free_dof_adr = self.model.jnt_dofadr[free_joints[0]]

    self._imu_gyro_adr = self.model.sensor("imu_ang_vel").adr[0]
    self._imu_vel_adr = self.model.sensor("imu_lin_vel").adr[0]

    self._moments = mass_model.get_moments()[mass_model.CHASSIS_BODY]
    self._nominal = mass_model.nominal_mass_vector(mass_model.CHASSIS_BODY)
    self._payload_idx = self._moments.index("payload")
    self._wiring_idx = self._moments.index("wiring_misc")
    self._mesh_idx = torch.tensor(
      [i for i, g in enumerate(self._moments.groups) if g not in ("wiring_misc", "payload")],
      dtype=torch.long,
    )
    self._wheel_nominal_kg = float(
      mass_model.nominal_masses(WHEEL_BODIES[0])["wheel"]
    )
    # Cached at nominal mass, right after compile -- reset() always scales
    # FROM this, never from the current model.body_inertia, since that value
    # is itself overwritten by the previous episode's random wheel mass and
    # scaling from it would compound the DR ratio across episodes instead of
    # resampling it independently each time.
    self._wheel_nominal_inertia = [
      self.model.body_inertia[bid].copy() for bid in self._wheel_body_ids
    ]

  # -- domain randomization: mass/CoM ---------------------------------------

  def _sample_load(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mcfg, pcfg = self.domain.mass, self.domain.payload
    level = self.difficulty
    rng = self._np_random
    g = self._moments.groups

    masses = torch.zeros(len(g))
    for i, group in enumerate(g):
      if group in ("wiring_misc", "payload"):
        continue
      lo, hi = lerp_range(mcfg.group_scale_easy[group], mcfg.group_scale_hard[group], level)
      masses[i] = self._nominal[i] * _uniform(rng, lo, hi)

    lo, hi = lerp_range(mcfg.wiring_kg_easy, mcfg.wiring_kg_hard, level)
    masses[self._wiring_idx] = _uniform(rng, lo, hi)
    wiring_pos = torch.tensor([
      _uniform(rng, *lerp_range(mcfg.wiring_pos_easy[a], mcfg.wiring_pos_hard[a], level))
      for a in range(3)
    ])

    lo, hi = lerp_range(mcfg.chassis_total_easy, mcfg.chassis_total_hard, level)
    target_total = _uniform(rng, lo, hi)
    mesh_sum = masses[self._mesh_idx].sum()
    mesh_target = max(target_total - float(masses[self._wiring_idx]), 0.5)
    masses[self._mesh_idx] *= mesh_target / mesh_sum.clamp_min(1e-6)

    lo, hi = lerp_range(pcfg.mass_kg_easy, pcfg.mass_kg_hard, level)
    masses[self._payload_idx] = _uniform(rng, lo, hi)
    payload_pos = torch.tensor([
      _uniform(rng, *lerp_range(pcfg.pos_x_easy, pcfg.pos_x_hard, level)),
      _uniform(rng, *lerp_range(pcfg.pos_y_easy, pcfg.pos_y_hard, level)),
      _uniform(rng, *lerp_range(pcfg.pos_z_easy, pcfg.pos_z_hard, level)),
    ])
    return masses, payload_pos, wiring_pos

  def _max_lean_rad(self, total_mass: float) -> float:
    force_max = 2.0 * WHEEL_EFFORT_LIMIT_NM / WHEEL_RADIUS_M
    return math.atan(force_max / (GRAVITY * max(total_mass, 1e-3)))

  def _required_lean_rad(self, com_x: float, com_height: float) -> float:
    return abs(math.atan2(com_x, max(com_height, 1e-3))) + self._slope_rad

  def _sample_load_feasible(self) -> None:
    fcfg = self.domain.feasibility
    masses, payload_pos, wiring_pos = self._sample_load()

    if fcfg.enabled:
      margin = math.radians(fcfg.margin_deg)
      for _ in range(fcfg.max_resample_attempts):
        chassis_mass, com = mass_model.fuse_com_only(
          masses, self._moments,
          point_positions={"payload": payload_pos, "wiring_misc": wiring_pos},
        )
        com_height = float(com[2]) + WHEEL_RADIUS_M
        required = self._required_lean_rad(float(com[0]), com_height) + margin
        available = self._max_lean_rad(float(chassis_mass) + self._wheel_total_kg)
        if required <= available:
          break
        masses, payload_pos, wiring_pos = self._sample_load()
      else:
        masses[self._payload_idx] = 0.0

    self._masses, self._payload_pos, self._wiring_pos = masses, payload_pos, wiring_pos
    self._write_chassis()

  def _write_chassis(self) -> None:
    mass, ipos, iquat, inertia = mass_model.fuse(
      self._masses, self._moments,
      point_positions={"payload": self._payload_pos, "wiring_misc": self._wiring_pos},
    )
    bid = self._chassis_id
    self.model.body_mass[bid] = float(mass)
    self.model.body_ipos[bid] = ipos.numpy()
    self.model.body_iquat[bid] = iquat.numpy()
    self.model.body_inertia[bid] = inertia.numpy()

    payload_kg = float(self._masses[self._payload_idx])
    pcfg = self.domain.payload
    frac = min(payload_kg / pcfg.max_kg_for_size, 1.0)
    half = pcfg.size_at_zero_kg + frac * (pcfg.size_at_max_kg - pcfg.size_at_zero_kg)
    self.model.geom_pos[self._payload_geom_id] = self._payload_pos.numpy()
    self.model.geom_size[self._payload_geom_id] = [half, half, half]

  # -- reset -----------------------------------------------------------------

  def reset(self, *, seed: int | None = None, options: dict | None = None):
    super().reset(seed=seed)
    if seed is not None:
      self._np_random = np.random.default_rng(seed)
    rng = self._np_random

    # Wheel mass DR first (nominal inertia scaled by mass ratio, same
    # "one homogeneous group" shortcut as javis/mdp/events.py).
    mcfg = self.domain.mass
    lo, hi = lerp_range(mcfg.wheel_scale_easy, mcfg.wheel_scale_hard, self.difficulty)
    total = 0.0
    for idx, bid in enumerate(self._wheel_body_ids):
      scale = _uniform(rng, lo, hi)
      m = self._wheel_nominal_kg * scale
      self.model.body_inertia[bid] = self._wheel_nominal_inertia[idx] * scale
      self.model.body_mass[bid] = m
      total += m
    self._wheel_total_kg = total

    # Terrain proxy: tilt gravity by a random slope angle instead of a
    # heightfield -- see module docstring. Skipped entirely when
    # terrain_type != "flat": the ground geometry itself provides the tilt/
    # roughness there, so gravity stays plain vertical to avoid double-
    # counting the slope.
    if self._terrain_type != "flat":
      self._slope_rad = 0.0
      self.model.opt.gravity = [0.0, 0.0, -GRAVITY]
    elif self._force_slope_rad is not None:
      self._slope_rad = self._force_slope_rad
      heading = 0.0
      s = math.sin(self._slope_rad)
      self.model.opt.gravity = [
        GRAVITY * s * math.cos(heading), GRAVITY * s * math.sin(heading), -GRAVITY * math.cos(self._slope_rad),
      ]
    else:
      tcfg = self.domain.terrain
      if _uniform(rng, 0.0, 1.0) < tcfg.flat_proportion:
        self._slope_rad = 0.0
      else:
        ratio = _uniform(rng, *tcfg.slope_range)
        self._slope_rad = math.atan(ratio)
      heading = _uniform(rng, 0.0, 2 * math.pi)
      s = math.sin(self._slope_rad)
      self.model.opt.gravity = [
        GRAVITY * s * math.cos(heading), GRAVITY * s * math.sin(heading), -GRAVITY * math.cos(self._slope_rad),
      ]

    self._sample_load_feasible()

    # Contact / drivetrain DR.
    ccfg = self.domain.contact
    lo, hi = lerp_range(ccfg.friction_easy, ccfg.friction_hard, self.difficulty)
    for gid in self._wheel_geom_ids:
      self.model.geom_friction[gid, 0] = _uniform(rng, lo, hi)
    for adr in self._wheel_dof_adr:
      self.model.dof_damping[adr] = _uniform(rng, 0.0, 0.03)

    dcfg = self.domain.drivetrain
    lo, hi = lerp_range(dcfg.gain_scale_easy, dcfg.gain_scale_hard, self.difficulty)
    scale = np.array([_uniform(rng, lo, hi), _uniform(rng, lo, hi)])
    self._pi = _WheelPI(
      kp=dcfg.kp_nm_per_rad_s * scale,
      ki=dcfg.ki_nm_per_rad * scale,
      integrator=np.zeros(2),
      ramped_target=np.zeros(2),
      last_torque=np.zeros(2),
    )

    # Initial pose/velocity: already-leaning, already-moving. See
    # javis/balance_task.py's reset_base -- same ranges.
    mujoco.mj_resetData(self.model, self.data)
    qpos = self.data.qpos
    j = self._init_xy_jitter
    qpos[self._free_qpos_adr + 0] = _uniform(rng, -j, j)
    qpos[self._free_qpos_adr + 1] = _uniform(rng, -j, j)
    # A bit more clearance than the flat-ground 0.12 m when the terrain
    # itself has relief (ramp/rough) -- INIT_STATE's convention measures
    # from a locally-flat z=0, which a bumpy hfield or a jittered spawn
    # point on a ramp doesn't exactly satisfy; contact resolves the rest.
    spawn_clearance = 0.12 if self._terrain_type == "flat" else 0.18
    qpos[self._free_qpos_adr + 2] = spawn_clearance + _uniform(rng, 0.0, 0.02)
    quat = _euler_to_quat(
      _uniform(rng, -0.35, 0.35), _uniform(rng, -0.35, 0.35), _uniform(rng, -math.pi, math.pi)
    )
    qpos[self._free_qpos_adr + 3 : self._free_qpos_adr + 7] = quat
    for adr in self._wheel_qpos_adr:
      qpos[adr] = 0.0

    qvel = self.data.qvel
    qvel[self._free_dof_adr + 0] = _uniform(rng, -0.3, 0.3)
    qvel[self._free_dof_adr + 1] = _uniform(rng, -0.3, 0.3)
    qvel[self._free_dof_adr + 2] = _uniform(rng, -0.15, 0.05)
    qvel[self._free_dof_adr + 3] = _uniform(rng, -0.3, 0.3)
    qvel[self._free_dof_adr + 4] = _uniform(rng, -0.3, 0.3)
    qvel[self._free_dof_adr + 5] = _uniform(rng, -0.3, 0.3)
    for adr in self._wheel_dof_adr:
      qvel[adr] = _uniform(rng, -0.5, 0.5)

    mujoco.mj_forward(self.model, self.data)

    self._pi.ramped_target[:] = [self.data.qvel[a] for a in self._wheel_dof_adr]

    # Persistent per-episode IMU bias (resampled once, held for the episode) --
    # see javis/balance_task.py's imu_noise() for the same step/bias figures.
    self._imu_lin_bias = self._noise_scale * rng.uniform(-0.03, 0.03, size=3)
    self._imu_ang_bias = self._noise_scale * rng.uniform(-0.03, 0.03, size=3)
    self._grav_bias = self._noise_scale * rng.uniform(-0.015, 0.015, size=3)

    self._command = self._sample_command()
    self._next_command_t = _uniform(rng, 3.0, 8.0)
    self._last_action = np.zeros(2, dtype=np.float32)
    self._step_count = 0
    self._next_push_t = _uniform(rng, 2.0, 5.0)
    pcfg = self.domain.payload
    self._next_payload_change_t = _uniform(rng, *pcfg.change_interval_s)
    self._t = 0.0

    frame = self._compute_frame()
    self._history = np.tile(frame, (OBS_HISTORY_LENGTH, 1))
    return self._history.flatten().astype(np.float32), {}

  def _sample_command(self) -> np.ndarray:
    rng = self._np_random
    if _uniform(rng, 0.0, 1.0) < 0.2:
      return np.zeros(3, dtype=np.float32)
    return np.array([
      _uniform(rng, -0.5, 0.5), 0.0, _uniform(rng, -1.0, 1.0)
    ], dtype=np.float32)

  # -- step --------------------------------------------------------------

  def step(self, action: np.ndarray):
    action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
    target = action * ACTION_SCALE
    rng = self._np_random

    for _ in range(DECIMATION):
      dt = PHYSICS_TIMESTEP_S
      max_delta = self.domain.drivetrain.vel_ramp_rad_s2 * dt
      delta = np.clip(target - self._pi.ramped_target, -max_delta, max_delta)
      self._pi.ramped_target = self._pi.ramped_target + delta

      measured = np.array([self.data.qvel[a] for a in self._wheel_dof_adr])
      error = self._pi.ramped_target - measured

      candidate = self._pi.integrator + self._pi.ki * error * dt
      unsaturated = np.abs(self._pi.kp * error + candidate) <= WHEEL_EFFORT_LIMIT_NM
      winding_back = np.abs(candidate) < np.abs(self._pi.integrator)
      self._pi.integrator = np.where(unsaturated | winding_back, candidate, self._pi.integrator)

      torque = np.clip(
        self._pi.kp * error + self._pi.integrator, -WHEEL_EFFORT_LIMIT_NM, WHEEL_EFFORT_LIMIT_NM
      )
      self._pi.last_torque = torque
      for i, aid in enumerate(self._wheel_actuator_ids):
        self.data.ctrl[aid] = torque[i]

      mujoco.mj_step(self.model, self.data)
      self._t += dt

    # Periodic push perturbation -- javis/balance_task.py's push_robot.
    if self._t >= self._next_push_t:
      self.data.qvel[self._free_dof_adr + 0] += _uniform(rng, -0.3, 0.3)
      self.data.qvel[self._free_dof_adr + 1] += _uniform(rng, -0.3, 0.3)
      self.data.qvel[self._free_dof_adr + 3] += _uniform(rng, -0.3, 0.3)
      self.data.qvel[self._free_dof_adr + 4] += _uniform(rng, -0.3, 0.3)
      self._next_push_t = self._t + _uniform(rng, 2.0, 5.0)

    # Mid-episode payload change -- "picks something up while driving."
    pcfg = self.domain.payload
    if pcfg.change_enabled and self._t >= self._next_payload_change_t:
      if _uniform(rng, 0.0, 1.0) < pcfg.change_prob_hard:
        lo, hi = lerp_range(pcfg.mass_kg_easy, pcfg.mass_kg_hard, self.difficulty)
        self._masses[self._payload_idx] = _uniform(rng, lo, hi)
        self._write_chassis()
      self._next_payload_change_t = self._t + _uniform(rng, *pcfg.change_interval_s)

    self._step_count += 1
    if self._t >= self._next_command_t:
      self._command = self._sample_command()
      self._next_command_t = self._t + _uniform(rng, 3.0, 8.0)

    frame = self._compute_frame(last_action=action)
    self._history = np.roll(self._history, -1, axis=0)
    self._history[-1] = frame
    self._last_action = action

    reward, reward_terms = self._compute_reward(action)
    terminated = self._check_terminated()
    truncated = self._step_count >= MAX_CONTROL_STEPS

    obs = self._history.flatten().astype(np.float32)
    info = {"reward_terms": reward_terms}
    return obs, reward, terminated, truncated, info

  # -- observations, reward, termination -----------------------------------

  def _compute_frame(self, last_action: np.ndarray | None = None) -> np.ndarray:
    lin_vel = self.data.sensordata[self._imu_vel_adr : self._imu_vel_adr + 3].copy()
    ang_vel = self.data.sensordata[self._imu_gyro_adr : self._imu_gyro_adr + 3].copy()

    xmat = self.data.xmat[self._chassis_id].reshape(3, 3)
    gravity_dir = self.model.opt.gravity / max(np.linalg.norm(self.model.opt.gravity), 1e-9)
    proj_gravity = xmat.T @ gravity_dir

    rng = self._np_random
    s = self._noise_scale
    lin_vel = lin_vel + self._imu_lin_bias + s * rng.uniform(-0.1, 0.1, size=3)
    ang_vel = ang_vel + self._imu_ang_bias + s * rng.uniform(-0.1, 0.1, size=3)
    proj_gravity = proj_gravity + self._grav_bias + s * rng.uniform(-0.03, 0.03, size=3)

    wheel_vel = np.array([self.data.qvel[a] for a in self._wheel_dof_adr])
    wheel_vel = wheel_vel + s * rng.uniform(-0.5, 0.5, size=2)

    action = self._last_action if last_action is None else last_action

    return np.concatenate([lin_vel, ang_vel, proj_gravity, wheel_vel, action, self._command]).astype(np.float32)

  def _forward_speed(self) -> float:
    xmat = self.data.xmat[self._chassis_id].reshape(3, 3)
    world_vel = self.data.cvel[self._chassis_id][3:6]
    return float((xmat.T @ world_vel)[0])

  def _compute_reward(self, action: np.ndarray) -> tuple[float, dict[str, float]]:
    lin_vel_body = np.array([self._forward_speed(), 0.0, self.data.qvel[self._free_dof_adr + 2]])

    xy_err = (self._command[0] - lin_vel_body[0]) ** 2 + lin_vel_body[2] ** 2
    r_lin = math.exp(-xy_err / 0.25)

    xmat = self.data.xmat[self._chassis_id].reshape(3, 3)
    ang_vel_local = xmat.T @ self.data.cvel[self._chassis_id][0:3]
    ang_z_err = (self._command[2] - ang_vel_local[2]) ** 2
    ang_xy_err = ang_vel_local[0] ** 2 + ang_vel_local[1] ** 2
    r_ang = math.exp(-(ang_z_err + ang_xy_err) / 0.5)

    gravity_dir = self.model.opt.gravity / max(np.linalg.norm(self.model.opt.gravity), 1e-9)
    proj_gravity = xmat.T @ gravity_dir
    r_upright = math.exp(-(proj_gravity[0] ** 2 + proj_gravity[1] ** 2) / 0.5)

    r_alive = 1.0
    r_action_rate = -float(np.sum((action - self._last_action) ** 2))
    r_pitch_rate = -float(ang_vel_local[1] ** 2)

    surface_speed = float(np.mean([self.data.qvel[a] for a in self._wheel_dof_adr])) * WHEEL_RADIUS_M
    r_slip = -float((surface_speed - self._forward_speed()) ** 2)

    torque_frac = self._pi.last_torque / WHEEL_EFFORT_LIMIT_NM
    r_torque_sat = -float(np.mean(torque_frac ** 2))

    terms = {
      "track_linear_velocity": 1.5 * r_lin,
      "track_angular_velocity": 1.0 * r_ang,
      "upright": 1.0 * r_upright,
      "is_alive": 0.5 * r_alive,
      "action_rate_l2": 0.05 * r_action_rate,
      "pitch_rate_l2": 0.02 * r_pitch_rate,
      "wheel_slip": 0.05 * r_slip,
      "torque_saturation": 0.02 * r_torque_sat,
    }
    return sum(terms.values()), terms

  def _check_terminated(self) -> bool:
    xmat = self.data.xmat[self._chassis_id].reshape(3, 3)
    gravity_dir = self.model.opt.gravity / max(np.linalg.norm(self.model.opt.gravity), 1e-9)
    proj_gravity = xmat.T @ gravity_dir
    fell_over = math.acos(np.clip(-proj_gravity[2], -1.0, 1.0)) > math.radians(60.0)

    chassis_down = False
    for i in range(self.data.ncon):
      c = self.data.contact[i]
      if c.geom1 == self._chassis_geom_id or c.geom2 == self._chassis_geom_id:
        chassis_down = True
        break
    return fell_over or chassis_down

  # -- rendering -------------------------------------------------------------

  def render(self):
    if self.render_mode != "rgb_array":
      return None
    if self._renderer is None:
      self._renderer = mujoco.Renderer(self.model, height=480, width=640)
    self._renderer.update_scene(self.data)
    return self._renderer.render()

  def close(self):
    if self._renderer is not None:
      self._renderer.close()
      self._renderer = None


# Registered under gymnasium's own registry (distinct from mjlab's separate
# task registry, javis/tasks.py) so any tool that expects a plain `gym.make`
# id -- e.g. github.com/Bigkatoan/SRL's `srl-train --env <id>`, which
# resolves envs via `gym.make()` -- can reach this env without importing
# javis.gym_env directly. `id="Javis-Balance-v0"` is picked at import time;
# importing this module (or anything that imports it, e.g. via
# scripts/train_srl.py) is what makes the id resolvable.
gym.register(id="Javis-Balance-v0", entry_point=JavisBalanceEnv)
