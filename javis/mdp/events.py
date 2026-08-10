"""Payload and component-mass domain randomization for the JAVIS balance task.

What this does that mjlab's built-in DR cannot
----------------------------------------------
mjlab offers `dr.body_mass` (mass only -- it warns that this is physically
inconsistent on its own), `dr.body_ipos` (CoM only) and `dr.pseudo_inertia`
(a consistent but *unstructured* random perturbation of the whole inertial).

None of those express the thing we actually want to randomize: "the battery
weighs 3.4 kg +-10%, the printed parts are one spool +-50%, and there is an
unknown 0-10 kg payload bolted on somewhere between 5 and 60 cm up." Those are
statements about identifiable components at known locations, and the resulting
(mass, CoM, inertia) triple is fully determined by them -- not something to
perturb independently.

`javis.mass_model` precomputes each component group's unit-mass moments, which
makes the fused body inertial a linear function of the group masses. This module
samples those masses and writes the exact result into the per-environment model.
Every environment therefore holds a mass distribution that some real build of
this robot could actually have.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from mjlab.envs.mdp.dr._core import _get_entity_indices
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .. import mass_model
from ..robot_constants import (
  CHASSIS_BODY_NAME,
  PAYLOAD_GEOM,
  WHEEL_BODIES,
  WHEEL_EFFORT_LIMIT_NM,
  WHEEL_RADIUS_M,
)
from ..sim_config import DifficultyState, JavisDomainCfg, Range, lerp_range

try:  # pragma: no cover - depends on mjlab internals
  from mjlab.envs.mdp.dr.body import _eigh_3x3_jacobi as _EIGH
except ImportError:  # pragma: no cover
  # torch.linalg.eigh works but permanently reserves several GB of VRAM for
  # cuSOLVER on first call, which is why mjlab ships its own 3x3 Jacobi solver.
  _EIGH = None

GRAVITY = 9.81
_STATE_ATTR = "_javis_dr_state"
_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

# Model fields every load-randomizing term touches. body_* needs the full
# `set_const` recompute (mass matrix, subtree masses); the geom_* fields are the
# payload box's cosmetic pose/size/colour.
_LOAD_FIELDS = (
  "body_mass",
  "body_ipos",
  "body_inertia",
  "body_iquat",
  "geom_pos",
  "geom_size",
  "geom_rbound",
  "geom_aabb",
  "geom_rgba",
)


##
# Per-environment state.
##


@dataclass
class DrState:
  """What each environment is currently built out of.

  Kept between events so the mid-episode payload change can alter one number
  without resampling (and thereby teleporting) the rest of the robot.
  """

  group_masses: torch.Tensor  # (num_envs, G) chassis group masses, kg
  payload_pos: torch.Tensor  # (num_envs, 3) mount position, chassis frame
  wiring_pos: torch.Tensor  # (num_envs, 3) harness lump position
  wheel_masses: torch.Tensor  # (num_envs, 2) left/right wheel mass, kg
  # Share of the last reset batch that still exceeded the feasibility cap after
  # the resample budget ran out. Reported by the curriculum term.
  last_infeasible_frac: float = 0.0

  def total_mass(self) -> torch.Tensor:
    return self.group_masses.sum(dim=-1) + self.wheel_masses.sum(dim=-1)


def get_state(env) -> DrState:
  """Fetch (creating on first use) this environment's load state."""
  state = getattr(env, _STATE_ATTR, None)
  if state is not None:
    return state

  device = env.device
  nominal = mass_model.nominal_mass_vector(mass_model.CHASSIS_BODY, device)
  moments = mass_model.get_moments()[mass_model.CHASSIS_BODY]
  wheel_nominal = mass_model.nominal_masses("wheel")["wheel"]

  def point_default(group: str) -> torch.Tensor:
    idx = moments.index(group)
    return torch.tensor(moments.unit_com[idx], dtype=torch.float32, device=device)

  state = DrState(
    group_masses=nominal.expand(env.num_envs, -1).clone(),
    payload_pos=point_default("payload").expand(env.num_envs, -1).clone(),
    wiring_pos=point_default("wiring_misc").expand(env.num_envs, -1).clone(),
    wheel_masses=torch.full((env.num_envs, 2), wheel_nominal, device=device),
  )
  setattr(env, _STATE_ATTR, state)
  return state


##
# Sampling helpers.
##


def _uniform(shape: tuple[int, ...], rng: Range, device) -> torch.Tensor:
  lo, hi = rng
  if hi <= lo:
    return torch.full(shape, lo, device=device)
  return torch.empty(shape, device=device).uniform_(lo, hi)


def _resolve_env_ids(env, env_ids: torch.Tensor | None) -> torch.Tensor:
  if env_ids is None:
    return torch.arange(env.num_envs, device=env.device, dtype=torch.long)
  return env_ids.to(env.device, dtype=torch.long)


def _payload_visual(
  payload_mass: torch.Tensor, cfg: JavisDomainCfg
) -> tuple[torch.Tensor, torch.Tensor]:
  """Box half-extent and colour for a payload of the given mass.

  Purely cosmetic -- the geom has density 0 and no contact -- but it is the
  only way to see at a glance, in the viewer or a recorded video, which robot
  is carrying what.
  """
  p = cfg.payload
  frac = (payload_mass / p.max_kg_for_size).clamp(0.0, 1.0)
  half = p.size_at_zero_kg + frac * (p.size_at_max_kg - p.size_at_zero_kg)
  size = half[..., None].expand(*half.shape, 3).clone()

  rgba = torch.empty(*frac.shape, 4, device=frac.device)
  rgba[..., 0] = 0.25 + 0.65 * frac  # light blue -> deep red with load
  rgba[..., 1] = 0.55 - 0.40 * frac
  rgba[..., 2] = 0.85 - 0.75 * frac
  rgba[..., 3] = torch.where(payload_mass > 1e-3, 0.70, 0.0)
  return size, rgba


_FORCE_MAX_N = 2.0 * WHEEL_EFFORT_LIMIT_NM / WHEEL_RADIUS_M


def _max_lean_rad(total_mass: torch.Tensor) -> torch.Tensor:
  """Steepest lean 2 x wheel torque can hold, at this total mass.

  F = 2 * tau_max / r is the whole ground-force budget; holding any lean
  theta costs M g tan(theta), so the robot runs out of authority at
  theta = atan(F / (M g)).
  """
  return torch.atan(_FORCE_MAX_N / (GRAVITY * total_mass.clamp_min(1e-3)))


def _required_lean_rad(
  com_x: torch.Tensor, com_height: torch.Tensor, max_slope_rad: float
) -> torch.Tensor:
  """Lean the robot must be able to sustain, given where its own weight sits.

  Two contributions, added because the worst case is both at once (leaning
  further uphill than the offset alone would):

  - theta_eq = atan(com_x / com_height): the offset payload's OWN static
    equilibrium lean. A CoM sitting ahead of the axle needs the robot resting
    at this angle before gravity even balances through the contact point --
    this is not a disturbance to reject, it is where the robot lives at rest.
    Earlier sweeps (scripts/eval_payload_sweep.py) found this the dominant
    failure axis, well ahead of total mass alone -- a mass-only check misses
    it entirely, which is the bug this function exists to fix.
  - max_slope_rad: the steepest terrain the robot will be asked to stand on.

  com_height is measured from the ground (wheel_radius + the chassis CoM's z
  in the body frame, which is itself measured from the wheel axle -- see
  scripts/inspect_mass.py for the same convention).
  """
  theta_eq = torch.atan2(com_x, com_height.clamp_min(1e-3))
  return theta_eq.abs() + max_slope_rad


def _sample_load(
  n: int, cfg: JavisDomainCfg, level: float, device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Draw n complete chassis configurations.

  Returns (group_masses (n, G), payload_pos (n, 3), wiring_pos (n, 3)).
  """
  moments = mass_model.get_moments()[mass_model.CHASSIS_BODY]
  nominal = mass_model.nominal_mass_vector(mass_model.CHASSIS_BODY, device)
  mcfg, pcfg = cfg.mass, cfg.payload

  masses = torch.zeros(n, len(moments.groups), device=device)
  mesh_cols: list[int] = []

  # 1. Component scatter. This is what moves the center of mass: each group
  #    sits somewhere different inside the chassis.
  for i, group in enumerate(moments.groups):
    if group in ("wiring_misc", "payload"):
      continue
    mesh_cols.append(i)
    rng = lerp_range(
      mcfg.group_scale_easy[group], mcfg.group_scale_hard[group], level
    )
    masses[:, i] = nominal[i] * _uniform((n,), rng, device)

  # 2. Cable harness and oddments: absolute mass at a position of its own.
  wiring_idx = moments.index("wiring_misc")
  masses[:, wiring_idx] = _uniform(
    (n,), lerp_range(mcfg.wiring_kg_easy, mcfg.wiring_kg_hard, level), device
  )
  wiring_pos = torch.stack(
    [
      _uniform((n,), lerp_range(mcfg.wiring_pos_easy[a], mcfg.wiring_pos_hard[a], level), device)
      for a in range(3)
    ],
    dim=-1,
  )

  # 3. Renormalize the CAD-backed groups to hit a target chassis total. Group
  #    scatter alone only spans ~4-7 kg; the requested envelope is 3-15 kg.
  #    Rescaling (rather than drawing the total independently) keeps the
  #    internal proportions plausible at every total.
  target_total = _uniform(
    (n,), lerp_range(mcfg.chassis_total_easy, mcfg.chassis_total_hard, level), device
  )
  mesh_idx = torch.tensor(mesh_cols, device=device, dtype=torch.long)
  mesh_sum = masses[:, mesh_idx].sum(dim=-1)
  mesh_target = (target_total - masses[:, wiring_idx]).clamp_min(0.5)
  masses[:, mesh_idx] *= (mesh_target / mesh_sum.clamp_min(1e-6))[:, None]

  # 4. Payload.
  payload_idx = moments.index("payload")
  masses[:, payload_idx] = _uniform(
    (n,), lerp_range(pcfg.mass_kg_easy, pcfg.mass_kg_hard, level), device
  )
  payload_pos = torch.stack(
    [
      _uniform((n,), lerp_range(pcfg.pos_x_easy, pcfg.pos_x_hard, level), device),
      _uniform((n,), lerp_range(pcfg.pos_y_easy, pcfg.pos_y_hard, level), device),
      _uniform((n,), lerp_range(pcfg.pos_z_easy, pcfg.pos_z_hard, level), device),
    ],
    dim=-1,
  )
  return masses, payload_pos, wiring_pos


##
# Writing to the model.
##


def _write_chassis(env, env_ids: torch.Tensor, state: DrState, cfg: JavisDomainCfg,
                   body_index: torch.Tensor, geom_index: torch.Tensor) -> None:
  """Fuse the stored group masses and push the result into the model."""
  moments = mass_model.get_moments()[mass_model.CHASSIS_BODY]
  masses = state.group_masses[env_ids]

  mass, ipos, iquat, inertia = mass_model.fuse(
    masses,
    moments,
    point_positions={
      "payload": state.payload_pos[env_ids],
      "wiring_misc": state.wiring_pos[env_ids],
    },
    eigh_fn=_EIGH,
  )

  model = env.sim.model
  model.body_mass[env_ids[:, None], body_index] = mass[:, None]
  model.body_ipos[env_ids[:, None], body_index] = ipos[:, None, :]
  model.body_iquat[env_ids[:, None], body_index] = iquat[:, None, :]
  model.body_inertia[env_ids[:, None], body_index] = inertia[:, None, :]

  payload_mass = masses[:, moments.index("payload")]
  size, rgba = _payload_visual(payload_mass, cfg)
  model.geom_pos[env_ids[:, None], geom_index] = state.payload_pos[env_ids][:, None, :]
  model.geom_size[env_ids[:, None], geom_index] = size[:, None, :]
  model.geom_rgba[env_ids[:, None], geom_index] = rgba[:, None, :]


def _body_index(env, body_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  cfg = SceneEntityCfg(asset_cfg.name, body_names=(body_name,))
  cfg.resolve(env.scene)
  return _get_entity_indices(env.scene[asset_cfg.name].indexing, cfg, "body", False)


def _geom_index(env, geom_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  cfg = SceneEntityCfg(asset_cfg.name, geom_names=(geom_name,))
  cfg.resolve(env.scene)
  return _get_entity_indices(env.scene[asset_cfg.name].indexing, cfg, "geom", False)


##
# Event terms.
##


@requires_model_fields(*_LOAD_FIELDS, recompute=RecomputeLevel.set_const)
def randomize_load(
  env,
  env_ids: torch.Tensor | None,
  cfg: JavisDomainCfg,
  difficulty: DifficultyState,
  max_slope_rad: float = 0.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Rebuild each resetting environment's chassis: component masses + payload.

  Use with ``mode="reset"``. Runs the feasibility filter (see
  `sim_config.FeasibilityCfg`) so environments are not handed a configuration
  the motors provably cannot hold up.
  """
  env_ids = _resolve_env_ids(env, env_ids)
  if env_ids.numel() == 0:
    return

  state = get_state(env)
  level = difficulty.level
  n = int(env_ids.numel())

  masses, payload_pos, wiring_pos = _sample_load(n, cfg, level, env.device)

  fcfg = cfg.feasibility
  if fcfg.enabled:
    moments = mass_model.get_moments()[mass_model.CHASSIS_BODY]
    wheel_total = state.wheel_masses[env_ids].sum(dim=-1)
    payload_idx = moments.index("payload")
    margin_rad = math.radians(fcfg.margin_deg)

    def is_bad(m: torch.Tensor, ppos: torch.Tensor, wpos: torch.Tensor) -> torch.Tensor:
      # fuse_com_only, not fuse: this runs up to max_resample_attempts times
      # per reset batch, and the feasibility check only needs the CoM, not
      # the full inertia tensor -- see fuse_com_only's docstring.
      chassis_mass, com = mass_model.fuse_com_only(
        m, moments, point_positions={"payload": ppos, "wiring_misc": wpos}
      )
      com_height = com[..., 2] + WHEEL_RADIUS_M
      required = _required_lean_rad(com[..., 0], com_height, max_slope_rad) + margin_rad
      available = _max_lean_rad(chassis_mass + wheel_total)
      return required > available

    for _ in range(fcfg.max_resample_attempts):
      bad = is_bad(masses, payload_pos, wiring_pos)
      if not bool(bad.any()):
        break
      k = int(bad.sum())
      new_masses, new_payload, new_wiring = _sample_load(k, cfg, level, env.device)
      masses[bad] = new_masses
      payload_pos[bad] = new_payload
      wiring_pos[bad] = new_wiring
    else:
      # Stragglers after the attempt budget: drop the payload entirely rather
      # than loop forever. Zeroing it (not partially trimming) is deliberate
      # -- the failure mode here can be an extreme MOUNT POSITION as much as
      # an extreme mass, and there is no simple closed-form "how much to
      # trim" once the criterion depends on lean angle rather than mass
      # alone. Zero payload reverts to the bare chassis's own near-centred
      # CoM (+1.4 mm fore/aft, scripts/inspect_mass.py), which is feasible at
      # any chassis mass in range with margin to spare. The chassis mass/
      # position itself is left alone -- that part of the sample is kept.
      bad = is_bad(masses, payload_pos, wiring_pos)
      masses[bad, payload_idx] = 0.0

    state.last_infeasible_frac = float(is_bad(masses, payload_pos, wiring_pos).float().mean())

  state.group_masses[env_ids] = masses
  state.payload_pos[env_ids] = payload_pos
  state.wiring_pos[env_ids] = wiring_pos

  _write_chassis(
    env,
    env_ids,
    state,
    cfg,
    _body_index(env, CHASSIS_BODY_NAME, asset_cfg),
    _geom_index(env, PAYLOAD_GEOM, asset_cfg),
  )


@requires_model_fields(*_LOAD_FIELDS, recompute=RecomputeLevel.set_const)
def change_payload_midepisode(
  env,
  env_ids: torch.Tensor | None,
  cfg: JavisDomainCfg,
  difficulty: DifficultyState,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Swap the payload mass mid-episode, keeping the mount position fixed.

  Use with ``mode="interval"``. This is the "picks something up / puts it down
  while driving" case: the policy gets no warning and no explicit signal, only
  the change in how the robot responds, which is precisely what the observation
  history is there to let it detect.

  Only the mass changes, not the mount pose -- a box that teleports around the
  chassis mid-episode is not a thing that happens, and it would make the
  transition unlearnable rather than merely hard.
  """
  pcfg = cfg.payload
  if not pcfg.change_enabled:
    return

  env_ids = _resolve_env_ids(env, env_ids)
  if env_ids.numel() == 0:
    return

  level = difficulty.level
  prob = pcfg.change_prob_easy + level * (pcfg.change_prob_hard - pcfg.change_prob_easy)
  if prob <= 0.0:
    return

  selected = env_ids[torch.rand(env_ids.shape, device=env.device) < prob]
  if selected.numel() == 0:
    return

  state = get_state(env)
  moments = mass_model.get_moments()[mass_model.CHASSIS_BODY]
  payload_idx = moments.index("payload")
  state.group_masses[selected, payload_idx] = _uniform(
    (int(selected.numel()),),
    lerp_range(pcfg.mass_kg_easy, pcfg.mass_kg_hard, level),
    env.device,
  )

  _write_chassis(
    env,
    selected,
    state,
    cfg,
    _body_index(env, CHASSIS_BODY_NAME, asset_cfg),
    _geom_index(env, PAYLOAD_GEOM, asset_cfg),
  )


def set_load_configuration(
  env,
  chassis_total_kg: torch.Tensor,
  payload_kg: torch.Tensor,
  payload_pos: torch.Tensor,
  cfg: JavisDomainCfg,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Force an exact, per-environment load configuration.

  The deterministic counterpart to `randomize_load`, used by
  scripts/eval_payload_sweep.py to walk a grid of builds instead of sampling
  them -- one grid cell per environment, so the whole sweep is a single
  vectorized rollout.

  Component proportions stay at nominal and are scaled together to reach
  `chassis_total_kg`, so every cell is a plausible robot rather than an
  arbitrary mass/CoM pair.

  Not an event term: call it directly after `env.reset()`. Leave
  `randomize_load` in the event config (its `requires_model_fields` is what
  gets these fields expanded per-world) and just overwrite afterwards.

  Args:
    chassis_total_kg: (num_envs,) total chassis mass, payload excluded.
    payload_kg: (num_envs,) payload mass.
    payload_pos: (num_envs, 3) mount position in the chassis frame.
  """
  state = get_state(env)
  moments = mass_model.get_moments()[mass_model.CHASSIS_BODY]
  nominal = mass_model.nominal_mass_vector(mass_model.CHASSIS_BODY, env.device)

  payload_idx = moments.index("payload")
  wiring_idx = moments.index("wiring_misc")
  mesh_idx = torch.tensor(
    [i for i, g in enumerate(moments.groups) if g not in ("payload", "wiring_misc")],
    device=env.device,
    dtype=torch.long,
  )

  wiring = float(nominal[wiring_idx])
  mesh_target = (chassis_total_kg - wiring).clamp_min(0.5)

  masses = nominal.expand(env.num_envs, -1).clone()
  masses[:, mesh_idx] *= (mesh_target / float(nominal[mesh_idx].sum()))[:, None]
  masses[:, wiring_idx] = wiring
  masses[:, payload_idx] = payload_kg

  state.group_masses[:] = masses
  state.payload_pos[:] = payload_pos

  all_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
  _write_chassis(
    env,
    all_ids,
    state,
    cfg,
    _body_index(env, CHASSIS_BODY_NAME, asset_cfg),
    _geom_index(env, PAYLOAD_GEOM, asset_cfg),
  )


@requires_model_fields(
  "body_mass", "body_ipos", "body_inertia", "body_iquat",
  recompute=RecomputeLevel.set_const,
)
def randomize_wheel_masses(
  env,
  env_ids: torch.Tensor | None,
  cfg: JavisDomainCfg,
  difficulty: DifficultyState,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Scale each wheel's mass independently, inertia following along.

  Left and right are drawn separately on purpose. Two hub motors are never
  identical, and unequal wheel inertia is a genuine yaw disturbance the policy
  has to absorb -- averaging it away would make sim easier than reality.

  A wheel is one homogeneous group, so scaling its mass by s scales its inertia
  by s and leaves ipos and iquat alone: no eigendecomposition needed.
  """
  env_ids = _resolve_env_ids(env, env_ids)
  if env_ids.numel() == 0:
    return

  state = get_state(env)
  rng = lerp_range(cfg.mass.wheel_scale_easy, cfg.mass.wheel_scale_hard, difficulty.level)
  nominal = mass_model.nominal_masses("wheel")["wheel"]
  model = env.sim.model

  for slot, body_name in enumerate(WHEEL_BODIES):
    index = _body_index(env, body_name, asset_cfg)
    scale = _uniform((int(env_ids.numel()),), rng, env.device)
    state.wheel_masses[env_ids, slot] = nominal * scale

    default_mass = env.sim.get_default_field("body_mass")[index]
    default_inertia = env.sim.get_default_field("body_inertia")[index]
    model.body_mass[env_ids[:, None], index] = default_mass * scale[:, None]
    model.body_inertia[env_ids[:, None], index] = (
      default_inertia * scale[:, None, None]
    )
