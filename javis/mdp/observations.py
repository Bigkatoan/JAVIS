"""Observation terms for the payload/balance task.

The actor gets nothing about its own mass -- that is the point of the task. It
sees the same IMU, wheel velocities, last action and command that the real
robot will publish, stacked over a short history so the load can be *inferred*
from how the robot responds rather than read off a wire.

The critic, which only exists during training, gets told the answer. This is
asymmetric actor-critic: the value function's job is to predict return, and
return depends heavily on how heavy and how lopsided this particular robot is.
Hiding that from the critic just makes value estimation noisy and slows
learning, with no benefit -- the critic is discarded at deployment. It costs one
observation group and no extra training phase, unlike teacher-student
distillation.
"""

from __future__ import annotations

import torch

from ..robot_constants import WHEEL_RADIUS_M
from . import events


def privileged_load_state(env) -> torch.Tensor:
  """What the robot is actually built out of this episode.

  Returns, per environment:
    [0]    total mass (kg), scaled to O(1)
    [1:4]  chassis center of mass in the chassis frame (m)
    [4]    payload mass (kg), scaled
    [5:8]  payload mount position (m)
    [8:10] left/right wheel mass (kg), scaled
  """
  from .. import mass_model

  state = events.get_state(env)
  moments = mass_model.get_moments()[mass_model.CHASSIS_BODY]

  masses = state.group_masses
  unit_com = torch.as_tensor(
    moments.unit_com, device=masses.device, dtype=masses.dtype
  ).expand(masses.shape[0], -1, -1).clone()
  unit_com[:, moments.index("payload")] = state.payload_pos
  unit_com[:, moments.index("wiring_misc")] = state.wiring_pos

  chassis_mass = masses.sum(dim=-1)
  com = (masses[..., None] * unit_com).sum(dim=-2) / chassis_mass[:, None].clamp_min(1e-6)
  total = chassis_mass + state.wheel_masses.sum(dim=-1)

  # /10 keeps every channel roughly in [-1, 1] without a running normalizer,
  # which matters because these are only ever seen by the critic and so never
  # pass through the actor's obs normalization.
  return torch.cat(
    [
      (total / 10.0)[:, None],
      com,
      (masses[:, moments.index("payload")] / 10.0)[:, None],
      state.payload_pos,
      state.wheel_masses / 10.0,
    ],
    dim=-1,
  )


def wheel_surface_velocity(env, asset_cfg=None) -> torch.Tensor:
  """Wheel speeds expressed as ground speed (m/s) rather than rad/s.

  Same information as `joint_vel_rel`, but in the same units and rough
  magnitude as the linear velocity command, so the policy does not have to
  learn the wheel-radius conversion from scratch. The real driver reads
  `vel_estimate` in turn/s and can apply the identical conversion.
  """
  from mjlab.managers.scene_entity_config import SceneEntityCfg

  cfg = asset_cfg or SceneEntityCfg("robot")
  asset = env.scene[cfg.name]
  return asset.data.joint_vel * WHEEL_RADIUS_M
