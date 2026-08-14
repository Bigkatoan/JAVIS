"""Reward and termination terms for balancing an unknown, off-center load.

The problem with rewarding "upright"
------------------------------------
`javis/velocity_task.py` weights mjlab's `upright` reward at 3.0 with
std=sqrt(0.15), because with a centered CoM the balance point IS vertical.

That stops being true the moment the load is off-center. A robot whose CoM sits
0.12 m ahead of the axle, with the CoM 0.29 m above the ground, has its static
equilibrium at atan(0.12 / 0.29) = 22.5 degrees of lean, not 0. Rewarding
verticality there penalizes the only posture that keeps it standing, caps the
reachable return, and pushes the policy toward accelerating continuously just to
hold itself up -- which wastes exactly the torque headroom a heavy build does
not have.

So this module reframes what "balanced" means:

  - keep mjlab's `upright` but weak and wide (weight 1.0, std sqrt(0.5)), as a
    mild preference rather than an objective;
  - reward *stillness* instead of *verticality* via `pitch_rate_l2` -- a robot
    holding a steady 20-degree lean is balanced, one oscillating through
    vertical is not;
  - penalize torque saturation, the measurable signature of a configuration
    sitting at the edge of what the motors can hold;
  - and let termination, not reward shaping, define failure.
"""

from __future__ import annotations

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor.contact_sensor import ContactSensor

_ROBOT = SceneEntityCfg("robot")


def pitch_rate_l2(env, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
  """Squared body-frame pitch rate.

  The wheels can only generate pitch torque, so pitch is the axis this
  controller actually owns; roll and yaw belong elsewhere. Penalizing the rate
  (not the angle) rewards holding *a* lean rather than swinging through
  vertical, which is what balancing an off-center load actually looks like.
  """
  asset = env.scene[asset_cfg.name]
  return torch.square(asset.data.root_link_ang_vel_b[:, 1])


def torque_saturation(env, action_name: str = "wheel_vel") -> torch.Tensor:
  """Mean squared fraction of the wheel torque budget in use.

  Reads the PI controller's real output rather than the policy's action, so it
  measures what the hardware would actually be asked for. Near 1.0 there is no
  authority left to reject a disturbance -- and at the top of the mass envelope
  there are configurations that would otherwise sit there permanently.

  Returns zeros when running on mjlab's builtin velocity actuator, which
  exposes no torque to read.
  """
  term = env.action_manager.get_term(action_name)
  limit = getattr(term.cfg, "effort_limit", None)
  if limit is None or not hasattr(term, "applied_torque"):
    return torch.zeros(env.num_envs, device=env.device)
  return torch.square(term.applied_torque / limit).mean(dim=-1)


def wheel_slip(env, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
  """Squared mismatch between wheel surface speed and actual forward speed.

  A heavy, high-CoM robot unloads a wheel when it leans hard and the wheel
  spins without moving anything. Penalizing that stops the policy from
  "solving" difficult configurations by shredding traction -- and it matters
  beyond sim, because on hardware the encoder would report motion the robot is
  not making, which the state estimator downstream would believe.
  """
  from ..robot_constants import WHEEL_RADIUS_M

  asset = env.scene[asset_cfg.name]
  surface_speed = asset.data.joint_vel.mean(dim=-1) * WHEEL_RADIUS_M
  forward_speed = asset.data.root_link_lin_vel_b[:, 0]
  return torch.square(surface_speed - forward_speed)


def chassis_grounded(env, sensor_name: str = "chassis_contact") -> torch.Tensor:
  """True once the chassis hull itself touches the ground.

  A more honest failure test than a fixed tilt threshold. With the CoM offsets
  this task allows, a large lean can be a legitimate equilibrium, whereas the
  body resting on the floor never is. Paired with a generous `bad_orientation`
  limit so a robot that is unambiguously gone still terminates promptly instead
  of sliding along on its face for the rest of the episode.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  found = sensor.data.found
  assert found is not None, (
    f"contact sensor '{sensor_name}' must request the 'found' field"
  )
  return found.sum(dim=-1) > 0
