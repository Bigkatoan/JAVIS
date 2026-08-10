"""An ODrive velocity loop, modelled properly, as an mjlab action term.

Why not just use mjlab's velocity actuator
------------------------------------------
`BuiltinVelocityActuatorCfg` creates a MuJoCo `<velocity>` actuator, which is
*proportional only*: torque = kv * (v_target - v). The real MKS xDrive Mini runs
ODrive's velocity controller, which is proportional **plus integral**
(`vel_gain = 0.25 N*m/(turn/s)`, `vel_integrator_gain = 0.15`, see
scripts/motor_web_test.py and scripts/tune_wheel_pid.py).

For most robots that difference is a detail. For this task it is the whole
problem. The integral term is exactly the mechanism that absorbs an unknown
load: put 8 kg on the robot and the real board silently winds up more current
until the wheel hits the commanded speed. A P-only sim gives the policy a
drivetrain whose steady-state velocity sags with load, so the policy learns to
compensate for that sag itself -- and then on hardware it compensates on top of
an integrator that has already done the job, and the two fight. That shows up as
low-frequency oscillation under load, which is both the hardest failure mode to
debug on a balancing robot and the most likely one to break something.

So: `BuiltinMotorActuatorCfg` for a plain torque actuator, and the PI loop runs
here, at the physics rate, with the real board's gains and ramp rate.

Set `DrivetrainCfg.use_pi_actuator = False` to fall back to mjlab's builtin
velocity actuator and A/B the difference.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from mjlab.envs.mdp.actions.actions import BaseAction, BaseActionCfg
from mjlab.actuator.actuator import TransmissionType

from ..sim_config import DifficultyState, DrivetrainCfg, lerp_range


@dataclass(kw_only=True)
class OdriveVelocityActionCfg(BaseActionCfg):
  """Wheel velocity targets, executed through a simulated ODrive PI loop."""

  drivetrain: DrivetrainCfg = field(default_factory=DrivetrainCfg)
  difficulty: DifficultyState = field(default_factory=DifficultyState)
  effort_limit: float = 3.105
  """Torque clamp, N*m. Must match the motor actuator's effort_limit."""

  def __post_init__(self):
    self.transmission_type = TransmissionType.JOINT

  def build(self, env) -> OdriveVelocityAction:
    return OdriveVelocityAction(self, env)


class OdriveVelocityAction(BaseAction):
  """Discrete PI velocity controller with anti-windup and a slew-rate limit.

  Per control step (50 Hz) the policy's action becomes a wheel velocity target.
  Per physics step (200 Hz) the loop:

    1. slews the working setpoint toward the target at `vel_ramp_rad_s2`
       (the board's INPUT_MODE_VEL_RAMP),
    2. integrates the velocity error into a torque accumulator,
    3. outputs tau = kp*e + integrator, clamped to +-effort_limit.

  Anti-windup is conditional integration: while the output is saturated, error
  that would push it further into saturation is not accumulated. Without it, a
  robot that is briefly torque-limited (which, at 10 kg of payload, is most of
  the time) builds an integrator it then has to unwind, and the resulting
  overshoot reads as an oscillation the policy cannot see the cause of.

  Per-environment gains and command latency are resampled on reset, so the
  policy also has to tolerate not knowing its own drivetrain exactly.
  """

  cfg: OdriveVelocityActionCfg

  def __init__(self, cfg: OdriveVelocityActionCfg, env):
    super().__init__(cfg=cfg, env=env)

    self._physics_dt = env.cfg.sim.mujoco.timestep
    dcfg = cfg.drivetrain

    shape = (self.num_envs, self.action_dim)
    self._integrator = torch.zeros(shape, device=self.device)
    self._ramped_target = torch.zeros(shape, device=self.device)
    self._kp = torch.full(shape, dcfg.kp_nm_per_rad_s, device=self.device)
    self._ki = torch.full(shape, dcfg.ki_nm_per_rad, device=self.device)

    # Command latency ring buffer, indexed in control steps. Sized for the
    # widest configured lag so the buffer never has to be reallocated.
    self._max_lag = max(dcfg.latency_steps_easy[1], dcfg.latency_steps_hard[1])
    self._cmd_history = torch.zeros(
      (self._max_lag + 1, self.num_envs, self.action_dim), device=self.device
    )
    self._history_head = 0
    self._lag = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self._delayed_target = torch.zeros(shape, device=self.device)
    self._last_torque = torch.zeros(shape, device=self.device)
    self._env_arange = torch.arange(self.num_envs, device=self.device)

    self._resample_drivetrain(
      torch.arange(self.num_envs, device=self.device, dtype=torch.long)
    )

  # Internals.

  def _resample_drivetrain(self, env_ids: torch.Tensor) -> None:
    dcfg = self.cfg.drivetrain
    level = self.cfg.difficulty.level
    n = int(env_ids.numel())
    if n == 0:
      return

    gain_range = lerp_range(dcfg.gain_scale_easy, dcfg.gain_scale_hard, level)
    lo, hi = gain_range
    # One scale per wheel: the two boards are tuned independently and drift
    # apart, so left and right do not share a gain error.
    scale = torch.empty((n, self.action_dim), device=self.device).uniform_(lo, hi)
    self._kp[env_ids] = dcfg.kp_nm_per_rad_s * scale
    self._ki[env_ids] = dcfg.ki_nm_per_rad * scale

    lag_lo = round(
      dcfg.latency_steps_easy[0]
      + level * (dcfg.latency_steps_hard[0] - dcfg.latency_steps_easy[0])
    )
    lag_hi = round(
      dcfg.latency_steps_easy[1]
      + level * (dcfg.latency_steps_hard[1] - dcfg.latency_steps_easy[1])
    )
    lag_hi = max(lag_hi, lag_lo)
    self._lag[env_ids] = torch.randint(
      lag_lo, lag_hi + 1, (n,), device=self.device, dtype=torch.long
    )

  @property
  def measured_velocity(self) -> torch.Tensor:
    return self._entity.data.joint_vel[:, self._target_ids]

  @property
  def applied_torque(self) -> torch.Tensor:
    """Last torque commanded, for the saturation reward term."""
    return self._last_torque

  # ActionTerm interface.

  def process_actions(self, actions: torch.Tensor) -> None:
    super().process_actions(actions)
    # Push this control step's target into the latency buffer and read back the
    # one that is `lag` steps old.
    self._history_head = (self._history_head + 1) % self._cmd_history.shape[0]
    self._cmd_history[self._history_head] = self._processed_actions
    read = (self._history_head - self._lag) % self._cmd_history.shape[0]
    self._delayed_target = self._cmd_history[read, self._env_arange]

  def apply_actions(self) -> None:
    dt = self._physics_dt
    limit = self.cfg.effort_limit

    # 1. Slew the setpoint (INPUT_MODE_VEL_RAMP on the real board).
    max_delta = self.cfg.drivetrain.vel_ramp_rad_s2 * dt
    delta = (self._delayed_target - self._ramped_target).clamp(-max_delta, max_delta)
    self._ramped_target = self._ramped_target + delta

    error = self._ramped_target - self.measured_velocity

    # 2. Conditional integration: only accumulate error that moves the output
    #    away from the rail it is currently against.
    candidate = self._integrator + self._ki * error * dt
    unsaturated = (self._kp * error + candidate).abs() <= limit
    winding_back = (candidate.abs() < self._integrator.abs())
    self._integrator = torch.where(
      unsaturated | winding_back, candidate, self._integrator
    )

    # 3. Output.
    torque = (self._kp * error + self._integrator).clamp(-limit, limit)
    self._last_torque = torque
    self._entity.set_joint_effort_target(torque, joint_ids=self._target_ids)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    super().reset(env_ids)
    if env_ids is None or isinstance(env_ids, slice):
      ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
    else:
      ids = env_ids.to(self.device, dtype=torch.long)

    self._integrator[ids] = 0.0
    # Start the ramp from the wheel's actual speed, not from zero: a reset that
    # leaves the wheels spinning would otherwise command a full-torque brake on
    # the first step.
    self._ramped_target[ids] = self.measured_velocity[ids]
    self._cmd_history[:, ids] = 0.0
    self._resample_drivetrain(ids)
