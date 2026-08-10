"""Shared plumbing for the evaluation scripts.

`scripts/eval_payload_sweep.py`, `scripts/record_payload_video.py` and
`scripts/export_onnx.py` all need the same three things: build the play-mode
env, load a checkpoint into a runner, and get an inference policy out. This
keeps that in one place rather than three.
"""

from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.runner import MjlabOnPolicyRunner

from .balance_task import (
  javis_balance_flat_env_cfg,
  javis_balance_ppo_runner_cfg,
  javis_balance_rough_env_cfg,
)
from .robot_constants import WHEEL_EFFORT_LIMIT_NM, WHEEL_RADIUS_M

GRAVITY = 9.81

TASKS = {
  "flat": (javis_balance_flat_env_cfg, "javis_payload_flat"),
  "rough": (javis_balance_rough_env_cfg, "javis_payload_rough"),
}


def make_env_cfg(
  task: str,
  num_envs: int,
  lin_vel_x: float,
  ang_vel_z: float,
  deterministic_load: bool = True,
  episode_length_s: float = 12.0,
):
  """Play-mode env config pinned to one twist command.

  Fixing the command matters for evaluation: the training command resamples
  every few seconds, which would mix "how well does it track" with "how well
  does it handle a command change". Here every environment is asked for the
  same thing for the whole rollout, so the tracking error means one thing.
  """
  cfg_fn, _ = TASKS[task]
  cfg = cfg_fn(play=True)
  cfg.scene.num_envs = num_envs
  cfg.episode_length_s = episode_length_s

  twist = cfg.commands["twist"]
  twist.ranges.lin_vel_x = (lin_vel_x, lin_vel_x)
  twist.ranges.lin_vel_y = (0.0, 0.0)
  twist.ranges.ang_vel_z = (ang_vel_z, ang_vel_z)
  twist.rel_standing_envs = 0.0
  twist.resampling_time_range = (1e9, 1e9)
  twist.debug_vis = False

  if deterministic_load:
    # Keep `randomize_load` -- its requires_model_fields declaration is what
    # gets body_*/geom_* expanded per world -- but drop the terms that would
    # keep changing the build after we have pinned it. The pinned values are
    # written by events.set_load_configuration() after each reset.
    cfg.events.pop("payload_change", None)
    cfg.events.pop("randomize_wheel_masses", None)
    cfg.events.pop("wheel_friction", None)
    cfg.events.pop("wheel_drag", None)
    cfg.events.pop("push_robot", None)
    # Spawn upright and still: the sweep is measuring the load envelope, not
    # recovery from a random initial tumble.
    cfg.events["reset_base"].params["pose_range"] = {
      "x": (0.0, 0.0),
      "y": (0.0, 0.0),
      "z": (0.0, 0.0),
      "roll": (0.0, 0.0),
      "pitch": (0.0, 0.0),
      "yaw": (0.0, 0.0),
    }
    cfg.events["reset_wheels"].params["velocity_range"] = (0.0, 0.0)
    cfg.curriculum = {}

  return cfg


def load_policy(
  task: str, checkpoint: Path, env_cfg, device: str, render_mode: str | None = None
):
  """Build the env and load `checkpoint` into an inference policy.

  Returns (wrapped_env, policy, runner). The runner is returned as well so
  callers can reuse mjlab's ONNX export. Pass ``render_mode="rgb_array"`` to
  get frames out of ``env.unwrapped.render()``.
  """
  _, experiment = TASKS[task]
  agent_cfg = javis_balance_ppo_runner_cfg(experiment)

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  runner = MjlabOnPolicyRunner(wrapped, asdict(agent_cfg), device=device)
  runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device)
  policy = runner.get_inference_policy(device=device)
  return wrapped, policy, runner


def max_supportable_mass(slope_rad: float = 0.0) -> float:
  """Heaviest robot that can hold a lean at all on the given slope.

  F = 2*tau_max/r is the whole ground-force budget; standing still on a slope
  alpha already spends M g sin(alpha) of it. Drawn as the feasibility boundary
  on the sweep heatmaps so the hardware limit is visible next to the results.
  """
  force_max = 2.0 * WHEEL_EFFORT_LIMIT_NM / WHEEL_RADIUS_M
  denom = math.tan(max(slope_rad, 1e-6)) if slope_rad > 0 else 1.0
  return force_max / (GRAVITY * denom)


def max_lean_deg(total_mass_kg: torch.Tensor | float) -> torch.Tensor | float:
  """Steepest lean the motors can hold at a given total mass, in degrees."""
  force_max = 2.0 * WHEEL_EFFORT_LIMIT_NM / WHEEL_RADIUS_M
  if isinstance(total_mass_kg, torch.Tensor):
    return torch.rad2deg(torch.atan(force_max / (total_mass_kg * GRAVITY)))
  return math.degrees(math.atan(force_max / (total_mass_kg * GRAVITY)))
