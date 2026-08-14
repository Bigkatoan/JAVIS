"""Difficulty ramp for the payload/balance task.

Training straight at the full envelope does not work. A policy that has never
balanced anything cannot learn to balance 10 kg mounted 0.6 m up on a 10-degree
slope, and with most episodes ending in the first half second there is nothing
in the batch to learn *from*. So the DR ranges start close to the robot's
best-estimate build and open up as the policy copes.

Progress is measured by mean episode length, not reward. Reward mixes tracking
accuracy with survival and drifts as the ranges widen -- so a rising level would
depress it even when things are going well. "How long does it stay up" stays
comparable across every difficulty setting.

The level is a single float shared by reference with every event term (see
`sim_config.DifficultyState`), so changing it here immediately changes what the
next reset samples.
"""

from __future__ import annotations

import torch

from ..sim_config import CurriculumCfg, DifficultyState
from . import events


def payload_difficulty(
  env,
  env_ids: torch.Tensor | None,
  difficulty: DifficultyState,
  cfg: CurriculumCfg,
) -> float:
  """Widen or narrow the DR ranges based on how long episodes are lasting.

  Returns the current level, which mjlab logs as a curriculum metric.
  """
  if not cfg.enabled:
    return difficulty.level

  max_steps = env.max_episode_length
  # Episodes that ended on the time limit are successes, so measuring the
  # resetting batch (rather than a running average over all envs) reacts fast
  # without needing extra state.
  if env_ids is not None and env_ids.numel() > 0:
    lengths = env.episode_length_buf[env_ids].float()
    fraction = float((lengths / max_steps).mean())

    if fraction > cfg.promote_above:
      difficulty.level += cfg.step_up * env_ids.numel()
    elif fraction < cfg.demote_below:
      difficulty.level -= cfg.step_down * env_ids.numel()
    difficulty.clamp()

  return difficulty.level


def infeasible_fraction(env, env_ids: torch.Tensor | None) -> float:
  """Share of the last sampled batch that the feasibility filter could not fix.

  Logged rather than acted on. A non-zero value means the requested mass
  envelope now reaches past what 2 x 3.1 N*m can hold up on the configured
  terrain -- useful to see, because it is the point where further widening buys
  nothing but noise.
  """
  return events.get_state(env).last_infeasible_frac


def current_total_mass(env, env_ids: torch.Tensor | None) -> float:
  """Mean total robot mass across environments, in kg. Purely for logging: it
  makes the curriculum visible as a physical quantity instead of a unitless
  level."""
  return float(events.get_state(env).total_mass().mean())
