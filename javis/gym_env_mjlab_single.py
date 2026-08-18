"""The real mjlab tasks (`javis/tasks.py`'s `Javis-Velocity-Flat`/
`Javis-Payload-Flat`/`Javis-Payload-Rough`), exposed as plain, single-env
`gymnasium.Env` ids -- `gym.make("Javis-Payload-Rough-v0")` -- instead of
`javis/gym_env_mjlab.py`'s `JavisMjlabVecEnv` (stable-baselines3's batched
`VecEnv` API) or `javis/gym_env.py`'s `Javis-Balance-v0` (a separate,
simplified plain-MuJoCo reimplementation of the physics, not this task).

Why a THIRD wrapper around the same task
-----------------------------------------
mjlab's `ManagerBasedRlEnv` is GPU-batched over `num_envs` by construction --
there's no "plain single env" mode on the mjlab side, just `num_envs=1`. This
module is the thin, unopinionated shim that makes `num_envs=1` look and
behave like an ordinary Gymnasium env: standard 5-tuple `step()`, no VecEnv
machinery, no SB3 dependency, `gym.make(id)` works after a plain `import
javis` (the registration below runs at import time, same pattern as
`Javis-Balance-v0`). Useful for manual testing, notebooks, teleoperation-style
scripting, or any other tool in the Gymnasium ecosystem that expects a single
`gym.Env` instance rather than a batched training pipeline -- for actual GPU
training at scale, use `configs/srl/javis_mjlab_{ppo,sac}.yaml` (thousands of
envs, one process) or `JavisMjlabVecEnv` (SB3), not this.

Observation space
------------------
mjlab publishes multiple named observation *groups* per step (`"actor"`,
`"critic"` for these tasks -- see `javis/balance_task.py`/`javis/velocity_
task.py`). Rather than arbitrarily picking one, this exposes the real
`gymnasium.spaces.Dict` of every group mjlab returns, keyed by group name,
each a flat `Box`. A policy that only wants what the real hardware would see
should read `obs["actor"]`; `obs["critic"]` carries the privileged terms
(e.g. `Javis-Payload-*`'s `load_state`) an asymmetric actor-critic algorithm
can use and a real deployment cannot.

`auto_reset=False`, on purpose
--------------------------------
Standard Gymnasium single-env semantics: once `step()` returns
`terminated or truncated`, the env does NOT reset itself -- the caller must
call `reset()` before stepping again (exactly how `CartPole-v1` and every
other stock Gymnasium env behaves). mjlab's own default (`auto_reset=True`)
resets *inside* `step()` and hands back the next episode's first observation
with no way to recover the true terminal one -- fine for a self-driving
training loop that never looks at the terminal obs, wrong for a Gymnasium
caller that does (e.g. logging the final state, or any wrapper that computes
something from the terminal transition).
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

_REGISTERED_TASK_IDS = ("Javis-Velocity-Flat", "Javis-Payload-Flat", "Javis-Payload-Rough")


class JavisMjlabGymEnv(gym.Env):
  """One real mjlab task, `num_envs=1`, plain `gymnasium.Env` semantics."""

  metadata = {"render_modes": [None, "rgb_array"]}

  def __init__(
    self,
    task_id: str = "Javis-Payload-Rough",
    device: str = "cuda:0",
    render_mode: str | None = None,
  ) -> None:
    if task_id not in _REGISTERED_TASK_IDS:
      raise ValueError(f"Unknown task_id {task_id!r}; expected one of {_REGISTERED_TASK_IDS}")
    self.task_id = task_id
    self.render_mode = render_mode

    cfg = load_env_cfg(task_id)
    cfg.scene.num_envs = 1
    cfg.auto_reset = False
    self._device = torch.device(device)
    self.env = ManagerBasedRlEnv(cfg, device=device, render_mode=render_mode)

    obs_dims = {
      group: int(dim[0]) for group, dim in self.env.observation_manager.group_obs_dim.items()
    }
    self.observation_space = spaces.Dict(
      {
        group: spaces.Box(-np.inf, np.inf, shape=(dim,), dtype=np.float32)
        for group, dim in obs_dims.items()
      }
    )
    action_dim = int(sum(self.env.action_manager.action_term_dim))
    self.action_space = spaces.Box(-1.0, 1.0, shape=(action_dim,), dtype=np.float32)

  def reset(
    self, *, seed: int | None = None, options: dict[str, Any] | None = None
  ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    super().reset(seed=seed)
    obs, info = self.env.reset(seed=seed)
    return self._squeeze_obs(obs), info

  def step(
    self, action: np.ndarray
  ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
    action_t = torch.as_tensor(action, device=self._device, dtype=torch.float32).reshape(1, -1)
    obs, reward, terminated, truncated, info = self.env.step(action_t)
    return (
      self._squeeze_obs(obs),
      float(reward.item()),
      bool(terminated.item()),
      bool(truncated.item()),
      info,
    )

  def render(self):
    if self.render_mode is None:
      return None
    return self.env.render()

  def close(self) -> None:
    self.env.close()

  def _squeeze_obs(self, obs: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    # (1, D) batched, mjlab's convention even at num_envs=1 -> (D,) plain,
    # matching this class's own `observation_space`.
    return {k: v[0].detach().cpu().numpy().astype(np.float32) for k, v in obs.items()}


# Registered under gymnasium's own registry at import time, same pattern as
# `javis/gym_env.py`'s `Javis-Balance-v0` -- `import javis` (or anything that
# imports this module) is what makes these ids resolvable via `gym.make()`.
for _task_id in _REGISTERED_TASK_IDS:
  gym.register(
    id=f"{_task_id}-v0",
    entry_point=JavisMjlabGymEnv,
    kwargs={"task_id": _task_id},
  )
