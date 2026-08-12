"""stable-baselines3-compatible VecEnv wrapping the REAL mjlab task
(`javis/balance_task.py`'s `Javis-Payload-Flat`/`Javis-Payload-Rough`),
instead of the plain-MuJoCo reimplementation in `javis/gym_env.py`.

Why this exists
----------------
`javis/gym_env.py` reimplements the physics/DR/reward/terrain by hand against
plain `mujoco` -- no GPU batching (one robot per OS process via
SubprocVecEnv), and its offscreen rendering is bare MuJoCo defaults: no
skybox, no reflections, no shadows, which looks noticeably worse than the
mjlab task's own rendering (mjlab's `ViewerConfig`/`OffscreenRenderer`
enables both by default, and the task's terrain generator + built-in
target-vs-actual velocity `debug_vis` were already built and validated in
`javis/mdp/commands.py`/`javis/balance_task.py`).

Rather than reimplement that rendering polish a second time, this wraps the
mjlab task directly. The core problem: mjlab's `ManagerBasedRlEnv` is already
GPU-batched over `num_envs` (torch tensors shaped (num_envs, ...), one CUDA
context), whereas SB3's `VecEnv` API is numpy-based with the same shape
convention but no notion of "the parallelism is already inside a single env
object". So this is a thin numpy<->torch translation layer, not N separate
env processes -- `num_envs` here should be mjlab-sized (hundreds to
thousands), not `javis/gym_env.py`'s SubprocVecEnv-sized (tens).

`auto_reset=False`
-------------------
mjlab's default `auto_reset=True` silently resets terminated/timed-out envs
INSIDE `step()` and returns the post-reset observation with no way to
recover the true terminal one. SB3 needs the true terminal observation (its
`infos[i]["terminal_observation"]`) to bootstrap value estimates correctly
across an episode boundary; silently handing it the next episode's first
observation instead would make every termination look like a mid-episode
transition. `ManagerBasedRlEnvCfg.auto_reset` exists for exactly this case
(see its docstring: "intended for users running their own training loop ...
or a wrapper that handles the reset between steps") -- this IS that wrapper:
`step_wait()` captures the pre-reset observation for any done env, then
calls `env.reset(env_ids=...)` itself before returning.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from mjlab.envs import ManagerBasedRlEnv

from .balance_task import javis_balance_flat_env_cfg, javis_balance_rough_env_cfg


class JavisMjlabVecEnv(VecEnv):
  """SB3 VecEnv over mjlab's Javis-Payload-Flat/Rough, GPU-batched in-process.

  `obs_group`: which mjlab observation group SB3 sees. "actor" (default)
  matches what the real policy/hardware gets; "critic" also exists (adds the
  privileged `load_state` term) but SB3's single-network policies have no
  asymmetric actor-critic slot to put it in, so using it would just leak
  privileged information into the actor -- leave this at "actor" unless you
  have a specific reason not to.
  """

  def __init__(
    self,
    num_envs: int = 1024,
    rough: bool = True,
    difficulty: float = 1.0,
    device: str = "cuda:0",
    render_mode: str | None = None,
    obs_group: str = "actor",
    seed: int | None = None,
    viewer_env_idx: int | None = None,
    viewer_max_extra_envs: int | None = None,
    viewer_width: int | None = None,
    viewer_height: int | None = None,
  ):
    """
    viewer_env_idx / viewer_max_extra_envs / viewer_width / viewer_height:
    only meaningful with render_mode="rgb_array". Override ViewerConfig
    fields (which envs the offscreen renderer shows -- env_idx plus its
    max_extra_envs nearest neighbors -- and the output resolution) so a
    recording env doesn't have to match whatever balance_task.py's default
    happens to be. Must be set here, before construction: the
    OffscreenRenderer is built once in ManagerBasedRlEnv.__init__ from
    cfg.viewer as it stands at that moment, so mutating cfg.viewer on an
    already-constructed env has no effect.
    """
    cfg = javis_balance_rough_env_cfg() if rough else javis_balance_flat_env_cfg()
    cfg.scene.num_envs = num_envs
    cfg.auto_reset = False
    if seed is not None:
      cfg.seed = seed
    if viewer_env_idx is not None:
      cfg.viewer.env_idx = viewer_env_idx
    if viewer_max_extra_envs is not None:
      cfg.viewer.max_extra_envs = viewer_max_extra_envs
    if viewer_width is not None:
      cfg.viewer.width = viewer_width
    if viewer_height is not None:
      cfg.viewer.height = viewer_height

    self.env = ManagerBasedRlEnv(cfg, device=device, render_mode=render_mode)
    self._obs_group = obs_group
    self._device = torch.device(device)

    # difficulty pins the same DifficultyState registry javis/sim_config.py
    # uses -- see balance_task.py's CURRICULUM_ENABLED note; this mirrors
    # play-mode's "pin at 1.0 regardless of where training curriculum sits".
    for difficulty_state in self._difficulty_states():
      difficulty_state.reset(difficulty)

    obs_dim = int(self.env.observation_manager.group_obs_dim[obs_group][0])
    action_dim = int(sum(self.env.action_manager.action_term_dim))
    observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(action_dim,), dtype=np.float32)
    super().__init__(num_envs, observation_space, action_space)

    self._actions: torch.Tensor | None = None
    obs, _ = self.env.reset()
    self._last_obs = self._to_numpy(obs[self._obs_group])

  def _difficulty_states(self):
    """Every DifficultyState object embedded in the built cfg (action terms'
    OdriveVelocityActionCfg.difficulty and _make_env_cfg's own difficulty),
    found by walking the cfg rather than re-deriving the registry key --
    see sim_config.py's module docstring for why the key/registry split
    exists in the first place."""
    seen = []
    for action_cfg in self.env.cfg.actions.values():
      d = getattr(action_cfg, "difficulty", None)
      if d is not None:
        seen.append(d)
    return seen

  # -- numpy/torch conversion --------------------------------------------

  def _to_numpy(self, t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy().astype(np.float32)

  # -- VecEnv interface -----------------------------------------------------

  def reset(self) -> np.ndarray:
    obs, _ = self.env.reset()
    self._last_obs = self._to_numpy(obs[self._obs_group])
    return self._last_obs

  def step_async(self, actions: np.ndarray) -> None:
    self._actions = torch.as_tensor(actions, device=self._device, dtype=torch.float32)

  def step_wait(self):
    assert self._actions is not None, "step_async must be called before step_wait"
    obs, reward, terminated, truncated, _extras = self.env.step(self._actions)
    done_t = terminated | truncated
    done = self._to_numpy(done_t).astype(bool)
    reward_np = self._to_numpy(reward)
    obs_np = self._to_numpy(obs[self._obs_group])

    infos: list[dict] = [{} for _ in range(self.num_envs)]
    done_ids = done.nonzero()[0]
    if len(done_ids) > 0:
      for i in done_ids:
        infos[i]["terminal_observation"] = obs_np[i].copy()
        if bool(truncated[i]) and not bool(terminated[i]):
          infos[i]["TimeLimit.truncated"] = True
      done_ids_t = torch.as_tensor(done_ids, device=self._device, dtype=torch.long)
      reset_obs, _ = self.env.reset(env_ids=done_ids_t)
      reset_obs_np = self._to_numpy(reset_obs[self._obs_group])
      obs_np[done_ids] = reset_obs_np[done_ids]

    self._last_obs = obs_np
    self._actions = None
    return obs_np, reward_np, done, infos

  def close(self) -> None:
    self.env.close()

  def get_attr(self, attr_name: str, indices=None) -> list:
    n = self.num_envs if indices is None else len(self._as_list(indices))
    return [getattr(self.env, attr_name)] * n

  def set_attr(self, attr_name: str, value, indices=None) -> None:
    setattr(self.env, attr_name, value)

  def env_method(self, method_name: str, *args, indices=None, **kwargs) -> list:
    n = self.num_envs if indices is None else len(self._as_list(indices))
    result = getattr(self.env, method_name)(*args, **kwargs)
    return [result] * n

  def env_is_wrapped(self, wrapper_class, indices=None) -> list[bool]:
    n = self.num_envs if indices is None else len(self._as_list(indices))
    return [False] * n

  def seed(self, seed: int | None = None) -> list[int]:
    used = self.env.seed(seed if seed is not None else -1)
    return [used] * self.num_envs

  def render(self, mode: str | None = None):
    return self.env.render()

  @staticmethod
  def _as_list(indices) -> list:
    if indices is None:
      return []
    if isinstance(indices, (list, tuple)):
      return list(indices)
    return [indices]
