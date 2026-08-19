"""A thin, torch-native, GPU-resident vectorized wrapper around the real
mjlab task (`javis/balance_task.py`'s `Javis-Payload-Rough`/`-Flat`), built
for the from-scratch (no SRL, no rsl_rl) PPO/SAC control experiment in
`scripts/scratch_ppo.py` / `scripts/scratch_sac.py`.

Why a FOURTH wrapper around the same task
-------------------------------------------
`javis/gym_env_mjlab.py` (SB3 `VecEnv`, numpy) and `javis/gym_env_mjlab_
single.py` (plain `gymnasium.Env`, `num_envs=1`) both exist for other
consumers. Neither is right for a hand-written CleanRL-style training loop:
SB3's `VecEnv` forces a numpy round-trip on every step (real cost at
thousands of envs/step on GPU) and the single-env wrapper has no batching at
all. This one never leaves torch/CUDA and returns exactly what a manual
rollout loop needs in one `step()` call -- see `JavisMjlabVecEnvTorch.step`'s
docstring for the two-observation split (`next_obs` vs `true_final_obs`) that
makes correct time-limit value bootstrapping possible without a second
wrapper-level reset dance in the caller.

`auto_reset=False`, on purpose (same reasoning as the other two wrappers)
---------------------------------------------------------------------------
mjlab's own default (`auto_reset=True`) resets *inside* `step()` and hands
back the next episode's first observation with no way to recover the true
terminal one. That matters here specifically because SRL's own mjlab
backend (`srl.envs.isaac_lab_wrapper.IsaacLabWrapper`, confirmed by reading
the installed package directly) does NOT set `auto_reset=False` -- it trains
against the post-reset observation on every truncation, i.e. no real
time-limit bootstrapping. This wrapper deliberately does the more correct
thing (`auto_reset=False` + explicit `reset(env_ids=...)` + a value-bootstrap
hook the training scripts use), both because it is the technically correct
way to handle a 20s/2000-step time limit and because it is a live, testable
difference between this implementation and SRL's.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

_REGISTERED_TASK_IDS = ("Javis-Velocity-Flat", "Javis-Payload-Flat", "Javis-Payload-Rough")


class JavisMjlabVecEnvTorch:
  """`num_envs`-batched, GPU-resident view of a real mjlab JAVIS task.

  All returned tensors live on `device` and are NOT copied to CPU/numpy --
  callers that need Python scalars must call `.item()`/`.cpu()` themselves.
  """

  def __init__(
    self,
    task_id: str = "Javis-Payload-Rough",
    num_envs: int = 4096,
    device: str = "cuda:0",
    obs_group: str = "actor",
    play: bool = False,
    seed: int | None = None,
  ) -> None:
    if task_id not in _REGISTERED_TASK_IDS:
      raise ValueError(f"Unknown task_id {task_id!r}; expected one of {_REGISTERED_TASK_IDS}")
    self.task_id = task_id
    self.num_envs = num_envs
    self.device = torch.device(device)
    self.obs_group = obs_group

    cfg = load_env_cfg(task_id, play=play)
    cfg.scene.num_envs = num_envs
    cfg.auto_reset = False
    if seed is not None:
      cfg.seed = seed

    self.env = ManagerBasedRlEnv(cfg, device=device)
    self.episode_length_s: float = float(cfg.episode_length_s)
    self.control_dt: float = float(cfg.decimation) * float(cfg.sim.mujoco.timestep)
    self.max_episode_steps: int = round(self.episode_length_s / self.control_dt)

    self.obs_dim = int(self.env.observation_manager.group_obs_dim[obs_group][0])
    self.action_dim = int(sum(self.env.action_manager.action_term_dim))

    # Running per-env episode accumulators -- lets the training scripts log a
    # `train/score_mean`-style rolling statistic without a second env pass.
    self._ep_return = torch.zeros(num_envs, device=self.device)
    self._ep_len = torch.zeros(num_envs, device=self.device, dtype=torch.long)

    obs, _ = self.env.reset(seed=seed)
    self._obs = obs[obs_group]

  def reset(self, seed: int | None = None) -> torch.Tensor:
    obs, _ = self.env.reset(seed=seed)
    self._obs = obs[self.obs_group]
    self._ep_return.zero_()
    self._ep_len.zero_()
    return self._obs

  def step(
    self, actions: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """One batched control step.

    Returns
    -------
    next_obs : (num_envs, obs_dim)
        What to feed the policy at t+1. For envs that just terminated/timed
        out, this is already the POST-reset (new episode's first)
        observation -- safe to act on immediately, no separate reset call
        needed in the caller.
    true_final_obs : (num_envs, obs_dim)
        The observation mjlab actually returned from `step()`, i.e. the
        real terminal observation for envs that just ended (PRE-reset) and
        identical to `next_obs` for envs that didn't. Feed this (not
        `next_obs`) to the critic when bootstrapping value across a
        time-limit truncation -- using `next_obs` there would bootstrap off
        the wrong episode's first state.
    reward, terminated, truncated : (num_envs,)
    extras : dict with "completed_returns" / "completed_lengths" (1-D
        tensors, one entry per env that ended this step, empty if none).
    """
    actions = actions.clamp(-1.0, 1.0).to(self.device)
    obs, reward, terminated, truncated, _mjlab_extras = self.env.step(actions)
    true_final_obs = obs[self.obs_group]

    self._ep_return += reward
    self._ep_len += 1

    done = terminated | truncated
    extras: dict = {}
    next_obs = true_final_obs
    if bool(done.any()):
      done_ids = done.nonzero(as_tuple=False).squeeze(-1)
      extras["completed_returns"] = self._ep_return[done_ids].clone()
      extras["completed_lengths"] = self._ep_len[done_ids].clone()
      reset_obs, _ = self.env.reset(env_ids=done_ids)
      next_obs = true_final_obs.clone()
      next_obs[done_ids] = reset_obs[self.obs_group][done_ids]
      self._ep_return[done_ids] = 0.0
      self._ep_len[done_ids] = 0

    self._obs = next_obs
    return next_obs, true_final_obs, reward, terminated, truncated, extras

  def close(self) -> None:
    self.env.close()


@dataclass
class EvalResult:
  score_mean: float
  score_max: float
  score_min: float
  episode_length_mean: float
  scores: list[float] = field(default_factory=list)
  wall_time_s: float = 0.0


@torch.no_grad()
def evaluate_policy(
  eval_env: JavisMjlabVecEnvTorch,
  deterministic_action_fn: Callable[[torch.Tensor], torch.Tensor],
  seed: int | None = None,
) -> EvalResult:
  """Run exactly one episode per env in `eval_env` (`num_envs` == episode
  count) to completion, deterministic actions, and report the mean/max/min
  undiscounted episode return -- the same protocol SRL's own
  `srl.cli.train._evaluate_agent` uses against this task (deterministic
  action, full episode, raw summed reward -- confirmed by reading the
  installed `srl/cli/train.py` directly), so numbers here are comparable in
  spirit to the `eval/score_mean` figures already recorded for SRL's PPO/SAC
  runs on this task. Not bit-identical: SRL runs its `episodes` sequentially
  through one env with per-episode seeding; this runs `num_envs` episodes in
  parallel through one already-batched env for speed. Both draw i.i.d.
  episodes from the same training-distribution (domain-randomized, resampled
  commands) config, so the two protocols estimate the same quantity.
  """
  t0 = time.monotonic()
  num_envs = eval_env.num_envs
  obs = eval_env.reset(seed=seed)
  ep_return = torch.zeros(num_envs, device=eval_env.device)
  ep_len = torch.zeros(num_envs, device=eval_env.device, dtype=torch.long)
  done_mask = torch.zeros(num_envs, device=eval_env.device, dtype=torch.bool)

  for _ in range(eval_env.max_episode_steps):
    if bool(done_mask.all()):
      break
    action = deterministic_action_fn(obs)
    next_obs, true_final_obs, reward, terminated, truncated, _extras = eval_env.step(action)
    active = ~done_mask
    ep_return[active] += reward[active]
    ep_len[active] += 1
    done_mask |= terminated | truncated
    obs = next_obs

  scores = ep_return.detach().cpu().tolist()
  lengths = ep_len.detach().float().cpu().tolist()
  return EvalResult(
    score_mean=sum(scores) / len(scores),
    score_max=max(scores),
    score_min=min(scores),
    episode_length_mean=sum(lengths) / len(lengths),
    scores=scores,
    wall_time_s=time.monotonic() - t0,
  )
