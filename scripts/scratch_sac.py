#!/usr/bin/env python3
"""From-scratch, CleanRL-style SAC trained directly against the real mjlab
JAVIS task (`Javis-Payload-Rough` by default) -- NO SRL. Companion control
experiment to `scripts/scratch_ppo.py`.

SRL's own SAC on this task (`configs/srl/javis_mjlab_sac_flashsac*.yaml`)
needed several stability patches (BatchNorm + weight-norm projection,
10x-lower `lr_alpha`, and finally a hard floor on the entropy coefficient,
`min_alpha`) to stop `alpha` collapsing to ~0 and an actor that has stopped
exploring occasionally finding an action sequence that pushes mjlab's
physics integrator into a divergent state -- which `javis/mdp/rewards.py`'s
unclamped `pitch_rate_l2` (squares a huge-but-finite angular velocity) turns
into an astronomical negative reward. Even WITH `min_alpha` floored, that
10M-step SAC run still declined from a 1.78 peak to a noisy 0.3-0.9 plateau,
well below PPO's plateau on the same task.

**Deliberate choice: this script ships with `--min-alpha 0.0` (no floor) by
default.** The point of a from-scratch, algorithm-agnostic SAC here is to
find out whether alpha-collapse + the `pitch_rate_l2` exploit is a property
of *this task's reward shaping* (would reproduce in a completely independent
implementation) or a quirk of SRL's specific architecture/training loop
(would NOT reproduce here). Pre-installing SRL's own fix would destroy the
one experiment that can tell those apart. `--min-alpha` exists as a flag for
a deliberate follow-up run, not as a silent default.

Other real implementation choices
------------------------------------
- **Plain MLP actor/critic, ReLU, no normalization layers** (no BatchNorm,
  no weight-norm projection) -- FlashSAC's stabilization tricks are SRL's
  fix, not part of a "minimal, fully-understood" baseline. Twin Q critics,
  squashed (tanh) Gaussian actor with the standard log-prob correction,
  automatic entropy tuning (`target_entropy = -action_dim`), all textbook
  SAC (Haarnoja et al. 2018/2019) -- nothing exotic.
- **Correct time-limit handling via the replay buffer's done mask**, not a
  reward hack: `javis/gym_env_mjlab_vec.py` runs with `auto_reset=False`
  specifically so each transition can store the TRUE post-action
  observation (`true_final_obs`) as `next_obs`, with `done = terminated`
  (excludes time-limit truncation). A transition that ends only by hitting
  the 20s time limit is therefore bootstrapped normally by the Q-target
  rather than treated as a dead end -- the standard fix for time limits in
  off-policy RL (Pardo et al., "Time Limits in Reinforcement Learning").
- **GPU-resident replay buffer** (preallocated CUDA tensors, circular
  overwrite) -- no CPU/numpy round-trip per transition, matching the
  general "keep everything on-device" throughput lesson already validated
  by SRL's own `use_gpu_buffer: true`, implemented independently here.
- **Large-batch/few-update-call throughput shape** (`batch_size=4096`,
  `gradient_steps=8` per env step across `n_envs=512`, i.e. ~2000 samples
  worth of drawn transitions per new 512 collected) -- an engineering
  throughput choice (this project's own prior SAC investigation found
  small-batch/many-tiny-update SAC badly GPU-underutilized on this task),
  not a stability hack, so it stays on by default unlike `min_alpha`.
- **Gradient-norm clipping (max_grad_norm, generous) and a NaN/Inf loss
  skip-guard** on every update -- not reward clipping. This does not hide
  the reward-explosion phenomenon (raw, unclipped reward is still what gets
  logged and what the Q-target sees), it only stops a single pathological
  transition from corrupting the network into unrecoverable NaN weights,
  which is basic numerical hygiene any real implementation would have, not
  a targeted fix for this task's specific failure mode.

Usage
-----
    .venv/bin/python scripts/scratch_sac.py \\
        --task Javis-Payload-Rough --device cuda --n-envs 512 \\
        --total-steps 10000000 --run-name scratch_sac
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from javis.gym_env_mjlab_vec import JavisMjlabVecEnvTorch, evaluate_policy

REPO_ROOT = Path(__file__).resolve().parent.parent


class Actor(nn.Module):
  def __init__(self, obs_dim: int, act_dim: int, hidden: tuple[int, ...] = (256, 256)):
    super().__init__()
    layers: list[nn.Module] = []
    last = obs_dim
    for h in hidden:
      layers += [nn.Linear(last, h), nn.ReLU()]
      last = h
    self.trunk = nn.Sequential(*layers)
    self.mean_layer = nn.Linear(last, act_dim)
    self.logstd_layer = nn.Linear(last, act_dim)

  def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    h = self.trunk(obs)
    mean = self.mean_layer(h)
    log_std = self.logstd_layer(h).clamp(-5.0, 2.0)
    return mean, log_std

  def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean, log_std = self(obs)
    std = log_std.exp()
    dist = Normal(mean, std)
    x = dist.rsample()
    y = torch.tanh(x)
    log_prob = dist.log_prob(x) - torch.log(1.0 - y.pow(2) + 1e-6)
    log_prob = log_prob.sum(-1)
    return y, log_prob, torch.tanh(mean)


class QNet(nn.Module):
  def __init__(self, obs_dim: int, act_dim: int, hidden: tuple[int, ...] = (256, 256)):
    super().__init__()
    layers: list[nn.Module] = []
    last = obs_dim + act_dim
    for h in hidden:
      layers += [nn.Linear(last, h), nn.ReLU()]
      last = h
    layers.append(nn.Linear(last, 1))
    self.net = nn.Sequential(*layers)

  def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    return self.net(torch.cat([obs, action], dim=-1)).squeeze(-1)


class TwinQ(nn.Module):
  def __init__(self, obs_dim: int, act_dim: int, hidden: tuple[int, ...] = (256, 256)):
    super().__init__()
    self.q1 = QNet(obs_dim, act_dim, hidden)
    self.q2 = QNet(obs_dim, act_dim, hidden)

  def forward(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return self.q1(obs, action), self.q2(obs, action)


class GPUReplayBuffer:
  """Preallocated, circular, GPU-resident replay buffer.

  `done` stores TERMINATION only (not time-limit truncation) -- see this
  file's module docstring. `next_obs` stores the TRUE post-action
  observation (pre-reset), not the policy-facing (post-reset) one.
  """

  def __init__(self, capacity: int, obs_dim: int, act_dim: int, device: torch.device):
    self.capacity = capacity
    self.device = device
    self.obs = torch.zeros(capacity, obs_dim, device=device)
    self.next_obs = torch.zeros(capacity, obs_dim, device=device)
    self.actions = torch.zeros(capacity, act_dim, device=device)
    self.rewards = torch.zeros(capacity, device=device)
    self.dones = torch.zeros(capacity, device=device)
    self.ptr = 0
    self.size = 0

  def add(
    self,
    obs: torch.Tensor,
    action: torch.Tensor,
    reward: torch.Tensor,
    next_obs: torch.Tensor,
    done: torch.Tensor,
  ) -> None:
    n = obs.shape[0]
    idx = (torch.arange(n, device=self.device) + self.ptr) % self.capacity
    self.obs[idx] = obs
    self.next_obs[idx] = next_obs
    self.actions[idx] = action
    self.rewards[idx] = reward
    self.dones[idx] = done.float()
    self.ptr = (self.ptr + n) % self.capacity
    self.size = min(self.size + n, self.capacity)

  def sample(self, batch_size: int):
    idx = torch.randint(0, self.size, (batch_size,), device=self.device)
    return self.obs[idx], self.actions[idx], self.rewards[idx], self.next_obs[idx], self.dones[idx]


class JsonlLogger:
  def __init__(self, path: Path):
    self.path = path
    self.path.parent.mkdir(parents=True, exist_ok=True)
    self._f = open(path, "a")

  def log(self, tag: str, value: float, step: int) -> None:
    self._f.write(json.dumps({"tag": tag, "value": float(value), "step": int(step), "time": time.time()}) + "\n")
    self._f.flush()

  def close(self) -> None:
    self._f.close()


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--task", default="Javis-Payload-Rough")
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--n-envs", type=int, default=512)
  p.add_argument("--total-steps", type=int, default=10_000_000)
  p.add_argument("--buffer-size", type=int, default=1_000_000)
  p.add_argument("--batch-size", type=int, default=4096)
  p.add_argument("--gradient-steps", type=int, default=8, help="grad updates per env.step() call")
  p.add_argument("--start-steps", type=int, default=10_000, help="uniform-random transitions before using the policy")
  p.add_argument("--update-after", type=int, default=5_000, help="transitions collected before training starts")
  p.add_argument("--gamma", type=float, default=0.99)
  p.add_argument("--tau", type=float, default=0.005)
  p.add_argument("--lr-actor", type=float, default=3e-4)
  p.add_argument("--lr-critic", type=float, default=3e-4)
  p.add_argument("--lr-alpha", type=float, default=3e-4)
  p.add_argument("--alpha-init", type=float, default=0.2)
  p.add_argument(
    "--min-alpha", type=float, default=0.0,
    help="floor on the auto-tuned entropy coefficient; 0.0 = no floor (see module docstring)",
  )
  p.add_argument("--max-grad-norm", type=float, default=10.0)
  p.add_argument("--eval-freq", type=int, default=1_000_000)
  p.add_argument("--eval-episodes", type=int, default=30)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--run-name", default="scratch_sac")
  return p.parse_args()


def main() -> None:
  args = parse_args()
  device = torch.device(args.device)
  torch.manual_seed(args.seed)

  run_dir = REPO_ROOT / "runs" / args.run_name
  ckpt_dir = REPO_ROOT / "checkpoints" / args.run_name
  ckpt_dir.mkdir(parents=True, exist_ok=True)
  logger = JsonlLogger(run_dir / "metrics.jsonl")

  env = JavisMjlabVecEnvTorch(task_id=args.task, num_envs=args.n_envs, device=str(device), seed=args.seed)
  eval_env = JavisMjlabVecEnvTorch(
    task_id=args.task, num_envs=args.eval_episodes, device=str(device), seed=args.seed + 12345
  )
  obs_dim, act_dim = env.obs_dim, env.action_dim
  print(f"[scratch_sac] task={args.task} obs_dim={obs_dim} act_dim={act_dim} n_envs={args.n_envs} "
        f"min_alpha={args.min_alpha} max_episode_steps={env.max_episode_steps}")

  actor = Actor(obs_dim, act_dim).to(device)
  critic = TwinQ(obs_dim, act_dim).to(device)
  target_critic = TwinQ(obs_dim, act_dim).to(device)
  target_critic.load_state_dict(critic.state_dict())
  for p_ in target_critic.parameters():
    p_.requires_grad_(False)

  actor_opt = torch.optim.Adam(actor.parameters(), lr=args.lr_actor)
  critic_opt = torch.optim.Adam(critic.parameters(), lr=args.lr_critic)

  target_entropy = -float(act_dim)
  log_alpha = torch.zeros(1, device=device, requires_grad=True)
  with torch.no_grad():
    log_alpha += torch.log(torch.tensor(args.alpha_init, device=device))
  alpha_opt = torch.optim.Adam([log_alpha], lr=args.lr_alpha)

  buffer = GPUReplayBuffer(args.buffer_size, obs_dim, act_dim, device)

  obs = env.reset(seed=args.seed)
  global_step = 0
  best_score: float | None = None
  start_time = time.monotonic()
  recent_returns: list[float] = []
  next_eval_at = args.eval_freq
  nan_skips = 0

  while global_step < args.total_steps:
    if global_step < args.start_steps:
      action = torch.empty(args.n_envs, act_dim, device=device).uniform_(-1.0, 1.0)
    else:
      with torch.no_grad():
        action, _logprob, _mean = actor.sample(obs)

    next_obs, true_final_obs, reward, terminated, truncated, extras = env.step(action)
    buffer.add(obs, action, reward, true_final_obs, terminated)
    obs = next_obs
    global_step += args.n_envs

    if "completed_returns" in extras:
      recent_returns.extend(extras["completed_returns"].detach().cpu().tolist())
      if len(recent_returns) > 2000:
        recent_returns = recent_returns[-2000:]

    last_critic_loss = last_actor_loss = last_alpha_loss = 0.0
    last_alpha = float(log_alpha.exp().item())
    if global_step >= args.update_after and buffer.size >= args.batch_size:
      for _ in range(args.gradient_steps):
        b_obs, b_act, b_rew, b_next_obs, b_done = buffer.sample(args.batch_size)
        alpha = log_alpha.exp().detach()

        with torch.no_grad():
          next_action, next_logprob, _ = actor.sample(b_next_obs)
          tq1, tq2 = target_critic(b_next_obs, next_action)
          min_tq = torch.min(tq1, tq2) - alpha * next_logprob
          target_q = b_rew + args.gamma * (1.0 - b_done) * min_tq

        q1, q2 = critic(b_obs, b_act)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        if torch.isfinite(critic_loss):
          critic_opt.zero_grad()
          critic_loss.backward()
          nn.utils.clip_grad_norm_(critic.parameters(), args.max_grad_norm)
          critic_opt.step()
          last_critic_loss = critic_loss.item()
        else:
          nan_skips += 1

        new_action, logprob, _ = actor.sample(b_obs)
        q1_new, q2_new = critic(b_obs, new_action)
        min_q_new = torch.min(q1_new, q2_new)
        actor_loss = (alpha * logprob - min_q_new).mean()
        if torch.isfinite(actor_loss):
          actor_opt.zero_grad()
          actor_loss.backward()
          nn.utils.clip_grad_norm_(actor.parameters(), args.max_grad_norm)
          actor_opt.step()
          last_actor_loss = actor_loss.item()
        else:
          nan_skips += 1

        alpha_loss = -(log_alpha * (logprob.detach() + target_entropy)).mean()
        if torch.isfinite(alpha_loss):
          alpha_opt.zero_grad()
          alpha_loss.backward()
          alpha_opt.step()
          last_alpha_loss = alpha_loss.item()
        else:
          nan_skips += 1

        if args.min_alpha > 0.0:
          with torch.no_grad():
            log_alpha.clamp_(min=torch.log(torch.tensor(args.min_alpha, device=device)))
        last_alpha = float(log_alpha.exp().item())

        with torch.no_grad():
          for p_online, p_target in zip(critic.parameters(), target_critic.parameters()):
            p_target.mul_(1.0 - args.tau).add_(args.tau * p_online)

    if global_step % (args.n_envs * 200) < args.n_envs:
      elapsed = time.monotonic() - start_time
      fps = global_step / elapsed if elapsed > 0 else 0.0
      logger.log("sac/alpha", last_alpha, global_step)
      logger.log("sac/critic_loss", last_critic_loss, global_step)
      logger.log("sac/actor_loss", last_actor_loss, global_step)
      logger.log("sac/alpha_loss", last_alpha_loss, global_step)
      logger.log("sac/nan_skips", nan_skips, global_step)
      logger.log("train/fps", fps, global_step)
      if recent_returns:
        logger.log("train/score_mean", sum(recent_returns) / len(recent_returns), global_step)
      score_str = f"{sum(recent_returns)/len(recent_returns):.3f}" if recent_returns else "n/a"
      print(f"[step {global_step}/{args.total_steps}] fps={fps:.0f} alpha={last_alpha:.4g} "
            f"critic_loss={last_critic_loss:.4g} actor_loss={last_actor_loss:.4g} "
            f"nan_skips={nan_skips} train/score_mean={score_str}")

    if global_step >= next_eval_at or global_step >= args.total_steps:
      actor.eval()

      def det_action_fn(o: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
          _mean, _log_std = actor(o)
          return torch.tanh(_mean)

      result = evaluate_policy(eval_env, det_action_fn, seed=args.seed + global_step)
      actor.train()
      logger.log("eval/score_mean", result.score_mean, global_step)
      logger.log("eval/score_max", result.score_max, global_step)
      logger.log("eval/score_min", result.score_min, global_step)
      logger.log("eval/episode_length_mean", result.episode_length_mean, global_step)
      print(f"[eval] step={global_step} score_mean={result.score_mean:.4f} "
            f"score_max={result.score_max:.4f} score_min={result.score_min:.4f} "
            f"ep_len_mean={result.episode_length_mean:.1f} ({result.wall_time_s:.1f}s)")

      ckpt = {
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "log_alpha": log_alpha.detach().cpu(),
        "step": global_step,
        "score_mean": result.score_mean,
        "args": vars(args),
      }
      torch.save(ckpt, ckpt_dir / "last.pt")
      if best_score is None or result.score_mean > best_score:
        best_score = result.score_mean
        torch.save(ckpt, ckpt_dir / "best.pt")
        print(f"[eval] new best score_mean={best_score:.4f} -> saved {ckpt_dir/'best.pt'}")

      next_eval_at = global_step + args.eval_freq

  logger.close()
  env.close()
  eval_env.close()
  print(f"[scratch_sac] done. total_steps={global_step} elapsed={time.monotonic()-start_time:.1f}s "
        f"best_score_mean={best_score} nan_skips={nan_skips}")


if __name__ == "__main__":
  main()
