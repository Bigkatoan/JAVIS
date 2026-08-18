#!/usr/bin/env python3
"""From-scratch, CleanRL-style PPO trained directly against the real mjlab
JAVIS task (`Javis-Payload-Rough` by default) -- NO SRL, NO rsl_rl. Written
as a control experiment: SRL's own PPO on this task peaks around
`eval/score_mean` ~3.18 near step 10M, declines through ~17M, then settles
into a flat plateau ~1.10-1.23 for the rest of a 40M-step run (see
`configs/srl/javis_mjlab_ppo.yaml` / `configs/srl/javis_mjlab_ppo_exp_
final.yaml` and `README.md`). This script exists to answer: does a minimal,
fully-understood PPO implementation show the same peak-then-decline shape on
this exact task/reward, or is that specific to SRL's implementation?

Real implementation choices made here, and why
------------------------------------------------
- **Observation normalization** (running mean/std over the "actor" obs
  group, updated online during rollout collection). mjlab's own working
  rsl_rl reference config for this task (`javis_balance_ppo_runner_cfg` in
  `javis/balance_task.py`) uses `obs_normalization=True`; SRL's PPO config
  historically did not. Included here from the start rather than treated as
  optional.
- **Scalar (state-independent) action std**, initialized to 1.0
  (`log_std=0`), not a state-dependent head. Same reference config
  (`std_type="scalar"`, `init_std=1.0`) -- found in this project's own prior
  investigation (`configs/srl/javis_mjlab_ppo.yaml`'s "matched std" note) to
  matter for reaching the reference's peak score.
- **Adaptive KL-controlled learning rate, applied every minibatch** (not
  once per epoch): after each gradient step, compute the approximate KL
  between the pre-update and post-update policy on that minibatch; if it
  exceeds `2*desired_kl`, shrink lr by /1.5 (floored at `min_lr`); if it's
  under `desired_kl/2`, grow lr by *1.5 (capped at `max_lr`). Same schedule
  rsl_rl's `schedule="adaptive"` uses, `max_lr` deliberately capped at 1e-3
  (not rsl_rl's raw 1e-2) -- this project's own prior investigation found
  the 10x-higher ceiling actively worse on this task (LR pinned near the
  ceiling for nearly the whole run).
- **Correct time-limit value bootstrapping.** `javis/gym_env_mjlab_vec.py`
  runs the env with `auto_reset=False` specifically so this script can add
  `gamma * V(true_terminal_obs)` to the reward at every truncation (not
  termination) before running GAE, and treat both terminated AND truncated
  as GAE-chain breaks (the correction already accounts for what continuing
  would have been worth). Confirmed by reading the installed `srl` package
  directly that SRL's own mjlab backend (`srl.envs.isaac_lab_wrapper.
  IsaacLabWrapper`) does NOT set `auto_reset=False` -- it trains against the
  post-reset observation on every timeout with no such correction. Whether
  that gap actually contributes to SRL's instability is an open question
  this script cannot answer by itself, but doing it correctly here is cheap
  and is the textbook-correct thing to do regardless.
- **Unbounded Gaussian policy, action clipped (not squashed) to [-1, 1]** at
  the env boundary -- matches the reference config
  (`GaussianDistribution`/`RslRlVecEnvWrapper(clip_actions=...)`), not a
  tanh-squashed policy.
- **n_envs=4096, n_steps=24** (98304 env-steps/iteration), matching both the
  rsl_rl reference and SRL's PPO config exactly -- no reason found to deviate,
  and it keeps GPU-batched throughput comparable to the already-measured SRL
  run (40M steps in ~1811s wall-clock on this same RTX 3090; this script
  should be in the same ballpark, likely somewhat slower due to no
  torch.compile/fused-kernel optimization).

Usage
-----
    .venv/bin/python scripts/scratch_ppo.py \\
        --task Javis-Payload-Rough --device cuda --n-envs 4096 \\
        --total-steps 40000000 --run-name scratch_ppo
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.distributions import Normal

from javis.gym_env_mjlab_vec import JavisMjlabVecEnvTorch, evaluate_policy

REPO_ROOT = Path(__file__).resolve().parent.parent


class RunningMeanStd:
  """Welford-style online mean/var, used for observation normalization.

  `update()` and `normalize()` are deliberately separate methods (not fused)
  so the training loop can normalize the true-terminal observation used for
  time-limit value bootstrapping WITHOUT double-counting it into the running
  statistics (it's the same physical state as the following reset's first
  frame in every other respect, so counting it once, via the main rollout
  observation, is enough).
  """

  def __init__(self, shape: tuple[int, ...], device: torch.device, epsilon: float = 1e-4):
    self.mean = torch.zeros(shape, device=device)
    self.var = torch.ones(shape, device=device)
    self.count = epsilon

  def update(self, x: torch.Tensor) -> None:
    batch_mean = x.mean(dim=0)
    batch_var = x.var(dim=0, unbiased=False)
    batch_count = x.shape[0]
    delta = batch_mean - self.mean
    tot_count = self.count + batch_count
    self.mean = self.mean + delta * batch_count / tot_count
    m_a = self.var * self.count
    m_b = batch_var * batch_count
    m2 = m_a + m_b + delta.pow(2) * self.count * batch_count / tot_count
    self.var = m2 / tot_count
    self.count = tot_count

  def normalize(self, x: torch.Tensor, clip: float = 10.0) -> torch.Tensor:
    return torch.clamp((x - self.mean) / torch.sqrt(self.var + 1e-8), -clip, clip)


def layer_init(layer: nn.Linear, std: float = math.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
  nn.init.orthogonal_(layer.weight, std)
  nn.init.constant_(layer.bias, bias_const)
  return layer


class ActorCritic(nn.Module):
  def __init__(self, obs_dim: int, act_dim: int, hidden: tuple[int, ...] = (256, 256, 128)):
    super().__init__()

    def mlp(out_dim: int, out_std: float) -> nn.Sequential:
      layers: list[nn.Module] = []
      last = obs_dim
      for h in hidden:
        layers += [layer_init(nn.Linear(last, h)), nn.ELU()]
        last = h
      layers += [layer_init(nn.Linear(last, out_dim), std=out_std)]
      return nn.Sequential(*layers)

    self.actor_mean = mlp(act_dim, 0.01)
    self.critic = mlp(1, 1.0)
    # Scalar (state-independent) log-std, init 0.0 -> std=1.0, matching the
    # rsl_rl reference config's std_type="scalar"/init_std=1.0.
    self.actor_logstd = nn.Parameter(torch.zeros(act_dim))

  def get_value(self, obs: torch.Tensor) -> torch.Tensor:
    return self.critic(obs).squeeze(-1)

  def get_action_and_value(self, obs: torch.Tensor, action: torch.Tensor | None = None):
    mean = self.actor_mean(obs)
    std = self.actor_logstd.exp().expand_as(mean)
    dist = Normal(mean, std)
    if action is None:
      action = dist.sample()
    logprob = dist.log_prob(action).sum(-1)
    entropy = dist.entropy().sum(-1)
    value = self.critic(obs).squeeze(-1)
    return action, logprob, entropy, value


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
  p.add_argument("--n-envs", type=int, default=4096)
  p.add_argument("--n-steps", type=int, default=24, help="rollout length per env per iteration")
  p.add_argument("--total-steps", type=int, default=40_000_000)
  p.add_argument("--num-minibatches", type=int, default=6)
  p.add_argument("--n-epochs", type=int, default=5)
  p.add_argument("--gamma", type=float, default=0.99)
  p.add_argument("--gae-lambda", type=float, default=0.95)
  p.add_argument("--clip-range", type=float, default=0.2)
  p.add_argument("--entropy-coef", type=float, default=0.005)
  p.add_argument("--vf-coef", type=float, default=1.0)
  p.add_argument("--max-grad-norm", type=float, default=1.0)
  p.add_argument("--lr", type=float, default=1e-3, help="initial lr for the adaptive-KL schedule")
  p.add_argument("--min-lr", type=float, default=1e-5)
  p.add_argument("--max-lr", type=float, default=1e-3)
  p.add_argument("--desired-kl", type=float, default=0.01)
  p.add_argument("--eval-freq", type=int, default=1_000_000, help="env steps between eval passes")
  p.add_argument("--eval-episodes", type=int, default=30)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--run-name", default="scratch_ppo")
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
  print(f"[scratch_ppo] task={args.task} obs_dim={obs_dim} act_dim={act_dim} n_envs={args.n_envs} "
        f"max_episode_steps={env.max_episode_steps}")

  agent = ActorCritic(obs_dim, act_dim).to(device)
  optimizer = torch.optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)
  obs_rms = RunningMeanStd((obs_dim,), device)

  n_envs, n_steps = args.n_envs, args.n_steps
  batch_size = n_envs * n_steps
  minibatch_size = batch_size // args.num_minibatches
  num_iterations = args.total_steps // batch_size

  obs_buf = torch.zeros((n_steps, n_envs, obs_dim), device=device)
  actions_buf = torch.zeros((n_steps, n_envs, act_dim), device=device)
  logprobs_buf = torch.zeros((n_steps, n_envs), device=device)
  rewards_buf = torch.zeros((n_steps, n_envs), device=device)
  dones_buf = torch.zeros((n_steps, n_envs), device=device)
  values_buf = torch.zeros((n_steps, n_envs), device=device)

  next_obs_raw = env.reset(seed=args.seed)
  global_step = 0
  lr = args.lr
  best_score: float | None = None
  start_time = time.monotonic()

  # Rolling window of recently-completed training episodes, for a cheap
  # `train/score_mean` signal between the authoritative periodic evals.
  recent_returns: list[float] = []

  next_eval_at = args.eval_freq

  for iteration in range(1, num_iterations + 1):
    for t in range(n_steps):
      obs_rms.update(next_obs_raw)
      norm_obs = obs_rms.normalize(next_obs_raw)
      obs_buf[t] = norm_obs
      with torch.no_grad():
        action, logprob, _entropy, value = agent.get_action_and_value(norm_obs)
      values_buf[t] = value
      actions_buf[t] = action
      logprobs_buf[t] = logprob

      next_obs_raw, true_final_raw, reward, terminated, truncated, extras = env.step(action)
      timeout_mask = (truncated & ~terminated).float()
      with torch.no_grad():
        bootstrap_value = agent.get_value(obs_rms.normalize(true_final_raw))
      rewards_buf[t] = reward + args.gamma * bootstrap_value * timeout_mask
      dones_buf[t] = (terminated | truncated).float()

      if "completed_returns" in extras:
        recent_returns.extend(extras["completed_returns"].detach().cpu().tolist())
        if len(recent_returns) > 2000:
          recent_returns = recent_returns[-2000:]

      global_step += n_envs

    with torch.no_grad():
      next_value = agent.get_value(obs_rms.normalize(next_obs_raw))

    advantages = torch.zeros_like(rewards_buf)
    lastgaelam = torch.zeros(n_envs, device=device)
    for t in reversed(range(n_steps)):
      nextnonterminal = 1.0 - dones_buf[t]
      nv = next_value if t == n_steps - 1 else values_buf[t + 1]
      delta = rewards_buf[t] + args.gamma * nv * nextnonterminal - values_buf[t]
      lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
      advantages[t] = lastgaelam
    returns = advantages + values_buf

    b_obs = obs_buf.reshape(-1, obs_dim)
    b_actions = actions_buf.reshape(-1, act_dim)
    b_logprobs = logprobs_buf.reshape(-1)
    b_advantages = advantages.reshape(-1)
    b_returns = returns.reshape(-1)
    b_values = values_buf.reshape(-1)

    last_approx_kl = 0.0
    last_pg_loss = last_v_loss = last_entropy = 0.0
    for _epoch in range(args.n_epochs):
      perm = torch.randperm(batch_size, device=device)
      for start in range(0, batch_size, minibatch_size):
        mb_inds = perm[start:start + minibatch_size]
        _new_action, newlogprob, entropy, newvalue = agent.get_action_and_value(
          b_obs[mb_inds], b_actions[mb_inds]
        )
        logratio = newlogprob - b_logprobs[mb_inds]
        ratio = logratio.exp()
        with torch.no_grad():
          approx_kl = ((ratio - 1) - logratio).mean()

        mb_adv = b_advantages[mb_inds]
        mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
        pg_loss1 = -mb_adv * ratio
        pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - args.clip_range, 1 + args.clip_range)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
        v_clipped = b_values[mb_inds] + torch.clamp(
          newvalue - b_values[mb_inds], -args.clip_range, args.clip_range
        )
        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
        v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

        entropy_loss = entropy.mean()
        loss = pg_loss - args.entropy_coef * entropy_loss + args.vf_coef * v_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
        optimizer.step()

        kl = approx_kl.item()
        if kl > args.desired_kl * 2.0:
          lr = max(args.min_lr, lr / 1.5)
        elif 0.0 < kl < args.desired_kl / 2.0:
          lr = min(args.max_lr, lr * 1.5)
        for g in optimizer.param_groups:
          g["lr"] = lr

        last_approx_kl, last_pg_loss = kl, pg_loss.item()
        last_v_loss, last_entropy = v_loss.item(), entropy_loss.item()

    elapsed = time.monotonic() - start_time
    fps = global_step / elapsed if elapsed > 0 else 0.0
    logger.log("ppo/lr", lr, global_step)
    logger.log("ppo/approx_kl", last_approx_kl, global_step)
    logger.log("ppo/policy_loss", last_pg_loss, global_step)
    logger.log("ppo/value_loss", last_v_loss, global_step)
    logger.log("ppo/entropy", last_entropy, global_step)
    logger.log("train/fps", fps, global_step)
    if recent_returns:
      logger.log("train/score_mean", sum(recent_returns) / len(recent_returns), global_step)

    if iteration % 10 == 0 or iteration == num_iterations:
      score_str = f"{sum(recent_returns)/len(recent_returns):.3f}" if recent_returns else "n/a"
      print(f"[iter {iteration}/{num_iterations}] step={global_step} fps={fps:.0f} "
            f"lr={lr:.2e} kl={last_approx_kl:.4f} train/score_mean={score_str}")

    if global_step >= next_eval_at or iteration == num_iterations:
      agent.eval()

      def det_action_fn(o: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
          return agent.actor_mean(obs_rms.normalize(o))

      result = evaluate_policy(eval_env, det_action_fn, seed=args.seed + iteration)
      agent.train()
      logger.log("eval/score_mean", result.score_mean, global_step)
      logger.log("eval/score_max", result.score_max, global_step)
      logger.log("eval/score_min", result.score_min, global_step)
      logger.log("eval/episode_length_mean", result.episode_length_mean, global_step)
      print(f"[eval] step={global_step} score_mean={result.score_mean:.4f} "
            f"score_max={result.score_max:.4f} score_min={result.score_min:.4f} "
            f"ep_len_mean={result.episode_length_mean:.1f} ({result.wall_time_s:.1f}s)")

      ckpt = {
        "agent": agent.state_dict(),
        "obs_rms_mean": obs_rms.mean,
        "obs_rms_var": obs_rms.var,
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
  print(f"[scratch_ppo] done. total_steps={global_step} elapsed={time.monotonic()-start_time:.1f}s "
        f"best_score_mean={best_score}")


if __name__ == "__main__":
  main()
