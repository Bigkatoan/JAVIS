#!/usr/bin/env python3
"""Evaluate an SRL-trained checkpoint's velocity-TRACKING accuracy, not just
its aggregate reward.

Why this exists
----------------
`eval/score_mean` (SRL's own periodic eval metric, what every plateau/
decline number in this project's training investigation has been judged
by) is a single scalar blending several reward terms together: tracking
accuracy, staying upright, staying alive, and small penalty terms. Direct
instrumentation of a live rollout found that ~71% of a typical episode's
steps have a near-zero commanded velocity (`JavisVelocityCommandCfg`'s
`rel_standing_envs` + however often a resampled command happens to land
near zero) -- so a policy can score well by mostly balancing in place,
without ever being tested hard on the steps that actually matter: holding
a real, sustained, nonzero commanded velocity. Since the deployment
requirement is specifically "moves according to the direction vector,
since that vector IS the control input", this script measures that
directly: mean tracking error, computed ONLY over steps where the
commanded velocity is non-trivial (`--lin-threshold`/`--ang-threshold`),
so a "stand still most of the time" policy cannot hide poor tracking
behind the steps where there was nothing to track.

Ground truth, not the noisy observation
------------------------------------------
Uses the exact quantities `mjlab.tasks.velocity.mdp.track_linear_velocity`/
`track_angular_velocity` (the reward terms `javis/balance_task.py` actually
trains against) read directly from the sim: `env.command_manager.
get_command("twist")` and `asset.data.root_link_lin_vel_b`/
`root_link_ang_vel_b`. This is deliberately NOT the same as what the policy
observes (which goes through `javis/balance_task.py`'s IMU noise + delay
model) -- the point here is to measure how well the underlying policy
tracks the true physical command, not to re-derive what it perceives.

Usage
-----
    .venv/bin/python scripts/eval_velocity_tracking.py \\
        --config configs/srl/javis_mjlab_ppo.yaml \\
        --checkpoint checkpoints/ppo_javis_mjlab_ppo/.../best_*.pt \\
        --episodes 30 --device cuda

Prints a JSON metrics dict to stdout and (if --log-path is given) appends
one JSONL row, in the same {"tag", "value", "step", "time"} shape the rest
of this project's `metrics.jsonl` files use, so it plots/greps identically.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import yaml

from javis.gym_env_mjlab_vec import JavisMjlabVecEnvTorch

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_srl_ppo_agent(config_path: str, checkpoint_path: str, device: str):
    """Reconstruct an SRL PPO agent from its YAML config + a checkpoint.

    Mirrors `srl.cli.train.main()`'s own model/agent construction exactly
    (`ModelBuilder.from_yaml` + `_build_algo_config`-equivalent field
    filtering) so the reconstructed network is bit-for-bit the same
    architecture the checkpoint was actually trained with.
    """
    from srl.algorithms.ppo import PPO
    from srl.cli.train import _build_algo_config  # reuse the real YAML->dataclass coercion
    from srl.core.config import PPOConfig
    from srl.registry.builder import ModelBuilder

    model = ModelBuilder.from_yaml(config_path)
    with open(config_path) as f:
        raw_cfg = yaml.safe_load(f)
    train_cfg = raw_cfg.get("train", {}) or {}
    # `_build_algo_config` (not a plain dict comprehension) matters here:
    # PyYAML's safe_load parses bare scientific notation like `1e-3` (no
    # decimal point) as a STRING, not a float -- a well-known YAML 1.1
    # gotcha -- so `lr`/`desired_kl`/etc. would otherwise reach PPOConfig
    # as strings and crash inside `torch.optim.Adam`. This is the exact
    # coercion `srl.cli.train.main()` itself applies before constructing
    # any algorithm config from a YAML `train:` block.
    agent = PPO(model, config=_build_algo_config(PPOConfig, train_cfg, num_envs=1), device=device)
    agent.load(checkpoint_path)
    agent.model.eval()
    return agent


@torch.no_grad()
def evaluate_tracking(
    agent,
    env: JavisMjlabVecEnvTorch,
    episodes: int,
    lin_threshold: float,
    ang_threshold: float,
    seed: int | None = None,
) -> dict:
    num_envs = env.num_envs
    assert episodes == num_envs, (
        f"this script runs exactly one episode per env slot -- pass "
        f"--episodes {num_envs} (env's num_envs) or rebuild the eval env "
        f"with num_envs={episodes}"
    )

    mjlab_env = env.env  # the underlying mjlab.envs.ManagerBasedRlEnv
    robot = mjlab_env.scene["robot"]

    obs = env.reset(seed=seed)
    ep_return = torch.zeros(num_envs, device=env.device)
    ep_len = torch.zeros(num_envs, device=env.device, dtype=torch.long)
    done_mask = torch.zeros(num_envs, device=env.device, dtype=torch.bool)

    # Per-env running sums, split by whether the command was "moving"
    # (above threshold) or "standing" (near zero) at that step.
    lin_err_sum_moving = torch.zeros(num_envs, device=env.device)
    ang_err_sum_moving = torch.zeros(num_envs, device=env.device)
    n_moving = torch.zeros(num_envs, device=env.device)
    lin_err_sum_standing = torch.zeros(num_envs, device=env.device)
    ang_err_sum_standing = torch.zeros(num_envs, device=env.device)
    n_standing = torch.zeros(num_envs, device=env.device)

    for _ in range(env.max_episode_steps):
        if bool(done_mask.all()):
            break

        # True (ground-truth) command + actual velocity, BEFORE stepping --
        # matches what the reward function reads (command/velocity as of
        # the state the action is taken from).
        command = mjlab_env.command_manager.get_command("twist")  # (N, 3): [vx, vy, wz]
        actual_lin = robot.data.root_link_lin_vel_b  # (N, 3)
        actual_ang = robot.data.root_link_ang_vel_b  # (N, 3)

        cmd_lin_xy = command[:, :2]
        cmd_ang_z = command[:, 2]
        lin_err = torch.linalg.norm(cmd_lin_xy - actual_lin[:, :2], dim=-1)
        ang_err = torch.abs(cmd_ang_z - actual_ang[:, 2])

        moving = (torch.linalg.norm(cmd_lin_xy, dim=-1) > lin_threshold) | (
            torch.abs(cmd_ang_z) > ang_threshold
        )
        active = ~done_mask
        moving_active = moving & active
        standing_active = (~moving) & active

        lin_err_sum_moving += torch.where(moving_active, lin_err, torch.zeros_like(lin_err))
        ang_err_sum_moving += torch.where(moving_active, ang_err, torch.zeros_like(ang_err))
        n_moving += moving_active.float()
        lin_err_sum_standing += torch.where(standing_active, lin_err, torch.zeros_like(lin_err))
        ang_err_sum_standing += torch.where(standing_active, ang_err, torch.zeros_like(ang_err))
        n_standing += standing_active.float()

        action = agent.predict({"actor": obs}, deterministic=True)[0]
        next_obs, true_final_obs, reward, terminated, truncated, extras = env.step(action)

        ep_return[active] += reward[active]
        ep_len[active] += 1
        done_mask |= terminated | truncated
        obs = next_obs

    scores = ep_return.detach().cpu().tolist()
    lengths = ep_len.detach().float().cpu().tolist()

    def _safe_mean(total: torch.Tensor, count: torch.Tensor) -> float:
        # Per-env mean, then averaged across envs that had >=1 qualifying
        # step -- avoids envs with zero "moving" steps (bad luck on command
        # resampling) silently dragging the aggregate toward zero via a 0/0
        # implicitly treated as 0.
        has_any = count > 0
        if not bool(has_any.any()):
            return float("nan")
        per_env_mean = torch.where(has_any, total / count.clamp_min(1.0), torch.zeros_like(total))
        return float(per_env_mean[has_any].mean().item())

    total_steps = float((n_moving + n_standing).sum().item())
    moving_frac = float(n_moving.sum().item()) / total_steps if total_steps > 0 else float("nan")

    return {
        "eval/score_mean": sum(scores) / len(scores),
        "eval/episode_length_mean": sum(lengths) / len(lengths),
        "eval/lin_vel_tracking_error_moving": _safe_mean(lin_err_sum_moving, n_moving),
        "eval/ang_vel_tracking_error_moving": _safe_mean(ang_err_sum_moving, n_moving),
        "eval/lin_vel_tracking_error_standing": _safe_mean(lin_err_sum_standing, n_standing),
        "eval/ang_vel_tracking_error_standing": _safe_mean(ang_err_sum_standing, n_standing),
        "eval/moving_step_fraction": moving_frac,
        "eval/episodes": float(num_envs),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="SRL YAML config the checkpoint was trained with")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--task", default="Javis-Payload-Rough")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument(
        "--lin-threshold", type=float, default=0.1,
        help="m/s -- commanded |lin_vel_xy| above this counts as 'moving'",
    )
    p.add_argument(
        "--ang-threshold", type=float, default=0.1,
        help="rad/s -- commanded |ang_vel_z| above this counts as 'moving'",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--step", type=int, default=None, help="training step this checkpoint is from (for logging)")
    p.add_argument("--log-path", type=Path, default=None, help="append JSONL rows here (metrics.jsonl format)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    agent = load_srl_ppo_agent(args.config, args.checkpoint, str(device))
    env = JavisMjlabVecEnvTorch(
        task_id=args.task, num_envs=args.episodes, device=str(device), seed=args.seed + 54321
    )

    result = evaluate_tracking(
        agent, env, episodes=args.episodes,
        lin_threshold=args.lin_threshold, ang_threshold=args.ang_threshold, seed=args.seed,
    )
    env.close()

    print(json.dumps(result, indent=2))

    if args.log_path is not None:
        args.log_path.parent.mkdir(parents=True, exist_ok=True)
        step = args.step if args.step is not None else 0
        with open(args.log_path, "a") as f:
            for tag, value in result.items():
                f.write(json.dumps({"tag": tag, "value": value, "step": step, "time": time.time()}) + "\n")


if __name__ == "__main__":
    main()
