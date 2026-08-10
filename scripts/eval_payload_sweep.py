#!/usr/bin/env python3
"""Sweep a policy across a grid of mass / center-of-mass configurations.

This is the quantitative answer to "does it still balance and still follow the
commanded vector when the robot weighs something different?". Unlike training,
nothing here is random: every grid cell is one exact build, run for a fixed
horizon under one fixed twist command, and scored.

One environment per (cell x seed), so the whole sweep is a single vectorized
rollout on the GPU.

Reported per cell:
  survival      fraction of seeds still upright at the end of the horizon
  t_fall        mean seconds until failure (censored at the horizon)
  err_vx        RMS linear velocity tracking error, m/s
  err_wz        RMS angular velocity tracking error, rad/s
  lean          mean |pitch|, deg
  sat           mean fraction of the wheel torque budget in use

The heatmaps also carry the analytic feasibility boundary
`max lean = atan(2*tau/(r*M*g))`, so results can be read against what the
hardware can do rather than in the abstract.

    .venv/bin/python scripts/eval_payload_sweep.py \
        --checkpoint logs/rsl_rl/javis_payload_flat/<run>/model_300.pt
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from javis import eval_utils
from javis.mdp import events as javis_events
from javis.sim_config import JavisDomainCfg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--task", choices=("flat", "rough"), default="flat")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seeds", type=int, default=16,
                   help="environments per grid cell")
    p.add_argument("--horizon-s", type=float, default=12.0)
    p.add_argument("--lin-vel-x", type=float, default=0.3,
                   help="commanded forward velocity, m/s")
    p.add_argument("--ang-vel-z", type=float, default=0.0,
                   help="commanded yaw rate, rad/s")
    p.add_argument("--chassis-kg", type=float, nargs="+",
                   default=[3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0])
    p.add_argument("--payload-kg", type=float, nargs="+",
                   default=[0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
    p.add_argument("--payload-x", type=float, nargs="+", default=[0.0],
                   help="payload fore/aft offset, m")
    p.add_argument("--payload-z", type=float, nargs="+", default=[0.30],
                   help="payload mount height above the axle, m")
    p.add_argument("--vel-gain", type=float, default=None,
                   help="override the ODrive vel_gain the policy runs against, "
                        "to check whether it still works at whatever gain the "
                        "real board turns out to reach")
    p.add_argument("--vel-integrator-gain", type=float, default=None)
    p.add_argument("--out-dir", type=Path, default=Path("logs/eval"))
    p.add_argument("--tag", default="",
                   help="suffix for the output filenames, so sweeps over "
                        "different axes of the same checkpoint don't overwrite")
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args()


def build_grid(args) -> tuple[np.ndarray, list[str]]:
    """Cartesian product of the swept axes, one row per cell."""
    axes = [args.chassis_kg, args.payload_kg, args.payload_x, args.payload_z]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 4)
    return grid, ["chassis_kg", "payload_kg", "payload_x", "payload_z"]


@torch.inference_mode()
def run_sweep(args, grid: np.ndarray):
    n_cells = grid.shape[0]
    num_envs = n_cells * args.seeds

    env_cfg = eval_utils.make_env_cfg(
        task=args.task,
        num_envs=num_envs,
        lin_vel_x=args.lin_vel_x,
        ang_vel_z=args.ang_vel_z,
        episode_length_s=args.horizon_s,
        vel_gain=args.vel_gain,
        vel_integrator_gain=args.vel_integrator_gain,
    )
    env, policy, _ = eval_utils.load_policy(
        args.task, args.checkpoint, env_cfg, args.device
    )
    base = env.unwrapped
    device = base.device

    # Cell index for each environment: [cell0 x seeds, cell1 x seeds, ...].
    cell_of_env = torch.arange(num_envs, device=device) // args.seeds
    grid_t = torch.tensor(grid, dtype=torch.float32, device=device)
    per_env = grid_t[cell_of_env]

    chassis_kg = per_env[:, 0]
    payload_kg = per_env[:, 1]
    payload_pos = torch.stack(
        [per_env[:, 2], torch.zeros_like(per_env[:, 2]), per_env[:, 3]], dim=-1
    )

    domain = JavisDomainCfg()
    env.reset()
    javis_events.set_load_configuration(
        base, chassis_kg, payload_kg, payload_pos, domain
    )
    obs = env.get_observations()

    steps = int(round(args.horizon_s / base.step_dt))
    alive = torch.ones(num_envs, dtype=torch.bool, device=device)
    steps_alive = torch.zeros(num_envs, device=device)
    sum_err_vx = torch.zeros(num_envs, device=device)
    sum_err_wz = torch.zeros(num_envs, device=device)
    sum_lean = torch.zeros(num_envs, device=device)
    sum_sat = torch.zeros(num_envs, device=device)

    robot = base.scene["robot"]
    action_term = base.action_manager.get_term("wheel_vel")
    effort_limit = getattr(action_term.cfg, "effort_limit", None)

    print(f"[sweep] {n_cells} cells x {args.seeds} seeds = {num_envs} envs, "
          f"{steps} steps @ {1.0 / base.step_dt:.0f} Hz")

    for step in range(steps):
        actions = policy(obs)
        obs, _, dones, extras = env.step(actions)

        cmd = base.command_manager.get_command("twist")
        lin = robot.data.root_link_lin_vel_b
        ang = robot.data.root_link_ang_vel_b
        # projected_gravity_b[2] is -1 when perfectly upright; acos gives tilt.
        tilt = torch.rad2deg(
            torch.acos((-robot.data.projected_gravity_b[:, 2]).clamp(-1.0, 1.0))
        )

        sum_err_vx += alive * (cmd[:, 0] - lin[:, 0]) ** 2
        sum_err_wz += alive * (cmd[:, 2] - ang[:, 2]) ** 2
        sum_lean += alive * tilt
        if effort_limit is not None and hasattr(action_term, "applied_torque"):
            sat = (action_term.applied_torque.abs() / effort_limit).mean(dim=-1)
            sum_sat += alive * sat
        steps_alive += alive.float()

        # A `done` that is not a time-out is a fall. Envs auto-reset behind us
        # (and get re-randomized), so freeze each one's score at its first
        # failure rather than scoring whatever robot replaced it.
        timed_out = extras.get("time_outs")
        fell = dones.bool()
        if timed_out is not None:
            fell = fell & ~timed_out.bool()
        alive &= ~fell

    denom = steps_alive.clamp_min(1.0)
    per_env_metrics = {
        "survival": alive.float(),
        "t_fall": torch.where(
            alive, torch.full_like(steps_alive, float(steps)), steps_alive
        ) * base.step_dt,
        "err_vx": (sum_err_vx / denom).sqrt(),
        "err_wz": (sum_err_wz / denom).sqrt(),
        "lean": sum_lean / denom,
        "sat": sum_sat / denom,
    }

    rows = []
    for cell in range(n_cells):
        sel = cell_of_env == cell
        row = {
            "chassis_kg": float(grid[cell, 0]),
            "payload_kg": float(grid[cell, 1]),
            "payload_x": float(grid[cell, 2]),
            "payload_z": float(grid[cell, 3]),
        }
        wheels = 2 * 2.936
        row["total_kg"] = row["chassis_kg"] + row["payload_kg"] + wheels
        row["max_lean_deg"] = eval_utils.max_lean_deg(row["total_kg"])
        for name, values in per_env_metrics.items():
            row[name] = float(values[sel].mean())
        rows.append(row)

    env.close()
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[sweep] wrote {path}")


def plot_heatmaps(rows: list[dict], args, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Plot whichever two axes were actually swept. Averaging over a varied
    # axis would hide exactly the structure the sweep was run to find -- a
    # 100% and a 0% cell average to a meaningless 50%.
    axis_keys = ["chassis_kg", "payload_kg", "payload_x", "payload_z"]
    varied = [k for k in axis_keys if len({r[k] for r in rows}) > 1]
    if len(varied) < 2:
        varied = (varied + [k for k in axis_keys if k not in varied])[:2]
    y_key, x_key = varied[0], varied[1]
    if len(varied) > 2:
        print(f"[sweep] note: {len(varied)} axes varied; plotting {y_key} vs "
              f"{x_key} and averaging over {varied[2:]}")

    x_vals = sorted({r[x_key] for r in rows})
    y_vals = sorted({r[y_key] for r in rows})

    panels = [
        ("survival", "survival rate", "viridis", (0.0, 1.0)),
        ("err_vx", "v_x RMSE (m/s)", "magma_r", None),
        ("err_wz", "yaw rate RMSE (rad/s)", "magma_r", None),
        ("lean", "mean tilt (deg)", "cividis", None),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.6))

    labels = {
        "chassis_kg": "chassis mass (kg)",
        "payload_kg": "payload (kg)",
        "payload_x": "payload x offset (m)",
        "payload_z": "payload height (m)",
    }

    for ax, (key, title, cmap, clim) in zip(axes, panels):
        img = np.full((len(y_vals), len(x_vals)), np.nan)
        for i, yv in enumerate(y_vals):
            for j, xv in enumerate(x_vals):
                vals = [r[key] for r in rows if r[x_key] == xv and r[y_key] == yv]
                if vals:
                    img[i, j] = float(np.mean(vals))

        kwargs = {"vmin": clim[0], "vmax": clim[1]} if clim else {}
        mesh = ax.imshow(img, origin="lower", aspect="auto", cmap=cmap, **kwargs)
        ax.set_xticks(range(len(x_vals)), [f"{v:g}" for v in x_vals])
        ax.set_yticks(range(len(y_vals)), [f"{v:g}" for v in y_vals])
        ax.set_xlabel(labels[x_key])
        ax.set_ylabel(labels[y_key])
        ax.set_title(title)
        fig.colorbar(mesh, ax=ax, fraction=0.046)

        for i in range(len(y_vals)):
            for j in range(len(x_vals)):
                if not np.isnan(img[i, j]):
                    ax.text(j, i, f"{img[i, j]:.2f}", ha="center", va="center",
                            fontsize=7, color="w")

        # Hardware feasibility boundary: where the motors can no longer hold
        # even a 10 degree lean. Anything past it is physics, not policy.
        # Only meaningful when both plotted axes change the total mass.
        if {x_key, y_key} == {"chassis_kg", "payload_kg"}:
            wheels = 2 * 2.936
            boundary = []
            for xv in x_vals:
                crossing = np.nan
                for i, yv in enumerate(y_vals):
                    if eval_utils.max_lean_deg(xv + yv + wheels) < 10.0:
                        crossing = i - 0.5
                        break
                boundary.append(crossing)
            if not all(np.isnan(b) for b in boundary):
                ax.plot(range(len(x_vals)), boundary, "r--", lw=1.6,
                        label="10 deg lean limit")
                ax.legend(loc="upper right", fontsize=7)

    cmd = f"v_x = {args.lin_vel_x} m/s, yaw = {args.ang_vel_z} rad/s"
    fig.suptitle(f"JAVIS payload sweep -- {args.task} terrain, {cmd}, "
                 f"{args.seeds} seeds x {args.horizon_s:g}s")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    print(f"[sweep] wrote {path}")


def print_table(rows: list[dict]) -> None:
    keys = ["chassis_kg", "payload_kg", "payload_x", "payload_z", "total_kg",
            "max_lean_deg", "survival", "t_fall", "err_vx", "err_wz", "lean", "sat"]
    print("\n" + "  ".join(f"{k:>11}" for k in keys))
    for r in rows:
        print("  ".join(f"{r[k]:>11.3f}" for k in keys))

    survived = [r for r in rows if r["survival"] >= 0.9]
    print(f"\n{len(survived)}/{len(rows)} cells at >=90% survival")
    if survived:
        print(f"  heaviest total mass held: {max(r['total_kg'] for r in survived):.2f} kg")
        print(f"  largest payload held:     {max(r['payload_kg'] for r in survived):.2f} kg")
        print(f"  largest x offset held:    "
              f"{max(abs(r['payload_x']) for r in survived):.3f} m")


def main() -> None:
    args = parse_args()
    grid, _ = build_grid(args)
    rows = run_sweep(args, grid)
    print_table(rows)

    stem = f"sweep_{args.task}_{args.checkpoint.stem}"
    if args.tag:
        stem = f"{stem}_{args.tag}"
    write_csv(rows, args.out_dir / f"{stem}.csv")
    if not args.no_plot:
        plot_heatmaps(rows, args, args.out_dir / f"{stem}.png")


if __name__ == "__main__":
    main()
