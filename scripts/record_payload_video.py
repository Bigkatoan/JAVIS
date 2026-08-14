#!/usr/bin/env python3
"""Record the same policy driving the same command under different loads.

Renders one clip per load configuration, headless, and concatenates them into a
single mp4 with a caption strip naming the build. Watching a light robot and a
lopsided 8 kg robot execute the same twist back to back says more about whether
the policy generalizes than any table does.

The payload box is visible in frame and shaded by mass (pale blue when empty,
deep red at 10 kg), so the load is legible without reading the caption.

    .venv/bin/python scripts/record_payload_video.py \\
        --checkpoint logs/rsl_rl/javis_payload_flat/<run>/model_800.pt
"""

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

from javis import eval_utils
from javis.mdp import events as javis_events
from javis.sim_config import JavisDomainCfg

# (label, chassis_kg, payload_kg, payload_xyz). Spans the envelope corners the
# policy is meant to cover, not just a nominal build.
DEFAULT_CONFIGS = [
    ("light / empty",   4.0, 0.0, (0.00, 0.00, 0.30)),
    ("nominal",         5.3, 1.0, (0.00, 0.00, 0.30)),
    ("heavy payload",   7.0, 8.0, (0.00, 0.00, 0.30)),
    ("offset forward",  7.0, 6.0, (0.12, 0.00, 0.30)),
    ("high mount",      7.0, 6.0, (0.00, 0.00, 0.60)),
    ("heavy chassis",  14.0, 2.0, (0.00, 0.00, 0.30)),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--task", choices=("flat", "rough"), default="flat")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--tour", choices=("payload", "terrain"), default="payload",
                   help="payload: fixed command, DEFAULT_CONFIGS payload sweep "
                        "(the original mode). terrain: fixed nominal payload, one "
                        "segment per terrain type (flat/slope/slope_inv/rough), "
                        "command resampled randomly mid-segment -- requires --task rough")
    p.add_argument("--seconds", type=float, default=8.0, help="per configuration")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=540)
    p.add_argument("--lin-vel-x", type=float, default=0.3)
    p.add_argument("--ang-vel-z", type=float, default=0.4)
    p.add_argument("--command-resample-s", type=float, default=4.0,
                   help="(--tour terrain only) seconds between random target draws")
    p.add_argument("--no-vectors", action="store_true",
                   help="skip the target-vs-current velocity vector panel")
    p.add_argument("--out", type=Path, default=Path("logs/eval/payload_configs.mp4"))
    return p.parse_args()


def render_vector_panel(
    target_xy: tuple[float, float],
    current_xy: tuple[float, float],
    target_wz: float,
    current_wz: float,
    max_speed: float,
    size_px: int = 260,
) -> np.ndarray:
    """Draw the two vectors that matter for judging tracking at a glance:
    target (commanded) velocity vs. current (actual) velocity, as arrows on a
    compass with "forward" (the robot's local +x) pointing up.

    Both come straight from the same frame the policy itself sees -- the
    command term's body-frame (vx, vy) and `root_link_lin_vel_b` -- so what is
    drawn is exactly what the policy is trying to match, not a world-frame
    reprojection of it. When the arrows overlap, tracking is good; when the
    orange (current) arrow points somewhere the green (target) one doesn't,
    that gap *is* the tracking error, and it is visible as a gap, not a number
    to interpret.

    Yaw rate has no spatial direction to draw, so it is a pair of numbers
    under the compass rather than a third arrow -- keeping this to exactly two
    vectors, per the ask.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, FancyArrow

    dpi = 100
    fig = plt.figure(figsize=(size_px / dpi, size_px / dpi), dpi=dpi)
    fig.patch.set_alpha(0.0)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.set_facecolor("#101216")
    ax.patch.set_alpha(0.78)

    lim = max_speed * 1.15
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.axis("off")

    for frac, label in ((0.5, None), (1.0, f"{max_speed:g} m/s")):
        ax.add_patch(Circle((0, 0), max_speed * frac, fill=False,
                            edgecolor="#3a4048", linewidth=1.0))
        if label:
            ax.text(0, max_speed * frac, label, color="#5a6270", fontsize=7,
                    ha="center", va="bottom")
    ax.axhline(0, color="#3a4048", linewidth=1.0)
    ax.axvline(0, color="#3a4048", linewidth=1.0)
    ax.text(0, lim * 0.98, "fwd", color="#5a6270", fontsize=7,
            ha="center", va="top")

    def arrow(xy: tuple[float, float], color: str, label: str) -> None:
        vx, vy = xy
        # Body frame: x = forward, y = left. Compass: up = forward,
        # right = -left, i.e. plot (-vy, vx).
        dx, dy = -vy, vx
        if abs(dx) > 1e-4 or abs(dy) > 1e-4:
            ax.add_patch(FancyArrow(
                0, 0, dx, dy, width=max_speed * 0.025,
                head_width=max_speed * 0.11, head_length=max_speed * 0.14,
                length_includes_head=True, color=color, alpha=0.9,
            ))
        else:
            ax.add_patch(Circle((0, 0), max_speed * 0.02, color=color, alpha=0.9))
        speed = float(np.hypot(vx, vy))
        ax.text(lim * -0.95, lim * (0.86 if label == "target" else 0.72),
                f"{label}  {speed:.2f} m/s", color=color, fontsize=8,
                ha="left", va="center", family="monospace")

    arrow(target_xy, "#3ddc6a", "target")
    arrow(current_xy, "#ff9d3d", "current")

    ax.text(lim * -0.95, lim * -0.88,
            f"yaw target {target_wz:+.2f}  actual {current_wz:+.2f} rad/s",
            color="#c7ccd3", fontsize=7, ha="left", va="center", family="monospace")

    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    return rgba


def composite_rgba(frame: np.ndarray, overlay_rgba: np.ndarray,
                   x: int, y: int) -> np.ndarray:
    """Alpha-blend `overlay_rgba` onto `frame` (RGB) with its top-left at (x, y)."""
    h, w = overlay_rgba.shape[:2]
    y2, x2 = min(y + h, frame.shape[0]), min(x + w, frame.shape[1])
    if y2 <= y or x2 <= x:
        return frame
    region = overlay_rgba[: y2 - y, : x2 - x]
    alpha = region[..., 3:4].astype(np.float32) / 255.0
    out = frame.copy()
    bg = out[y:y2, x:x2].astype(np.float32)
    fg = region[..., :3].astype(np.float32)
    out[y:y2, x:x2] = (alpha * fg + (1 - alpha) * bg).astype(np.uint8)
    return out


def render_caption(lines: list[str], width: int) -> np.ndarray:
    """Rasterize a caption strip with matplotlib.

    matplotlib is already a dependency (the sweep heatmaps use it) and ships
    its own fonts, so this needs nothing extra installed and renders the same
    on any machine -- which matters for a clip meant to be shared.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dpi = 100
    height = 22 * len(lines) + 14
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.patch.set_facecolor("#101216")
    for i, line in enumerate(lines):
        fig.text(
            0.01, 1.0 - (i + 0.85) / len(lines) * 0.92, line,
            color="#f0f2f5" if i == 0 else "#aab2bd",
            fontsize=13 if i == 0 else 10,
            family="monospace", va="center",
        )
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return buf


@torch.inference_mode()
def record(args) -> None:
    env_cfg = eval_utils.make_env_cfg(
        task=args.task,
        num_envs=1,
        lin_vel_x=args.lin_vel_x,
        ang_vel_z=args.ang_vel_z,
        # Long enough that the clip never ends on a time-out mid-recording.
        episode_length_s=args.seconds * len(DEFAULT_CONFIGS) + 10.0,
    )
    env_cfg.viewer.width = args.width
    env_cfg.viewer.height = args.height
    env_cfg.viewer.distance = 2.2
    env_cfg.viewer.elevation = -15.0
    env_cfg.viewer.azimuth = 120.0
    env_cfg.viewer.max_extra_envs = 0

    env, policy, _ = eval_utils.load_policy(
        args.task, args.checkpoint, env_cfg, args.device, render_mode="rgb_array"
    )
    base = env.unwrapped
    robot = base.scene["robot"]
    domain = JavisDomainCfg()
    # Fixed rather than auto-scaled per frame: a constant ring scale is what
    # makes the panel comparable across configurations and across time, and
    # 0.6 covers the +-0.5 m/s the policy was trained on with a small margin.
    vector_scale = max(0.6, abs(args.lin_vel_x) * 1.2)

    steps = int(round(args.seconds / base.step_dt))
    render_every = max(1, int(round(1.0 / (args.fps * base.step_dt))))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(args.out, fps=args.fps, macro_block_size=1)
    frames = 0

    for label, chassis_kg, payload_kg, pos in DEFAULT_CONFIGS:
        env.reset()
        javis_events.set_load_configuration(
            base,
            torch.full((1,), chassis_kg, device=base.device),
            torch.full((1,), payload_kg, device=base.device),
            torch.tensor([pos], device=base.device, dtype=torch.float32),
            domain,
        )
        obs = env.get_observations()
        total = chassis_kg + payload_kg + 2 * 2.936
        header = [
            label,
            f"chassis {chassis_kg:4.1f} kg   payload {payload_kg:4.1f} kg   "
            f"total {total:5.1f} kg   max lean "
            f"{eval_utils.max_lean_deg(total):.0f} deg",
            f"mount ({pos[0]:+.2f}, {pos[1]:+.2f}, {pos[2]:.2f}) m    "
            f"command  vx {args.lin_vel_x:+.2f} m/s   wz {args.ang_vel_z:+.2f} rad/s",
        ]
        print(f"[video] {label}: total {total:.1f} kg")

        fell_at: float | None = None
        for step in range(steps):
            obs, _, dones, extras = env.step(policy(obs))
            t = step * base.step_dt
            timed_out = extras.get("time_outs")
            fell = bool(dones[0]) and not (
                timed_out is not None and bool(timed_out[0])
            )
            if fell_at is None and fell:
                fell_at = t

            if step % render_every == 0:
                frame = base.render()
                if frame is None:
                    raise RuntimeError("render() returned None; is render_mode set?")
                status = f"FELL at {fell_at:.2f}s" if fell_at is not None else "upright"
                strip = render_caption(header + [f"t {t:5.2f}s    {status}"],
                                       frame.shape[1])
                h = min(strip.shape[0], frame.shape[0])
                out = np.asarray(frame).copy()
                out[:h] = strip[:h]

                if not args.no_vectors:
                    cmd = base.command_manager.get_command("twist")[0].cpu().numpy()
                    lin_actual = robot.data.root_link_lin_vel_b[0, :2].cpu().numpy()
                    ang_actual = float(robot.data.root_link_ang_vel_b[0, 2])
                    panel = render_vector_panel(
                        target_xy=(float(cmd[0]), float(cmd[1])),
                        current_xy=(float(lin_actual[0]), float(lin_actual[1])),
                        target_wz=float(cmd[2]), current_wz=ang_actual,
                        max_speed=vector_scale,
                    )
                    out = composite_rgba(out, panel, x=out.shape[1] - panel.shape[1] - 8,
                                         y=out.shape[0] - panel.shape[0] - 8)

                writer.append_data(out)
                frames += 1

    writer.close()
    env.close()
    print(f"[video] wrote {args.out} ({frames} frames, "
          f"{frames / args.fps:.1f}s at {args.fps} fps)")


TERRAIN_TYPE_NAMES = ("flat", "slope (uphill)", "slope (downhill)", "rough")


@torch.inference_mode()
def record_terrain_tour(args) -> None:
    """One segment per terrain type, command resampled at random along the way.

    Runs enough environments in parallel that every terrain column (mjlab's
    curriculum=True terrain generator gives each sub-terrain type its own
    column -- see javis/balance_task.py's _terrain_cfg) is populated by at
    least one of them, then records whichever env sits on each type in turn.
    All envs step together the whole time; only which one gets rendered
    changes between segments, so this costs one build, not four.

    Payload is pinned to one moderate, fixed build (not randomized) so
    terrain and command -- the two things actually asked for -- are what's
    varying, not a third factor confounding both.
    """
    if args.task != "rough":
        print("[video] --tour terrain needs a terrain generator; forcing --task rough")
    num_envs = 16  # enough that random type assignment covers all 4 with margin

    env_cfg = eval_utils.make_env_cfg(
        task="rough", num_envs=num_envs, lin_vel_x=0.0, ang_vel_z=0.0,
        deterministic_load=True, episode_length_s=1e9,
    )
    env_cfg.viewer.width = args.width
    env_cfg.viewer.height = args.height
    # Pulled back and flatter than record()'s payload-tour framing (2.2 / -15):
    # slope/rough patches vary in height at the spawn point (their env_origins
    # carry a real z offset, unlike flat ground), and a close, steep default
    # can end up looking straight into the terrain surface for those.
    env_cfg.viewer.distance = 3.5
    env_cfg.viewer.elevation = -10.0
    env_cfg.viewer.azimuth = 120.0
    env_cfg.viewer.max_extra_envs = 0

    env, policy, _ = eval_utils.load_policy(
        "rough", args.checkpoint, env_cfg, args.device, render_mode="rgb_array"
    )
    base = env.unwrapped
    domain = JavisDomainCfg()

    env.reset()
    terrain_types = base.scene.terrain.terrain_types.cpu()
    env_by_type: dict[int, int] = {}
    for i, t in enumerate(terrain_types.tolist()):
        env_by_type.setdefault(t, i)
    missing = [n for t, n in enumerate(TERRAIN_TYPE_NAMES) if t not in env_by_type]
    if missing:
        print(f"[video] warning: no env landed on {missing} out of {num_envs} envs "
              f"-- re-run for different luck, or raise num_envs in this script")

    chassis_kg, payload_kg, payload_pos = 6.0, 4.0, (0.0, 0.0, 0.30)
    javis_events.set_load_configuration(
        base,
        torch.full((num_envs,), chassis_kg, device=base.device),
        torch.full((num_envs,), payload_kg, device=base.device),
        torch.tensor([payload_pos] * num_envs, device=base.device, dtype=torch.float32),
        domain,
    )
    obs = env.get_observations()
    command_term = base.command_manager.get_term("twist")

    steps = int(round(args.seconds / base.step_dt))
    render_every = max(1, int(round(1.0 / (args.fps * base.step_dt))))
    resample_every = max(1, int(round(args.command_resample_s / base.step_dt)))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(args.out, fps=args.fps, macro_block_size=1)
    frames = 0
    global_step = 0
    rng = np.random.default_rng(0)

    def resample_command() -> tuple[float, float]:
        vx = float(rng.uniform(-0.5, 0.5))
        wz = float(rng.uniform(-1.0, 1.0))
        command_term.vel_command_b[:, 0] = vx
        command_term.vel_command_b[:, 1] = 0.0
        command_term.vel_command_b[:, 2] = wz
        return vx, wz

    cur_vx, cur_wz = resample_command()

    for terrain_id, type_name in enumerate(TERRAIN_TYPE_NAMES):
        env_idx = env_by_type.get(terrain_id)
        if env_idx is None:
            continue
        env_cfg.viewer.env_idx = env_idx
        total = chassis_kg + payload_kg + 2 * 2.936
        print(f"[video] terrain={type_name} (env {env_idx})")

        fell_at: float | None = None
        for step in range(steps):
            if global_step % resample_every == 0:
                cur_vx, cur_wz = resample_command()
            global_step += 1

            obs, _, dones, extras = env.step(policy(obs))
            t = step * base.step_dt
            timed_out = extras.get("time_outs")
            fell = bool(dones[env_idx]) and not (
                timed_out is not None and bool(timed_out[env_idx])
            )
            if fell_at is None and fell:
                fell_at = t

            if step % render_every == 0:
                frame = base.render()
                if frame is None:
                    raise RuntimeError("render() returned None; is render_mode set?")
                status = f"FELL at {fell_at:.2f}s" if fell_at is not None else "upright"
                header = [
                    f"terrain: {type_name}",
                    f"chassis {chassis_kg:.1f} kg  payload {payload_kg:.1f} kg  "
                    f"total {total:.1f} kg   (fixed -- terrain/command are what's varying)",
                    f"target  vx {cur_vx:+.2f} m/s   wz {cur_wz:+.2f} rad/s   "
                    f"(resamples every {args.command_resample_s:g}s)",
                ]
                strip = render_caption(header + [f"t {t:5.2f}s    {status}"],
                                       frame.shape[1])
                h = min(strip.shape[0], frame.shape[0])
                out = np.asarray(frame).copy()
                out[:h] = strip[:h]

                if not args.no_vectors:
                    cmd = base.command_manager.get_command("twist")[env_idx].cpu().numpy()
                    robot = base.scene["robot"]
                    lin_actual = robot.data.root_link_lin_vel_b[env_idx, :2].cpu().numpy()
                    ang_actual = float(robot.data.root_link_ang_vel_b[env_idx, 2])
                    panel = render_vector_panel(
                        target_xy=(float(cmd[0]), float(cmd[1])),
                        current_xy=(float(lin_actual[0]), float(lin_actual[1])),
                        target_wz=float(cmd[2]), current_wz=ang_actual,
                        max_speed=0.6,
                    )
                    out = composite_rgba(out, panel, x=out.shape[1] - panel.shape[1] - 8,
                                         y=out.shape[0] - panel.shape[0] - 8)

                writer.append_data(out)
                frames += 1

    writer.close()
    env.close()
    print(f"[video] wrote {args.out} ({frames} frames, "
          f"{frames / args.fps:.1f}s at {args.fps} fps)")


def main() -> None:
    args = parse_args()
    if args.tour == "terrain":
        record_terrain_tour(args)
    else:
        record(args)


if __name__ == "__main__":
    main()
