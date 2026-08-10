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
    p.add_argument("--seconds", type=float, default=8.0, help="per configuration")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=540)
    p.add_argument("--lin-vel-x", type=float, default=0.3)
    p.add_argument("--ang-vel-z", type=float, default=0.4)
    p.add_argument("--out", type=Path, default=Path("logs/eval/payload_configs.mp4"))
    return p.parse_args()


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
    domain = JavisDomainCfg()

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
                writer.append_data(out)
                frames += 1

    writer.close()
    env.close()
    print(f"[video] wrote {args.out} ({frames} frames, "
          f"{frames / args.fps:.1f}s at {args.fps} fps)")


def main() -> None:
    record(parse_args())


if __name__ == "__main__":
    main()
