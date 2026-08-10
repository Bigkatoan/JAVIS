#!/usr/bin/env python3
"""Drive a trained JAVIS balance policy live with the keyboard.

Opens an interactive MuJoCo window. The policy does all the balancing; this
script only overrides the twist command (v_x, w_z) every physics step from
the last key pressed, the same way `/cmd_vel` would arrive from a joystick on
the real robot.

    W / S   set forward / backward speed to max (tap again to re-affirm)
    A / D   set turn left / right rate to max
    Q / E   same as A / D, in case those collide with camera controls
    SPACE   zero the command (stop)
    R       reset the episode
    ESC     quit

Each key SETS a target (not "hold to drive"): tap W and the command ramps up
to full forward and stays there until you tap something else. This is a
deliberate choice, not a shortcut -- `mujoco.viewer`'s `key_callback` is only
guaranteed to fire on key press, with no matching release event, so there is
no reliable way to detect "still being held" from it. A per-key toggle needs
only the press.

    .venv/bin/python scripts/teleop_keyboard.py \\
        --checkpoint logs/rsl_rl/javis_payload_flat/<run>/model_<n>.pt

    # drive a specific build instead of the nominal robot
    .venv/bin/python scripts/teleop_keyboard.py --checkpoint <ckpt> \\
        --chassis-kg 12 --payload-kg 6 --payload-pos 0.06 0 0.45

Why a raw `mujoco.viewer.launch_passive` window and not mjlab's own viewer:
this needs to inject external key state into the command every step, which is
a few lines against the plain MuJoCo viewer's `key_callback` and avoids
depending on mjlab's viser-based viewer internals. mjlab drives physics on the
GPU (mujoco-warp); this window renders a plain `MjData` that is resynced from
that GPU state every frame, purely for display -- the policy and physics never
touch it.
"""

import argparse
import time
from pathlib import Path

import glfw
import mujoco
import mujoco.viewer
import torch

from javis import eval_utils
from javis.mdp import events as javis_events
from javis.sim_config import JavisDomainCfg

KEY_W, KEY_A, KEY_S, KEY_D = glfw.KEY_W, glfw.KEY_A, glfw.KEY_S, glfw.KEY_D
KEY_Q, KEY_E, KEY_R = glfw.KEY_Q, glfw.KEY_E, glfw.KEY_R
KEY_SPACE, KEY_ESCAPE = glfw.KEY_SPACE, glfw.KEY_ESCAPE


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--task", choices=("flat", "rough"), default="flat")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-lin-vel", type=float, default=0.5,
                   help="m/s commanded at full W/S, matches training range")
    p.add_argument("--max-ang-vel", type=float, default=1.0,
                   help="rad/s commanded at full A/D, matches training range")
    p.add_argument("--accel", type=float, default=2.0,
                   help="how fast the command ramps toward the held key's "
                        "value, 1/s -- higher snaps faster, lower feels softer")
    p.add_argument("--chassis-kg", type=float, default=None,
                   help="fix the chassis mass instead of using the nominal build")
    p.add_argument("--payload-kg", type=float, default=0.0)
    p.add_argument("--payload-pos", type=float, nargs=3, default=(0.0, 0.0, 0.30),
                   metavar=("X", "Y", "Z"))
    return p.parse_args()


class KeyState:
    """Latest commanded (forward, turn) target, in [-1, 1] each.

    Each key press SETS the target rather than being polled for "held" state
    -- see the module docstring for why. `on_key` is exactly what
    `mujoco.viewer.launch_passive(key_callback=...)` calls, with the raw GLFW
    keycode as its only argument.
    """

    def __init__(self) -> None:
        self.forward = 0.0
        self.turn = 0.0
        self.reset_requested = False
        self.quit = False

    def on_key(self, keycode: int) -> None:
        if keycode == KEY_ESCAPE:
            self.quit = True
        elif keycode == KEY_R:
            self.reset_requested = True
        elif keycode == KEY_SPACE:
            self.forward = self.turn = 0.0
        elif keycode == KEY_W:
            self.forward = 1.0
        elif keycode == KEY_S:
            self.forward = -1.0
        elif keycode in (KEY_A, KEY_Q):
            self.turn = 1.0
        elif keycode in (KEY_D, KEY_E):
            self.turn = -1.0


def sync_render_model(base, mj_model: mujoco.MjModel) -> None:
    """Copy env 0's per-world visual fields onto the static viewer template.

    `base.sim.mj_model` is one host-side template shared across the (possibly
    many) per-world GPU models; body mass, geom size/position/colour set per
    environment by `javis.mdp.events` live only on the GPU side
    (`base.sim.model`). This is a one-time cosmetic copy so the payload box in
    the window actually matches what was configured, not the compiled default.
    """
    m = base.sim.model
    mj_model.geom_pos[:] = m.geom_pos[0].detach().cpu().numpy()
    mj_model.geom_size[:] = m.geom_size[0].detach().cpu().numpy()
    mj_model.geom_rgba[:] = m.geom_rgba[0].detach().cpu().numpy()


def sync_render_data(base, data: mujoco.MjData) -> None:
    """Copy env 0's current pose into a plain MjData, for display only."""
    qpos = base.sim.data.qpos[0].detach().cpu().numpy()
    qvel = base.sim.data.qvel[0].detach().cpu().numpy()
    data.qpos[: qpos.shape[0]] = qpos
    data.qvel[: qvel.shape[0]] = qvel


@torch.inference_mode()
def main() -> None:
    args = parse_args()

    env_cfg = eval_utils.make_env_cfg(
        task=args.task, num_envs=1, lin_vel_x=0.0, ang_vel_z=0.0,
        deterministic_load=args.chassis_kg is not None,
        episode_length_s=1e9,
    )
    env, policy, _ = eval_utils.load_policy(
        args.task, args.checkpoint, env_cfg, args.device
    )
    base = env.unwrapped

    if args.chassis_kg is not None:
        javis_events.set_load_configuration(
            base,
            torch.full((1,), args.chassis_kg, device=base.device),
            torch.full((1,), args.payload_kg, device=base.device),
            torch.tensor([args.payload_pos], device=base.device, dtype=torch.float32),
            JavisDomainCfg(),
        )
        total = args.chassis_kg + args.payload_kg + 2 * 2.936
        print(f"[teleop] fixed build: chassis {args.chassis_kg:.1f} kg, payload "
              f"{args.payload_kg:.1f} kg at {args.payload_pos}, total {total:.1f} kg, "
              f"max lean {eval_utils.max_lean_deg(total):.0f} deg")

    obs = env.get_observations()
    command_term = base.command_manager.get_term("twist")

    mj_model = base.sim.mj_model
    sync_render_model(base, mj_model)
    mj_data = mujoco.MjData(mj_model)
    sync_render_data(base, mj_data)
    mujoco.mj_forward(mj_model, mj_data)

    keys = KeyState()
    cur_forward, cur_turn = 0.0, 0.0

    print("\ncontrols: W/S forward/back, A/D or Q/E turn, SPACE stop, R reset, ESC quit\n")

    with mujoco.viewer.launch_passive(
        mj_model, mj_data, key_callback=keys.on_key
    ) as viewer:
        viewer.cam.distance = 1.8
        viewer.cam.elevation = -15
        viewer.cam.azimuth = 90

        while viewer.is_running() and not keys.quit:
            step_start = time.monotonic()

            if keys.reset_requested:
                env.reset()
                obs = env.get_observations()
                cur_forward = cur_turn = 0.0
                keys.reset_requested = False

            target_forward, target_turn = keys.forward, keys.turn

            # Ramp toward the target rather than snapping, so the command the
            # policy sees looks like a joystick, not a step function -- the
            # policy was trained on commands that resample and then hold, and
            # a discontinuous jump every keystroke is a rougher input than
            # anything in that distribution.
            dt = base.step_dt
            alpha = min(1.0, args.accel * dt)
            cur_forward += alpha * (target_forward - cur_forward)
            cur_turn += alpha * (target_turn - cur_turn)

            command_term.vel_command_b[0, 0] = cur_forward * args.max_lin_vel
            command_term.vel_command_b[0, 1] = 0.0
            command_term.vel_command_b[0, 2] = cur_turn * args.max_ang_vel

            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            if bool(dones[0]):
                print("[teleop] fell over -- press R to reset")

            sync_render_data(base, mj_data)
            mujoco.mj_forward(mj_model, mj_data)
            viewer.sync()

            elapsed = time.monotonic() - step_start
            time.sleep(max(0.0, dt - elapsed))

    env.close()


if __name__ == "__main__":
    main()
