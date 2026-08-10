#!/usr/bin/env python3
"""Export a trained actor to ONNX and verify it against PyTorch.

Produces the artifact the Jetson driver will actually run, plus the numbers
needed to wire it up: observation layout in order, action meaning and units,
and the control rate the policy was trained at. Getting that contract wrong is
the classic sim-to-real failure -- a policy fed its observations in a different
order than it was trained on fails silently, and on a balancing robot "fails
silently" means it falls over.

The parity check is not a formality. The exported graph folds in observation
normalization and the flattened observation history; if either is wrong the
ONNX output diverges from PyTorch, and this catches it here rather than on the
robot.

    .venv/bin/python scripts/export_onnx.py \\
        --checkpoint logs/rsl_rl/javis_payload_flat/<run>/model_800.pt
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from javis import eval_utils
from javis.balance_task import OBS_HISTORY_LENGTH
from javis.robot_constants import WHEEL_JOINTS, WHEEL_RADIUS_M


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--task", choices=("flat", "rough"), default="flat")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="defaults to the checkpoint's own directory")
    p.add_argument("--samples", type=int, default=256,
                   help="random observations used for the parity check")
    p.add_argument("--tolerance", type=float, default=1e-4)
    return p.parse_args()


def observation_contract(env, history_length: int) -> list[dict]:
    """Per-term slice of the observation vector, and its internal layout.

    The layout is TERM-major, not frame-major, and it is easy to get backwards.
    mjlab keeps one circular buffer per term and flattens each one before
    concatenating, so the vector is:

        [ term0: frame(t-11) .. frame(t-0) ][ term1: frame(t-11) .. frame(t-0) ] ...

    not a sequence of whole frames. A driver that assembles it frame-major will
    produce a vector of the right length that means something completely
    different, and nothing will raise -- the robot will just fall over. Dumped
    to JSON next to the .onnx so it cannot drift out of sync with the code.
    """
    manager = env.unwrapped.observation_manager
    names = manager.active_terms["actor"]
    dims = manager.group_obs_term_dim["actor"]

    contract, offset = [], 0
    for name, dim in zip(names, dims):
        width = int(np.prod(dim))
        per_frame = width // history_length
        contract.append({
            "name": name,
            "slice": [offset, offset + width],
            "width": width,
            "values_per_frame": per_frame,
            "frames": history_length,
            "frame_order": "oldest first; the last block is the current step",
        })
        offset += width
    return contract


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or args.checkpoint.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = eval_utils.make_env_cfg(
        task=args.task, num_envs=1, lin_vel_x=0.0, ang_vel_z=0.0,
        deterministic_load=False,
    )
    env, _, runner = eval_utils.load_policy(
        args.task, args.checkpoint, env_cfg, args.device
    )

    onnx_name = f"{args.checkpoint.stem}.onnx"
    runner.export_policy_to_onnx(str(out_dir), filename=onnx_name)
    onnx_path = out_dir / onnx_name
    print(f"[onnx] wrote {onnx_path}")

    # Parity: same random observations through both graphs.
    import onnxruntime as ort

    obs_dim = env.unwrapped.observation_manager.group_obs_dim["actor"][0]
    rng = np.random.default_rng(0)
    sample = rng.normal(size=(args.samples, obs_dim)).astype(np.float32)

    policy_module = runner.alg.get_policy().as_onnx(verbose=False)
    policy_module.to("cpu").eval()
    with torch.inference_mode():
        torch_out = policy_module(torch.from_numpy(sample)).cpu().numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_spec = session.get_inputs()[0]
    input_name = input_spec.name
    assert input_spec.shape[-1] == obs_dim, (
        f"ONNX expects {input_spec.shape[-1]} observations, env produces {obs_dim}"
    )

    # mjlab exports with dynamic_axes={}, so the batch dimension is frozen at
    # the dummy input's size (1). Feed one row at a time -- which is also
    # exactly how the Jetson will call it, one control step at a time.
    batch = input_spec.shape[0] if isinstance(input_spec.shape[0], int) else args.samples
    onnx_out = np.concatenate(
        [session.run(None, {input_name: sample[i:i + batch]})[0]
         for i in range(0, args.samples, batch)],
        axis=0,
    )[: args.samples]

    max_abs = float(np.abs(torch_out - onnx_out).max())
    ok = max_abs < args.tolerance
    print(f"[onnx] parity over {args.samples} samples: max |torch - onnx| = "
          f"{max_abs:.3e}  ({'OK' if ok else 'FAIL'}, tolerance {args.tolerance:g})")

    contract = {
        "checkpoint": str(args.checkpoint),
        "onnx": str(onnx_path),
        "control_rate_hz": round(1.0 / env.unwrapped.step_dt, 3),
        "observation": {
            "total_dim": int(obs_dim),
            "history_length": OBS_HISTORY_LENGTH,
            "layout": (
                "TERM-major: each term contributes its whole history as one "
                "contiguous block, then the blocks are concatenated. Within a "
                "block, frames run oldest to newest. NOT a sequence of frames."
            ),
            "terms": observation_contract(env, OBS_HISTORY_LENGTH),
            "note": (
                "Normalization is baked into the exported graph -- feed raw "
                "observations, exactly as listed, in this order. Zero-fill the "
                "history on startup; the policy was trained with buffers that "
                "start at zero after every reset."
            ),
        },
        "action": {
            "dim": len(WHEEL_JOINTS),
            "order": list(WHEEL_JOINTS),
            "meaning": "wheel velocity target",
            "units": "rad/s after multiplying the network output by `scale`",
            "scale": float(env.unwrapped.action_manager.get_term("wheel_vel").scale),
            "wheel_radius_m": WHEEL_RADIUS_M,
            "to_odrive_input_vel": (
                "input_vel [turn/s] = target [rad/s] / (2*pi); the drivetrain is "
                "1:1, so no gear ratio is applied (SIM2REAL.md sec 3)."
            ),
        },
        "parity": {"max_abs_diff": max_abs, "tolerance": args.tolerance, "ok": ok},
    }
    contract_path = out_dir / f"{args.checkpoint.stem}_contract.json"
    contract_path.write_text(json.dumps(contract, indent=2))
    print(f"[onnx] wrote {contract_path}")

    env.close()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
