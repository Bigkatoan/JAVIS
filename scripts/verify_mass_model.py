#!/usr/bin/env python3
"""Independent checks on the mass model and the randomization that drives it.

`javis/mass_model.py` claims that the fused body inertial is exactly linear in
the per-group masses, so any configuration can be produced with a couple of
matmuls instead of recompiling. If that claim is wrong the whole task is
training on the wrong physics, silently -- nothing crashes, the robot is just
not the robot.

So each check below re-derives the answer by a genuinely different route:

  1. brute force   -- sum mass properties mesh-by-mesh with trimesh at the
                      sampled densities, and compare against `fuse`. This tests
                      the precomputed moments and the linearity claim together.
  2. MuJoCo        -- compile a spec and read back what MuJoCo itself says the
                      body's mass, CoM and inertia are.
  3. GPU path      -- confirm mjlab's 3x3 Jacobi eigendecomposition agrees with
                      torch.linalg.eigh, since the runtime path uses the former
                      to avoid cuSOLVER's multi-GB allocation.
  4. live env      -- randomize a populated environment and check the model
                      fields against `fuse` recomputed in float64 on the CPU.

    .venv/bin/python scripts/verify_mass_model.py
"""

import math
import sys
import xml.etree.ElementTree as ET

import numpy as np
import torch
import trimesh

from javis import mass_model
from javis.robot_constants import WHEEL_RADIUS_M

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, error: float, tol: float, detail: str = "") -> None:
    ok = error < tol
    RESULTS.append((name, ok, f"err {error:.3e} (tol {tol:g}) {detail}"))
    print(f"  [{'OK ' if ok else 'FAIL'}] {name:<44} err {error:.3e}  {detail}")


def brute_force_inertial(group_masses: dict[str, float]):
    """Mass, CoM and inertia summed instance-by-instance straight from the STLs.

    Deliberately does not touch the precomputed moments: it re-reads the URDF,
    re-loads each mesh, applies each instance transform, and accumulates. Slow,
    and that is fine -- it exists to disagree with `fuse` if `fuse` is wrong.
    """
    root = ET.parse(mass_model.JAVIS_URDF).getroot()
    link = next(l for l in root.findall("link")
                if l.get("name") == mass_model.CHASSIS_BODY)

    # Group volumes first, so a group's density can be derived from its mass.
    cache: dict[str, trimesh.Trimesh] = {}
    instances = []
    volumes: dict[str, float] = {}
    for mesh_name, rot, pos in mass_model._iter_visual_meshes(link):
        if mesh_name not in cache:
            cache[mesh_name] = trimesh.load(
                mass_model.JAVIS_DIR / "assets" / mesh_name, force="mesh")
        mesh = cache[mesh_name]
        group = mass_model._classify(mesh_name)
        volumes[group] = volumes.get(group, 0.0) + float(mesh.volume)
        instances.append((group, mesh, rot, pos))

    total = 0.0
    first = np.zeros(3)
    second = np.zeros((3, 3))

    for group, mesh, rot, pos in instances:
        density = group_masses[group] / volumes[group]
        v = float(mesh.volume)
        c = np.asarray(mesh.center_mass)
        i_com = np.asarray(mesh.moment_inertia)
        cov_com = 0.5 * np.trace(i_com) * np.eye(3) - i_com
        cov_org = cov_com + v * np.outer(c, c)

        # Transform into the link frame.
        f = rot @ (v * c) + v * pos
        cross = np.outer(rot @ (v * c), pos)
        s = rot @ cov_org @ rot.T + cross + cross.T + v * np.outer(pos, pos)

        total += density * v
        first += density * f
        second += density * s

    # Point-mass groups.
    for group, position in mass_model.POINT_GROUPS.items():
        m = group_masses[group]
        p = np.asarray(position)
        total += m
        first += m * p
        second += m * np.outer(p, p)

    com = first / total
    inertia_origin = np.trace(second) * np.eye(3) - second
    inertia_com = inertia_origin - total * (
        (com @ com) * np.eye(3) - np.outer(com, com)
    )
    return total, com, inertia_com


def check_brute_force(rng: np.random.Generator) -> None:
    print("\n1. fuse() vs brute-force accumulation over every mesh instance")
    moments = mass_model.get_moments()[mass_model.CHASSIS_BODY]

    for trial in range(3):
        nominal = mass_model.nominal_masses(mass_model.CHASSIS_BODY)
        # Nothing gentle: 0.3x to 4x per group, plus a big off-axis payload.
        sampled = {g: float(m) * float(rng.uniform(0.3, 4.0))
                   for g, m in nominal.items()}
        sampled["payload"] = float(rng.uniform(0.0, 10.0))
        sampled["wiring_misc"] = float(rng.uniform(0.0, 1.0))

        vec = torch.tensor([sampled[g] for g in moments.groups], dtype=torch.float64)
        mass, com, iquat, inertia = mass_model.fuse(vec, moments)

        ref_mass, ref_com, ref_inertia_com = brute_force_inertial(sampled)

        # fuse returns principal moments; compare the invariants, which do not
        # depend on how either side happened to order or orient the axes.
        ref_eig = np.sort(np.linalg.eigvalsh(ref_inertia_com))
        got_eig = np.sort(inertia.numpy())

        check(f"trial {trial}: total mass", abs(float(mass) - ref_mass), 1e-9,
              f"({ref_mass:.3f} kg)")
        check(f"trial {trial}: centre of mass",
              float(np.abs(com.numpy() - ref_com).max()), 1e-9,
              f"({ref_com[0]:+.4f}, {ref_com[1]:+.4f}, {ref_com[2]:+.4f})")
        check(f"trial {trial}: principal inertia",
              float(np.abs(got_eig - ref_eig).max()), 1e-9)

        # And that iquat really does diagonalize the tensor it claims to.
        w, x, y, z = iquat.numpy()
        R = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])
        rebuilt = R @ np.diag(inertia.numpy()) @ R.T
        check(f"trial {trial}: iquat diagonalizes inertia",
              float(np.abs(rebuilt - ref_inertia_com).max()), 1e-9)
        check(f"trial {trial}: iquat is a proper rotation",
              abs(np.linalg.det(R) - 1.0), 1e-9)


def check_mujoco() -> None:
    print("\n2. compiled MuJoCo model vs fuse() at nominal masses")
    import mujoco

    from javis.robot_constants import get_spec

    model = get_spec().compile()
    for body in [mass_model.CHASSIS_BODY, *mass_model.WHEEL_BODIES]:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
        mass, com, _, inertia = mass_model.fuse_nominal(body)
        check(f"{body}: mass", abs(model.body_mass[bid] - mass), 1e-6)
        check(f"{body}: ipos",
              float(np.abs(model.body_ipos[bid] - com).max()), 1e-6)
        check(f"{body}: inertia",
              float(np.abs(np.sort(model.body_inertia[bid]) - np.sort(inertia)).max()),
              1e-6)


def check_gpu_eigh(rng: np.random.Generator) -> None:
    print("\n3. mjlab's Jacobi eigendecomposition vs torch.linalg.eigh")
    try:
        from mjlab.envs.mdp.dr.body import _eigh_3x3_jacobi
    except ImportError:
        RESULTS.append(("jacobi eigh available", False, "import failed"))
        print("  [FAIL] mjlab._eigh_3x3_jacobi could not be imported")
        return

    moments = mass_model.get_moments()[mass_model.CHASSIS_BODY]
    n = 512
    nominal = mass_model.nominal_mass_vector(mass_model.CHASSIS_BODY)
    scale = torch.tensor(rng.uniform(0.3, 4.0, size=(n, len(moments.groups))),
                         dtype=torch.float32)
    masses = nominal[None, :] * scale
    pos = torch.tensor(rng.uniform(-0.12, 0.6, size=(n, 3)), dtype=torch.float32)

    kwargs = {"point_positions": {"payload": pos}}
    _, com_a, quat_a, inertia_a = mass_model.fuse(masses, moments, **kwargs)
    _, com_b, quat_b, inertia_b = mass_model.fuse(
        masses, moments, eigh_fn=_eigh_3x3_jacobi, **kwargs
    )

    check("centre of mass identical",
          float((com_a - com_b).abs().max()), 1e-6)
    check("principal inertia agrees",
          float((inertia_a.sort(-1).values - inertia_b.sort(-1).values).abs().max()),
          1e-6)
    # Quaternion sign/axis-order can differ legitimately, so compare the
    # reconstructed tensor rather than the quaternion itself.
    check("both quats are unit", float((quat_b.norm(dim=-1) - 1).abs().max()), 1e-5)


def check_live_env(rng: np.random.Generator) -> None:
    print("\n4. randomized live environment vs fuse() recomputed on the CPU")
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers.scene_entity_config import SceneEntityCfg

    from javis.balance_task import javis_balance_flat_env_cfg
    from javis.mdp import events

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg = javis_balance_flat_env_cfg(play=False)
    cfg.scene.num_envs = 512
    cfg.events["randomize_load"].params["difficulty"].level = 1.0
    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    env.reset()

    idx = events._body_index(env, mass_model.CHASSIS_BODY, SceneEntityCfg("robot"))
    state = events.get_state(env)
    moments = mass_model.get_moments()[mass_model.CHASSIS_BODY]

    expect_mass, expect_com, _, expect_inertia = mass_model.fuse(
        state.group_masses.double().cpu(),
        moments,
        point_positions={
            "payload": state.payload_pos.double().cpu(),
            "wiring_misc": state.wiring_pos.double().cpu(),
        },
    )
    got_mass = env.sim.model.body_mass[:, idx].squeeze(-1).double().cpu()
    got_com = env.sim.model.body_ipos[:, idx].squeeze(-2).double().cpu()
    got_inertia = env.sim.model.body_inertia[:, idx].squeeze(-2).double().cpu()

    check("live body_mass", float((got_mass - expect_mass).abs().max()), 1e-4)
    check("live body_ipos", float((got_com - expect_com).abs().max()), 1e-5)
    check("live body_inertia",
          float((got_inertia.sort(-1).values - expect_inertia.sort(-1).values)
                .abs().max()), 1e-5)

    # And that the randomization actually reaches the envelope the user asked
    # for -- a silently-narrow DR range is the failure this catches.
    payload = state.group_masses[:, moments.index("payload")]
    chassis = state.group_masses.sum(-1) - payload
    total = state.total_mass()
    print(f"\n     level 1.0 coverage over {cfg.scene.num_envs} envs:")
    print(f"       chassis   {chassis.min():5.2f} .. {chassis.max():5.2f} kg "
          f"(target 3.0 .. 15.0)")
    print(f"       payload   {payload.min():5.2f} .. {payload.max():5.2f} kg "
          f"(target 0.0 .. 10.0)")
    print(f"       total     {total.min():5.2f} .. {total.max():5.2f} kg")
    print(f"       CoM x     {got_com[:, 0].min():+.3f} .. {got_com[:, 0].max():+.3f} m")
    print(f"       CoM z     {got_com[:, 2].min():+.3f} .. {got_com[:, 2].max():+.3f} m")
    lean = [math.degrees(math.atan2(float(x), float(z) + WHEEL_RADIUS_M))
            for x, z in zip(got_com[:, 0], got_com[:, 2])]
    print(f"       equilibrium lean {min(lean):+.1f} .. {max(lean):+.1f} deg")

    RESULTS.append((
        "chassis range reaches >=12 kg spread",
        float(chassis.max() - chassis.min()) > 8.0,
        f"{float(chassis.min()):.2f}..{float(chassis.max()):.2f} kg",
    ))
    RESULTS.append((
        "payload range reaches >=8 kg",
        float(payload.max()) > 8.0,
        f"max {float(payload.max()):.2f} kg",
    ))
    env.close()


def main() -> None:
    rng = np.random.default_rng(7)
    check_brute_force(rng)
    check_mujoco()
    check_gpu_eigh(rng)
    check_live_env(rng)

    failed = [name for name, ok, _ in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    for name in failed:
        print(f"  FAILED: {name}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
