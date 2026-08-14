#!/usr/bin/env python3
"""Interactively view the JAVIS rover in an mjlab Scene (flat ground plane,
no RL task involved -- just a sanity check that robot.urdf + robot_constants.py
still build into a valid simulation).

Run after changing javis/robot.urdf or javis/robot_constants.py:
    .venv/bin/python scripts/view_robot.py

Inspecting the mass model:
    # colour every chassis part by which mass group it was assigned to --
    # the fastest way to catch a mesh landing in the wrong group, which would
    # silently move the centre of mass
    .venv/bin/python scripts/view_robot.py --color-by-group

    # see (and feel) what a load does
    .venv/bin/python scripts/view_robot.py --payload-kg 8 --payload-pos 0.1 0 0.5
"""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import mujoco.viewer

from javis import mass_model
from javis.robot_constants import PAYLOAD_GEOM, get_javis_robot_cfg, get_spec

from mjlab.scene import Scene, SceneCfg
from mjlab.terrains import TerrainEntityCfg

# Distinct hues per group; anything unassigned stays grey and stands out.
GROUP_COLORS = {
    "battery": (0.90, 0.20, 0.20, 1.0),
    "printed": (0.30, 0.65, 0.95, 1.0),
    "jetson": (0.20, 0.85, 0.40, 1.0),
    "camera": (0.95, 0.75, 0.15, 1.0),
    "odrive": (0.80, 0.35, 0.90, 1.0),
    "imu": (0.15, 0.90, 0.90, 1.0),
    "hardware": (0.55, 0.55, 0.60, 1.0),
    "electronics_misc": (0.95, 0.50, 0.25, 1.0),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--payload-kg", type=float, default=0.0)
    p.add_argument("--payload-pos", type=float, nargs=3, metavar=("X", "Y", "Z"),
                   default=(0.0, 0.0, 0.30),
                   help="payload mount position in the chassis frame, m")
    p.add_argument("--color-by-group", action="store_true",
                   help="tint each chassis mesh by its javis.mass_model group")
    return p.parse_args()


def _mesh_to_group() -> dict[str, str]:
    """Which group each chassis mesh filename was classified into."""
    root = ET.parse(mass_model.JAVIS_URDF).getroot()
    out: dict[str, str] = {}
    for link in root.findall("link"):
        if link.get("name") != mass_model.CHASSIS_BODY:
            continue
        for mesh_name, _, _ in mass_model._iter_visual_meshes(link):
            out[Path(mesh_name).stem] = mass_model._classify(mesh_name)
    return out


def apply_group_colors(spec: mujoco.MjSpec) -> None:
    mesh_group = _mesh_to_group()
    counts: dict[str, int] = {}
    for geom in spec.geoms:
        if geom.type != mujoco.mjtGeom.mjGEOM_MESH or not geom.meshname:
            continue
        group = mesh_group.get(Path(geom.meshname).stem)
        if group is None:
            continue
        geom.rgba = list(GROUP_COLORS.get(group, (0.5, 0.5, 0.5, 1.0)))
        counts[group] = counts.get(group, 0) + 1

    print("\ngeoms tinted per group:")
    for group, n in sorted(counts.items()):
        rgb = GROUP_COLORS.get(group, (0.5, 0.5, 0.5, 1.0))
        print(f"  {group:18s} {n:4d} geoms   rgb({rgb[0]:.2f}, {rgb[1]:.2f}, {rgb[2]:.2f})")


def apply_payload(spec: mujoco.MjSpec, payload_kg: float, pos) -> None:
    """Move the payload geom and refuse the chassis inertial with it included."""
    chassis = next(b for b in spec.bodies if b.name == mass_model.CHASSIS_BODY)
    geom = next(g for g in chassis.geoms if g.name == PAYLOAD_GEOM)
    geom.pos = list(pos)

    moments = mass_model.get_moments()[mass_model.CHASSIS_BODY]
    masses = mass_model.nominal_mass_vector(mass_model.CHASSIS_BODY)
    masses[moments.index("payload")] = payload_kg

    import torch

    mass, com, iquat, inertia = mass_model.fuse(
        masses, moments,
        point_positions={"payload": torch.tensor(pos, dtype=torch.float32)},
    )
    chassis.mass = float(mass)
    chassis.ipos = com.tolist()
    chassis.iquat = iquat.tolist()
    chassis.inertia = inertia.tolist()

    from javis.robot_constants import WHEEL_RADIUS_M

    height = float(com[2]) + WHEEL_RADIUS_M
    import math

    print(f"\npayload {payload_kg:.2f} kg at {tuple(pos)}")
    print(f"  chassis mass    {float(mass):.3f} kg")
    print(f"  CoM             ({com[0]:+.4f}, {com[1]:+.4f}, {com[2]:+.4f}) m")
    print(f"  CoM above floor {height:.4f} m")
    print(f"  equilibrium lean {math.degrees(math.atan2(float(com[0]), height)):+.1f} deg")


def main() -> None:
    args = parse_args()

    def spec_fn() -> mujoco.MjSpec:
        spec = get_spec()
        if args.color_by_group:
            apply_group_colors(spec)
        if args.payload_kg > 0.0:
            apply_payload(spec, args.payload_kg, tuple(args.payload_pos))
        return spec

    robot_cfg = get_javis_robot_cfg()
    robot_cfg.spec_fn = spec_fn

    scene_cfg = SceneCfg(
        terrain=TerrainEntityCfg(terrain_type="plane"),
        entities={"robot": robot_cfg},
        num_envs=1,
    )
    scene = Scene(scene_cfg, device="cpu")
    mujoco.viewer.launch(scene.compile())


if __name__ == "__main__":
    main()
