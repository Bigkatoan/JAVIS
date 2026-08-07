#!/usr/bin/env python3
"""Interactively view the JAVIS rover in an mjlab Scene (flat ground plane,
no RL task involved -- just a sanity check that robot.urdf + robot_constants.py
still build into a valid, standing simulation).

Run after changing javis/robot.urdf or javis/robot_constants.py:
    venv/bin/python scripts/view_robot.py
"""
import mujoco.viewer

from javis.robot_constants import get_javis_robot_cfg

from mjlab.scene import Scene, SceneCfg
from mjlab.terrains import TerrainEntityCfg


def main() -> None:
    scene_cfg = SceneCfg(
        terrain=TerrainEntityCfg(terrain_type="plane"),
        entities={"robot": get_javis_robot_cfg()},
        num_envs=1,
    )
    scene = Scene(scene_cfg, device="cpu")
    model = scene.compile()
    mujoco.viewer.launch(model)


if __name__ == "__main__":
    main()
