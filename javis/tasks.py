"""Registers JAVIS's mjlab tasks so they show up in `list-envs`/`train`/`play`.

Discovered automatically via the "mjlab.tasks" entry point declared in
pyproject.toml (see mjlab/__init__.py:_import_registered_packages) once this
package is installed (`pip install -e .`) -- no manual import needed.
"""

from mjlab.tasks.registry import register_mjlab_task

from .balance_task import (
  javis_balance_flat_env_cfg,
  javis_balance_ppo_runner_cfg,
  javis_balance_rough_env_cfg,
)
from .velocity_task import javis_ppo_runner_cfg, javis_velocity_env_cfg


def register() -> None:
  # Baseline: fixed mass, flat ground, no payload. Kept as the control to
  # compare the payload tasks against -- do not fold it into them.
  register_mjlab_task(
    task_id="Javis-Velocity-Flat",
    env_cfg=javis_velocity_env_cfg(),
    play_env_cfg=javis_velocity_env_cfg(play=True),
    rl_cfg=javis_ppo_runner_cfg(),
  )

  # Randomized mass / center of mass / payload. Flat first for fast iteration,
  # then the same thing on slopes and rough ground.
  register_mjlab_task(
    task_id="Javis-Payload-Flat",
    env_cfg=javis_balance_flat_env_cfg(),
    play_env_cfg=javis_balance_flat_env_cfg(play=True),
    rl_cfg=javis_balance_ppo_runner_cfg("javis_payload_flat"),
  )
  register_mjlab_task(
    task_id="Javis-Payload-Rough",
    env_cfg=javis_balance_rough_env_cfg(),
    play_env_cfg=javis_balance_rough_env_cfg(play=True),
    rl_cfg=javis_balance_ppo_runner_cfg("javis_payload_rough"),
  )


register()
