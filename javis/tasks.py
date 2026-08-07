"""Registers JAVIS's mjlab tasks so they show up in `list-envs`/`train`/`play`.

Discovered automatically via the "mjlab.tasks" entry point declared in
pyproject.toml (see mjlab/__init__.py:_import_registered_packages) once this
package is installed (`pip install -e .`) -- no manual import needed.
"""

from mjlab.tasks.registry import register_mjlab_task

from .velocity_task import javis_ppo_runner_cfg, javis_velocity_env_cfg


def register() -> None:
  register_mjlab_task(
    task_id="Javis-Velocity-Flat",
    env_cfg=javis_velocity_env_cfg(),
    play_env_cfg=javis_velocity_env_cfg(play=True),
    rl_cfg=javis_ppo_runner_cfg(),
  )


register()
