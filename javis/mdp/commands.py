"""Patches a real mjlab bug that crashes the web viewer for this robot.

`UniformVelocityCommand.create_gui` builds a "Max <axis>" joystick slider per
command axis with a hard-coded `min=0.1`, then asserts
`min <= initial_value <= max` using the axis's own configured range maximum as
the initial value (mjlab 1.5.3, `tasks/velocity/mdp/velocity_command.py`).

That is fine for a legged robot, where every axis has some nonzero range. It
is not fine here: this chassis is a differential-drive base, physically unable
to strafe, so `lin_vel_y` is deliberately `(0.0, 0.0)` in both
`javis/velocity_task.py` and `javis/balance_task.py`. The slider's initial
value is then 0.0, which fails `0.0 >= 0.1`, and `play --viewer viser` crashes
before the window ever opens -- reported to a real user running
`scripts/watch_training.sh`.

`JavisVelocityCommand` reimplements `create_gui` with one clamp (the "Max"
slider's floor becomes max(range, 0.1), matching mjlab's own hard-coded
minimum) and defers everything else -- resampling, `compute`, `debug_vis`,
the 3D command arrows -- to the base class unchanged. The clamp only affects
where that one GUI slider starts; it never touches `self.cfg.ranges`, so what
gets sampled during training and eval is exactly what was configured.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from mjlab.tasks.velocity.mdp.velocity_command import (
  UniformVelocityCommand,
  UniformVelocityCommandCfg,
)

if TYPE_CHECKING:
  import viser

  from mjlab.envs import ManagerBasedRlEnv


class JavisVelocityCommand(UniformVelocityCommand):
  def create_gui(
    self,
    name: str,
    server: "viser.ViserServer",
    get_env_idx: Callable[[], int],
    on_change: Callable[[], None] | None = None,
    request_action: Callable[[str, Any], None] | None = None,
  ) -> None:
    """Verbatim copy of the base implementation, minus the crash.

    See the module docstring. `on_change` and `request_action` are accepted
    for interface compatibility but unused, matching the base class (which
    also ignores them here -- only its position/pose GUI variants use them).
    """
    del on_change, request_action
    from viser import Icon

    ranges = self.cfg.ranges
    axes = [
      ("lin_vel_x", ranges.lin_vel_x[1]),
      ("lin_vel_y", ranges.lin_vel_y[1]),
      ("ang_vel_z", ranges.ang_vel_z[1]),
    ]
    sliders: list = []

    with server.gui.add_folder(name.capitalize()):
      enabled = server.gui.add_checkbox("Enable", initial_value=False)

      for label, max_val in axes:
        # The one changed line: mjlab hard-codes min=0.1 here and asserts
        # min <= initial_value <= max against the raw (possibly zero)
        # configured max. Clamping the initial value to that same floor
        # satisfies the assertion; the joystick slider two lines down still
        # has a real min/max of (-max_val, max_val) = (0, 0), so it cannot
        # actually command a nonzero lin_vel_y regardless of this display
        # floor -- physically correct for a non-holonomic base.
        max_input = server.gui.add_slider(
          f"Max {label}",
          initial_value=max(max_val, 0.1),
          step=0.1,
          min=0.1,
          max=10.0,
        )
        slider = server.gui.add_slider(
          label,
          min=-max_val,
          max=max_val,
          step=0.05,
          initial_value=0.0,
        )

        @max_input.on_update
        def _(_ev, _s=slider, _m=max_input) -> None:
          _s.min = -_m.value
          _s.max = _m.value

        sliders.append(slider)

      zero_btn = server.gui.add_button("Zero", icon=Icon.SQUARE_X)

      @zero_btn.on_click
      def _(_) -> None:
        for s in sliders:
          s.value = 0.0

    self._joystick_enabled = enabled
    self._joystick_sliders = sliders
    self._joystick_get_env_idx = get_env_idx


@dataclass(kw_only=True)
class JavisVelocityCommandCfg(UniformVelocityCommandCfg):
  """Drop-in replacement for `UniformVelocityCommandCfg` -- same fields,
  same `Ranges` nested type (inherited unchanged), only `build()` differs."""

  def build(self, env: "ManagerBasedRlEnv") -> JavisVelocityCommand:
    return JavisVelocityCommand(self, env)
