"""Every domain-randomization number for the payload/balance task, in one place.

The RL task, the DR events and the evaluation sweep all read their ranges from
here, so replacing a guess with a measurement from SIM2REAL.md means editing one
file rather than hunting through task definitions.

Two ranges per quantity
-----------------------
Each randomized quantity has an `easy` range and a `hard` range. The curriculum
(javis/mdp/curriculums.py) keeps a difficulty `level` in [0, 1] and every event
linearly interpolates between the two:

    effective = easy + level * (hard - easy)

At level 0 the robot is close to its best-estimate build; at level 1 it spans
the full envelope the user asked for (chassis 3-15 kg, payload 0-10 kg, CoM
offsets to +-12 cm, mount heights to 0.6 m). Training at level 1 from step one
does not work -- a policy that has never balanced anything cannot learn to
balance 10 kg mounted 0.6 m up -- hence the ramp.

How the level is shared
-----------------------
The curriculum term writes the level; every event term reads it. They cannot
share a mutable object by reference, because the managers deep-copy each term
config *separately* -- each term would end up with its own private copy and the
events would never see the curriculum's updates. (That bug is easy to miss:
the curriculum still logs a rising level while the events keep sampling the
starting ranges.)

So `DifficultyState` holds only a string key, and the level itself lives in a
module-level registry. Copying the config copies the key, and every copy
therefore reads and writes the same entry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

Range = tuple[float, float]

# ODrive states velocities in turns/s; the controller math here works in rad/s.
_TWO_PI = 2.0 * math.pi


def lerp_range(easy: Range, hard: Range, level: float) -> Range:
  """Interpolate a (lo, hi) pair between its easy and hard extremes."""
  return (
    easy[0] + level * (hard[0] - easy[0]),
    easy[1] + level * (hard[1] - easy[1]),
  )


_LEVEL_REGISTRY: dict[str, float] = {}


class DifficultyState:
  """A handle on a curriculum level in [0, 1]. See module docstring.

  Deliberately not a dataclass holding the value: copies of this object must
  stay connected to each other, so the value lives in `_LEVEL_REGISTRY` and
  only the key travels with the copy.
  """

  def __init__(self, key: str, initial: float = 0.0) -> None:
    self.key = key
    _LEVEL_REGISTRY.setdefault(key, initial)

  @property
  def level(self) -> float:
    return _LEVEL_REGISTRY.get(self.key, 0.0)

  @level.setter
  def level(self, value: float) -> None:
    _LEVEL_REGISTRY[self.key] = value

  def reset(self, value: float) -> None:
    _LEVEL_REGISTRY[self.key] = value

  def clamp(self) -> None:
    self.level = min(1.0, max(0.0, self.level))

  def __repr__(self) -> str:
    return f"DifficultyState(key={self.key!r}, level={self.level:.3f})"


##
# Mass composition.
##


@dataclass(frozen=True)
class MassDomainCfg:
  """How the chassis mass is allowed to differ from javis.mass_model's nominal.

  Two independent mechanisms, because they do different jobs:

  - `group_scale` scatters the individual component masses. This is what moves
    the CENTER OF MASS around, since the groups sit at different places in the
    chassis. Ranges reflect how well each number is actually known: the battery
    and wheels are on a scale (+-10%), the printed parts are a one-spool
    estimate (+-50%), the Jetson and camera are catalog figures (+-20%), the
    MKS boards and IMU are guesses (+-40%).
  - `chassis_total_kg` then renormalizes every mesh-backed group so the TOTAL
    lands where we want. Without it, per-group scatter alone only spans roughly
    4-7 kg, nowhere near the 3-15 kg envelope requested. Applying it as a
    rescale (rather than an independent draw) keeps the internal proportions
    physically sensible at any total.
  """

  # group name -> multiplicative scale range on its nominal mass.
  group_scale_easy: dict[str, Range] = field(
    default_factory=lambda: {
      "battery": (0.95, 1.05),
      "printed": (0.85, 1.15),
      "jetson": (0.95, 1.05),
      "camera": (0.95, 1.05),
      "odrive": (0.90, 1.10),
      "imu": (0.90, 1.10),
      "hardware": (0.90, 1.10),
      "electronics_misc": (0.85, 1.15),
    }
  )
  group_scale_hard: dict[str, Range] = field(
    default_factory=lambda: {
      # Weighed on a scale 2026-08-06; +-10% covers scale error and cell drift.
      "battery": (0.90, 1.10),
      # One 1 kg spool, from memory. The user asked for +-50% here explicitly.
      "printed": (0.50, 1.50),
      # Catalog figures for a known part number.
      "jetson": (0.80, 1.20),
      "camera": (0.80, 1.20),
      # Never weighed; vendor listings only quote shipping weight.
      "odrive": (0.60, 1.40),
      "imu": (0.60, 1.40),
      # Steel density x CAD volume, cross-checked against a bearing catalog.
      "hardware": (0.70, 1.30),
      # Mixed copper/plastic at an assumed 3.0 g/cm^3; least-known density.
      "electronics_misc": (0.50, 1.50),
    }
  )

  # Chassis total AFTER group scatter, in kg. Nominal is ~5.27 kg.
  chassis_total_easy: Range = (4.5, 6.5)
  chassis_total_hard: Range = (3.0, 15.0)

  # Cable harness, heat-shrink, zip ties, tape -- real mass that is not in CAD
  # at all and that the user cannot weigh. Absolute kg, not a scale.
  wiring_kg_easy: Range = (0.15, 0.45)
  wiring_kg_hard: Range = (0.0, 1.0)
  # Where that lump sits, in the chassis link frame (m). Spread over the
  # interior volume: cables run everywhere.
  wiring_pos_easy: tuple[Range, Range, Range] = ((-0.02, 0.02), (-0.02, 0.02), (0.10, 0.20))
  wiring_pos_hard: tuple[Range, Range, Range] = ((-0.08, 0.08), (-0.06, 0.06), (0.05, 0.30))

  # Per-wheel, sampled INDEPENDENTLY for left and right. Two hub motors are
  # never identical, and the asymmetry is a real yaw disturbance.
  wheel_scale_easy: Range = (0.97, 1.03)
  wheel_scale_hard: Range = (0.90, 1.10)


##
# Payload.
##


@dataclass(frozen=True)
class PayloadCfg:
  """The cargo the robot carries: how heavy, mounted where, changing when."""

  mass_kg_easy: Range = (0.0, 1.0)
  mass_kg_hard: Range = (0.0, 10.0)

  # Mount position in the chassis link frame (m). z is measured from the wheel
  # axle, so z=0.6 is ~0.7 m above the ground.
  pos_x_easy: Range = (-0.02, 0.02)
  pos_x_hard: Range = (-0.12, 0.12)
  pos_y_easy: Range = (-0.02, 0.02)
  pos_y_hard: Range = (-0.12, 0.12)
  pos_z_easy: Range = (0.20, 0.32)
  pos_z_hard: Range = (0.05, 0.60)

  # Visual box half-extent, scaled with mass so a heavy load looks heavy. The
  # box has density=0 and no contact, so this is cosmetic -- the inertia comes
  # from the point mass in javis.mass_model.
  size_at_zero_kg: float = 0.03
  size_at_max_kg: float = 0.11
  max_kg_for_size: float = 10.0

  # Mid-episode load change ("nhat/tha vat"): every interval, this fraction of
  # environments resamples payload mass while keeping the mount pose fixed.
  change_enabled: bool = True
  change_interval_s: Range = (3.0, 8.0)
  change_prob_easy: float = 0.0
  change_prob_hard: float = 0.35


##
# Contact, drivetrain, terrain.
##


@dataclass(frozen=True)
class ContactCfg:
  """Wheel-ground friction. Independent per wheel: floors are not uniform and
  one wheel losing grip is a real failure mode this policy should survive."""

  friction_easy: Range = (0.8, 1.2)
  friction_hard: Range = (0.4, 1.6)


@dataclass(frozen=True)
class DrivetrainCfg:
  """The ODrive velocity loop.

  Gains are written in **ODrive native units** -- literally the numbers typed
  into `axis0.controller.config` -- so tuning the board to match the simulation
  is a copy, not a conversion. The per-radian values the controller math
  actually uses are derived properties at the bottom.

  `use_pi_actuator` selects javis/mdp/actions.py's explicit PI loop over
  mjlab's BuiltinVelocityActuatorCfg. The builtin one is proportional only,
  while the real board runs P+I. That integral term is exactly what absorbs a
  load change on hardware; leaving it out of sim teaches the policy to do that
  job itself, and then it double-compensates against the real board. Set False
  to A/B against the builtin.
  """

  use_pi_actuator: bool = True

  # !! These are the gains the hardware must be RETUNED TO, not what it has. !!
  #
  # The board currently runs vel_gain = 0.25, which against a 0.0122 kg*m^2
  # wheel is a velocity-loop time constant of 307 ms -- while the robot's own
  # fall time constant is sqrt(h/g) = 171 ms. No balance controller can
  # stabilize a pendulum through an inner loop slower than the pendulum: the
  # wheel is still on its way to the commanded speed when the robot has already
  # gone over. It is worse while driving, where the loop must also accelerate
  # 11 kg of robot (effective inertia 0.119 kg*m^2, so ~3 s at that gain).
  #
  # Chosen with scripts/tune_sim_gains.py, which sweeps candidates against both
  # the loaded and the unloaded wheel and scores them exactly the way
  # scripts/tune_wheel_pid.py scores real hardware. At these values the loop is
  # 4.9 ms unloaded and 48 ms driving, comfortably inside the 171 ms the balance
  # problem allows, so the wheel behaves as the near-ideal velocity source the
  # policy assumes.
  #
  # What this means electrically, before putting it on hardware: with
  # Kt = 0.207 N*m/A a vel_gain of 15.7 saturates the 15 A limit at a velocity
  # error of only 1.2 rad/s. Large errors therefore command full torque, which
  # is right for balance recovery -- but encoder noise is amplified ~63x
  # relative to today, so expect to back off if the wheels buzz at rest.
  # Raise it in steps on a stand, not in one jump on the floor.
  vel_gain: float = 15.0
  """Proportional gain, N*m/(turn/s). ODrive `controller.config.vel_gain`.

  Round number on purpose: this is a target to tune the hardware towards, and
  the sweep is flat enough between 9 and 19 that a memorable value costs
  nothing. Gives kp*dt/J = 0.49 at the 2.5 ms physics timestep -- a 4x
  stability margin."""
  vel_integrator_gain: float = 75.0
  """Integral gain, N*m/(turn/s)/s. ODrive
  `controller.config.vel_integrator_gain`. ODrive's own rule of thumb is
  0.5 * bandwidth * vel_gain; this is that at a 10 Hz bandwidth."""
  vel_ramp_rate: float = 25.0
  """Slew limit, turn/s^2. ODrive `controller.config.vel_ramp_rate`, used by
  INPUT_MODE_VEL_RAMP (see scripts/motor_web_test.py)."""

  # What the board is set to today. Kept so the gap stays visible. To check
  # whether a policy survives at whatever gain the hardware actually reaches:
  #   scripts/eval_payload_sweep.py --vel-gain 6 --vel-integrator-gain 30
  vel_gain_as_configured: float = 0.25
  vel_integrator_gain_as_configured: float = 0.15

  # Wide, because the retuned value is a target rather than a measurement: the
  # policy must not depend on the board landing exactly on it.
  gain_scale_easy: Range = (0.9, 1.1)
  gain_scale_hard: Range = (0.6, 1.6)

  # Command latency in control steps (20 ms each at 50 Hz): USB round trip plus
  # ROS2 hops. Unmeasured -- SIM2REAL.md sec 6 still has the control rate blank.
  latency_steps_easy: tuple[int, int] = (0, 1)
  latency_steps_hard: tuple[int, int] = (1, 3)

  @property
  def kp_nm_per_rad_s(self) -> float:
    """vel_gain converted from ODrive's per-turn units to per-radian."""
    return self.vel_gain / _TWO_PI

  @property
  def ki_nm_per_rad(self) -> float:
    return self.vel_integrator_gain / _TWO_PI

  @property
  def vel_ramp_rad_s2(self) -> float:
    return self.vel_ramp_rate * _TWO_PI

  def stability_alpha(self, timestep: float, wheel_inertia: float = 0.0122) -> float:
    """Forward-Euler stability number for the PI loop, `kp * dt / J`.

    The loop computes torque explicitly once per physics step, so its feedback
    path is forward Euler: it diverges above 2 and rings below that. `J` is the
    *unloaded* wheel, the smallest inertia the loop can ever see -- which is
    exactly the case when a wheel unloads or slips, i.e. when the robot is
    already in trouble and the loop most needs to behave.

    The real board has no such limit (its loop runs at 8 kHz); this is purely
    a constraint on how finely the simulation must be integrated to represent
    the gain honestly.
    """
    return self.kp_nm_per_rad_s * timestep / wheel_inertia


@dataclass(frozen=True)
class TerrainMixCfg:
  """Sub-terrain proportions for the rough variant of the task.

  Deliberately no stairs, stepping stones or narrow beams: an 8-inch wheel on a
  two-point contact base cannot climb a step, so including them would only add
  unlearnable episodes. Slopes and mild roughness are what an indoor robot
  actually meets, and a slope is also the interesting confuser -- it tilts the
  gravity vector exactly the way an offset payload does, so the policy has to
  tell them apart from the dynamics rather than from projected_gravity alone.
  """

  flat_proportion: float = 0.40
  slope_proportion: float = 0.20
  inverted_slope_proportion: float = 0.15
  rough_proportion: float = 0.25

  # ~0 to 10 degrees (tan 10deg = 0.176).
  slope_range: Range = (0.0, 0.18)
  # Peak-to-peak height noise, m.
  rough_noise_range: Range = (0.0, 0.02)
  rough_noise_step: float = 0.005


##
# Feasibility.
##


@dataclass(frozen=True)
class FeasibilityCfg:
  """Reject sampled configurations the hardware physically cannot balance.

  A wheeled inverted pendulum arrests a lean by driving its contact point out
  from under the CoM, with ground force capped at F = 2*tau_max/r = 63.4 N. The
  steepest lean it can hold is atan(F / (M g)); standing still on a slope alpha
  already spends M g sin(alpha) of that budget. Above roughly 25 kg on a 10-deg
  slope there is nothing left, and the episode is unwinnable no matter what the
  policy does.

  The user chose the extreme mass envelope knowing this. Rather than narrow the
  envelope, we keep it and resample the configurations inside it that are
  provably impossible, so gradient signal is not spent on them. Set
  `enabled=False` to train on the raw envelope and see the failures.
  """

  enabled: bool = True
  # Lean margin the robot must retain beyond just holding station, in degrees.
  margin_deg: float = 5.0
  max_resample_attempts: int = 8


##
# Top level.
##


@dataclass(frozen=True)
class JavisDomainCfg:
  mass: MassDomainCfg = field(default_factory=MassDomainCfg)
  payload: PayloadCfg = field(default_factory=PayloadCfg)
  contact: ContactCfg = field(default_factory=ContactCfg)
  drivetrain: DrivetrainCfg = field(default_factory=DrivetrainCfg)
  terrain: TerrainMixCfg = field(default_factory=TerrainMixCfg)
  feasibility: FeasibilityCfg = field(default_factory=FeasibilityCfg)


@dataclass(frozen=True)
class CurriculumCfg:
  """How fast the difficulty ramp opens up.

  Driven by mean episode length rather than reward: reward mixes tracking and
  survival and drifts as the ranges widen, whereas "how long does it stay up"
  is comparable across difficulty levels.
  """

  enabled: bool = True
  start_level: float = 0.0
  # Fraction of the maximum episode length that counts as "coping".
  promote_above: float = 0.80
  demote_below: float = 0.45
  # Level change per curriculum update (one per env step, so keep it small).
  step_up: float = 2.0e-4
  step_down: float = 4.0e-4


DEFAULT_DOMAIN = JavisDomainCfg()
DEFAULT_CURRICULUM = CurriculumCfg()
