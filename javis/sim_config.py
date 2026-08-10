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

from dataclasses import dataclass, field

Range = tuple[float, float]


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
  """The ODrive velocity loop, and how much of it we admit we don't know.

  `use_pi_actuator` selects javis/mdp/actions.py's explicit PI loop over
  mjlab's BuiltinVelocityActuatorCfg. The builtin one is proportional only,
  while the real board runs P+I (vel_gain 0.25 N*m/(turn/s), vel_integrator_gain
  0.15). That integral term is exactly what absorbs a load change on hardware;
  leaving it out of sim teaches the policy to do that job itself, and then it
  double-compensates on the real robot. Set False to A/B against the builtin.
  """

  use_pi_actuator: bool = True

  # !! The gains currently on the hardware CANNOT balance this robot. !!
  #
  # The board runs vel_gain = 0.25 N*m/(turn/s) = 0.0398 N*m/(rad/s)
  # (scripts/motor_web_test.py, scripts/setup_odrive.py). Against a wheel
  # inertia of 0.0122 kg*m^2 that is a velocity-loop time constant of
  # J/kp = 307 ms, while the robot's own fall time constant is
  # sqrt(h/g) = 171 ms. A balance controller cannot stabilize a pendulum
  # through an inner loop slower than the pendulum -- the wheel is still on its
  # way to the commanded speed when the robot has already gone over. In sim it
  # shows up as a policy that saturates its command range and still falls.
  #
  # For the loop to be fast enough to be transparent to the balance controller
  # it needs tau ~= 20 ms, i.e. kp = J/tau = 0.61 N*m/(rad/s), which is ODrive
  # vel_gain = 3.83 N*m/(turn/s) -- roughly 15x what is configured now, and 10x
  # the top of the range scripts/tune_wheel_pid.py currently sweeps (that sweep
  # was done on a free-spinning wheel, where soft gains are perfectly fine).
  #
  # These defaults are therefore the values the hardware must be RETUNED to,
  # not the values it has. Logged as an action item in SIM2REAL.md sec 3.
  kp_nm_per_rad_s: float = 0.61
  # ODrive's own rule of thumb is vel_integrator_gain = 0.5 * bandwidth *
  # vel_gain; at a 10 Hz bandwidth that is 19 N*m/(turn/s)/s = 3.0 in per-radian
  # units. The integral term is what silently absorbs an unknown payload.
  ki_nm_per_rad: float = 3.0

  # What the board is actually set to today, for reference and for the A/B in
  # scripts/eval_payload_sweep.py --board-gains.
  kp_as_configured: float = 0.0398
  ki_as_configured: float = 0.0239

  # INPUT_MODE_VEL_RAMP on the real board (scripts/motor_web_test.py).
  vel_ramp_rad_s2: float = 25.0 * 2.0 * 3.141592653589793

  # Wide, because the retuned value is a target rather than a measurement: the
  # policy should not depend on hitting it exactly.
  gain_scale_easy: Range = (0.9, 1.1)
  gain_scale_hard: Range = (0.6, 1.6)

  # Command latency in control steps (20 ms each at 50 Hz): USB round trip plus
  # ROS2 hops. Unmeasured -- SIM2REAL.md sec 6 still has the control rate blank.
  latency_steps_easy: tuple[int, int] = (0, 1)
  latency_steps_hard: tuple[int, int] = (1, 3)


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
