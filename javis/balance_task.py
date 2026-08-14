"""Balance + velocity tracking under an unknown, shifting load.

This is `javis/velocity_task.py` taken to the case the robot will actually meet:
the same twist-tracking objective, but the mass, the center of mass and the
payload all change from episode to episode -- and sometimes mid-episode --
without the policy being told. It has to work it out from how the robot
responds, which is why the actor sees a stacked observation history rather than
a single frame.

Registered as two tasks:

  Javis-Payload-Flat   flat ground. Fast iteration, isolates the load problem.
  Javis-Payload-Rough  slopes and mild roughness. The real target.

Both keep `Javis-Velocity-Flat` untouched as a baseline to compare against.

Design notes live next to the code they explain:
  javis/mass_model.py   -- why mass is modelled per component group
  javis/mdp/events.py   -- why the DR is written by hand and not composed from
                           mjlab's dr.body_* helpers
  javis/mdp/actions.py  -- why the ODrive integral term has to be simulated
  javis/mdp/rewards.py  -- why rewarding "upright" is wrong once the CoM moves
  javis/sim_config.py   -- every randomization range, in one editable place
"""

from __future__ import annotations

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointVelocityActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity import mdp as velocity_mdp
from mjlab.terrains import (
  BoxFlatTerrainCfg,
  HfPyramidSlopedTerrainCfg,
  HfRandomUniformTerrainCfg,
  TerrainEntityCfg,
  TerrainGeneratorCfg,
)
from mjlab.utils.noise import NoiseModelWithAdditiveBiasCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from .mdp import actions as javis_actions
from .mdp import commands as javis_commands
from .mdp import curriculums as javis_curriculums
from .mdp import events as javis_events
from .mdp import observations as javis_obs
from .mdp import rewards as javis_rewards
from .robot_constants import (
  CHASSIS_BODY_NAME,
  CHASSIS_COLLISION_GEOM,
  WHEEL_COLLISION_GEOMS,
  WHEEL_EFFORT_LIMIT_NM,
  WHEEL_JOINTS,
  get_javis_robot_cfg,
)
from .sim_config import (
  CurriculumCfg,
  DifficultyState,
  JavisDomainCfg,
  lerp_range,
)

_ROBOT = SceneEntityCfg("robot")
_CHASSIS = SceneEntityCfg("robot", body_names=(CHASSIS_BODY_NAME,))

# Control rate. Real hardware ceiling, from measurement/recollection rather
# than the original 50 Hz placeholder (SIM2REAL.md sec 6 -- still not a
# rigorous benchmark, update this the moment a real number exists):
# USB/ROS2 round trip tops out around 100-150 Hz. 100 is the conservative end
# of that range on purpose -- a policy trained at the SLOWER rate (sparser
# corrections between observations) is the harder case, and generalizes
# forward to a real loop running faster than it trained at more safely than
# the other direction would.
CONTROL_HZ = 100.0

# 400 Hz. Not derived from CONTROL_HZ -- this is a separate, lower-level
# requirement: the ODrive PI loop in javis/mdp/actions.py computes torque
# explicitly once per physics step, so its stiffness is capped by this
# timestep (DrivetrainCfg.stability_alpha). At the coarser 200 Hz
# javis/velocity_task.py still uses, the retuned drivetrain gain would ring;
# see SIM2REAL.md sec 3. decimation is what reconciles the two rates.
PHYSICS_TIMESTEP_S = 0.0025
_DECIMATION_FLOAT = 1.0 / (CONTROL_HZ * PHYSICS_TIMESTEP_S)
assert _DECIMATION_FLOAT == round(_DECIMATION_FLOAT), (
  f"CONTROL_HZ={CONTROL_HZ} does not divide evenly into PHYSICS_TIMESTEP_S="
  f"{PHYSICS_TIMESTEP_S} (decimation would be {_DECIMATION_FLOAT}, not an "
  "integer) -- pick a CONTROL_HZ that does, or decimation will silently "
  "round to a slightly different actual control rate than CONTROL_HZ claims."
)
DECIMATION = round(_DECIMATION_FLOAT)

# ~0.24 s of physical time, scaled to whatever CONTROL_HZ is -- long enough to
# contain a full response to an action (what lets the policy infer inertia
# from the response history) and short enough to stay Jetson-sized. Frame
# count follows the control rate so the physical time window stays fixed:
# 12 frames at 50 Hz and 24 at 100 Hz are the same ~240 ms of history.
OBS_HISTORY_SECONDS = 0.24
OBS_HISTORY_LENGTH = round(OBS_HISTORY_SECONDS * CONTROL_HZ)

CONTACT_SENSOR_NAME = "chassis_contact"

# False: skip the difficulty ramp entirely and train against the full "hard"
# envelope from iteration 0 (sim_config.py's *_hard ranges, terrain always at
# max slope/roughness -- see _terrain_cfg). User's call, made with the ramp's
# purpose stated plainly first: a policy that has never balanced anything may
# get little or no learnable signal against 10 kg mounted 0.6 m up on day one,
# since most envs could fail within the first few steps of every episode. If
# that's what happens (reward and episode length both flat near their floor
# for a long stretch, not just noisy), it's a real signal to flip this back to
# True rather than a smoke-test of "let it run 5000 more iterations." One flag
# either direction -- nothing else needs touching to switch back.
CURRICULUM_ENABLED = False


# mjlab's HfPyramidSlopedTerrainCfg/HfRandomUniformTerrainCfg default to
# horizontal_scale=0.1 (10cm hfield cells). The single-convex-hull
# `chassis_collision` geom (robot_constants.py -- 343 chassis parts merged
# into one hull, see README's "Simulation notes" for why) spans enough
# ground cells at once during a tip-over onto sloped/rough terrain that
# MuJoCo's own hfield-vs-geom near-phase collision list overflows its fixed
# 50-pair cap ("height field collision overflow ... please adjust
# resolution: decrease the number of hfield rows/cols or modify size of
# colliding geom" -- MuJoCo's own suggested fix, applied literally here).
# Confirmed via a real 512-env/30k-step run: 1105 warnings at the default
# 0.1 scale, 0 at 0.25 (rerun below, same steps/envs). A proper fix would
# decompose the hull into smaller convex pieces (already a known gap, see
# README); this is the cheap terrain-side mitigation until that happens.
HF_HORIZONTAL_SCALE = 0.25


def _terrain_cfg(domain: JavisDomainCfg, rough: bool) -> TerrainEntityCfg:
  """Flat plane, or a mix of slopes and mild roughness.

  Deliberately no stairs, stepping stones or narrow beams: an 8-inch wheel on a
  two-point contact base cannot climb a step, so those would only contribute
  episodes nothing can learn from. Slopes are the interesting case anyway --
  they tilt the gravity vector exactly the way an offset payload does, so the
  policy has to separate the two from the dynamics rather than reading
  `projected_gravity` and guessing.
  """
  if not rough:
    return TerrainEntityCfg(terrain_type="plane")

  t = domain.terrain
  return TerrainEntityCfg(
    terrain_type="generator",
    terrain_generator=TerrainGeneratorCfg(
      size=(8.0, 8.0),
      border_width=4.0,
      num_rows=8,
      num_cols=8,
      curriculum=True,
      # Pinned to the hard end, not (0.0, 1.0): CURRICULUM_ENABLED=False below
      # means nothing ever promotes a row from easy to hard, so instead every
      # row is generated AT the hardest configured slope/roughness regardless
      # of which one an env sits on. Row position becomes irrelevant this way
      # (no need to also fight with max_init_terrain_level to pin envs to a
      # specific row -- every row is now identical). Restore (0.0, 1.0) if
      # CURRICULUM_ENABLED goes back to True.
      difficulty_range=(1.0, 1.0),
      sub_terrains={
        "flat": BoxFlatTerrainCfg(proportion=t.flat_proportion),
        "slope": HfPyramidSlopedTerrainCfg(
          proportion=t.slope_proportion,
          slope_range=t.slope_range,
          horizontal_scale=HF_HORIZONTAL_SCALE,
        ),
        "slope_inv": HfPyramidSlopedTerrainCfg(
          proportion=t.inverted_slope_proportion,
          slope_range=t.slope_range,
          inverted=True,
          horizontal_scale=HF_HORIZONTAL_SCALE,
        ),
        "rough": HfRandomUniformTerrainCfg(
          proportion=t.rough_proportion,
          noise_range=t.rough_noise_range,
          noise_step=t.rough_noise_step,
          scale_with_difficulty=True,
          horizontal_scale=HF_HORIZONTAL_SCALE,
        ),
      },
    ),
  )


def _observations(domain: JavisDomainCfg) -> dict[str, ObservationGroupCfg]:
  # Exactly what the real robot can publish: BMX160 gyro, cuVSLAM-derived
  # linear velocity, gravity direction, ODrive encoder velocities, the last
  # command sent, and the twist being asked for. No joint_pos -- a continuous
  # wheel's absolute angle says nothing about posture.
  #
  # delay_*_lag models the USB + ROS2 hop between sensor and policy. Unmeasured
  # (SIM2REAL.md sec 6 still has the control rate blank), so it is randomized
  # rather than assumed to be zero.
  #
  # IMU-sourced terms carry TWO noise sources stacked, not one:
  #   - per-step white noise (unchanged from before) -- the sensor's own
  #     instantaneous noise floor.
  #   - a bias resampled once per episode and held constant for its duration
  #     (NoiseModelWithAdditiveBiasCfg) -- a real MEMS IMU's zero-rate offset
  #     drifts boot to boot but is roughly constant within one power cycle.
  # This matters beyond realism: pure i.i.d. per-step noise averages out of
  # anything the policy integrates over time, so a policy trained only against
  # it can implicitly lean on that cancellation. A persistent bias cannot
  # cancel the same way -- it is a genuine, unremovable estimation error the
  # policy has to be robust to, which is a materially harder and more honest
  # training signal. Magnitudes below are order-of-magnitude placeholders (no
  # BMX160 datasheet noise-density figure has been pulled in), sized as a
  # fraction of the existing per-step noise rather than derived from a
  # spec -- replace once SIM2REAL.md sec 5 has a real number.
  def imu_noise(step_range: float, bias_range: float) -> NoiseModelWithAdditiveBiasCfg:
    return NoiseModelWithAdditiveBiasCfg(
      noise_cfg=Unoise(n_min=-step_range, n_max=step_range),
      bias_noise_cfg=Unoise(n_min=-bias_range, n_max=bias_range),
    )

  actor_terms = {
    "base_lin_vel": ObservationTermCfg(
      func=envs_mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
      noise=imu_noise(0.1, 0.03),
      delay_min_lag=1,
      delay_max_lag=3,
    ),
    "base_ang_vel": ObservationTermCfg(
      func=envs_mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=imu_noise(0.1, 0.03),
      delay_min_lag=1,
      delay_max_lag=3,
    ),
    "projected_gravity": ObservationTermCfg(
      func=envs_mdp.projected_gravity,
      noise=imu_noise(0.03, 0.015),
      delay_min_lag=1,
      delay_max_lag=3,
    ),
    "wheel_vel": ObservationTermCfg(
      func=envs_mdp.joint_vel_rel,
      noise=Unoise(n_min=-0.5, n_max=0.5),
      delay_min_lag=1,
      delay_max_lag=3,
    ),
    "actions": ObservationTermCfg(func=envs_mdp.last_action),
    "command": ObservationTermCfg(
      func=envs_mdp.generated_commands,
      params={"command_name": "twist"},
    ),
  }

  # The critic is told what the robot is made of. It only exists during
  # training, so this costs nothing at deployment and stops the value function
  # from having to guess why two identical-looking states earn different
  # returns. See javis/mdp/observations.py.
  critic_terms = dict(actor_terms)
  critic_terms["load_state"] = ObservationTermCfg(func=javis_obs.privileged_load_state)

  return {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      enable_corruption=True,
      history_length=OBS_HISTORY_LENGTH,
      flatten_history_dim=True,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      history_length=OBS_HISTORY_LENGTH,
      flatten_history_dim=True,
    ),
  }


def _events(
  domain: JavisDomainCfg, difficulty: DifficultyState, rough: bool
) -> dict[str, EventTermCfg]:
  max_slope_rad = math.atan(domain.terrain.slope_range[1]) if rough else 0.0

  events: dict[str, EventTermCfg] = {
    # Rebuild the robot: component masses, harness, payload, mount position.
    "randomize_load": EventTermCfg(
      func=javis_events.randomize_load,
      mode="reset",
      params={
        "cfg": domain,
        "difficulty": difficulty,
        "max_slope_rad": max_slope_rad,
      },
    ),
    "randomize_wheel_masses": EventTermCfg(
      func=javis_events.randomize_wheel_masses,
      mode="reset",
      params={"cfg": domain, "difficulty": difficulty},
    ),
    "reset_base": EventTermCfg(
      func=envs_mdp.reset_root_state_uniform,
      mode="reset",
      params={
        # roll/pitch widened from +-0.2 to +-0.35 rad (~20 deg): every episode
        # starts already leaning, not just occasionally. Still comfortably
        # short of the 60 deg fall_over limit, so it stays a recoverable (if
        # harder) start rather than a start inside the termination band.
        "pose_range": {
          "x": (-0.5, 0.5),
          "y": (-0.5, 0.5),
          "z": (0.0, 0.02),
          "roll": (-0.35, 0.35),
          "pitch": (-0.35, 0.35),
          "yaw": (-math.pi, math.pi),
        },
        # Previously {} -- every episode started from exactly zero velocity,
        # which is not a case this robot will ever actually be in: real
        # resets happen after a stumble, a hand-off, or a push, not from rest.
        # Reusing push_robot's own x/y/roll/pitch magnitudes below rather than
        # inventing new numbers: "start already having just been shoved" and
        # "get shoved mid-episode" are the same physical event, so the same
        # scale applies, and it's a magnitude training already tolerates via
        # push_robot. z and yaw are new axes (push_robot doesn't touch them):
        # a small downward settling velocity, and a modest initial spin.
        "velocity_range": {
          "x": (-0.3, 0.3),
          "y": (-0.3, 0.3),
          "z": (-0.15, 0.05),
          "roll": (-0.3, 0.3),
          "pitch": (-0.3, 0.3),
          "yaw": (-0.3, 0.3),
        },
      },
    ),
    "reset_wheels": EventTermCfg(
      func=envs_mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (0.0, 0.0),
        "velocity_range": (-0.5, 0.5),
        "asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINTS),
      },
    ),
    # Per-wheel, not shared: one wheel on a rug and one on tile is a real
    # situation, and it produces a yaw disturbance that a shared value hides.
    "wheel_friction": EventTermCfg(
      mode="reset",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=WHEEL_COLLISION_GEOMS),
        "operation": "abs",
        "ranges": lerp_range(
          domain.contact.friction_easy, domain.contact.friction_hard, 1.0
        ),
        "shared_random": False,
      },
    ),
    # Physical wheel drag, as opposed to the control gain that
    # scripts/calibrate_actuator.py fitted. Never measured, so the range starts
    # at zero and reaches roughly that fitted figure.
    "wheel_drag": EventTermCfg(
      mode="reset",
      func=dr.joint_damping,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINTS),
        "operation": "abs",
        "ranges": (0.0, 0.03),
      },
    ),
    "push_robot": EventTermCfg(
      func=envs_mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(2.0, 5.0),
      params={
        "velocity_range": {
          "x": (-0.3, 0.3),
          "y": (-0.3, 0.3),
          "roll": (-0.3, 0.3),
          "pitch": (-0.3, 0.3),
        },
      },
    ),
  }

  if domain.payload.change_enabled:
    # The "picks something up while driving" case. Expensive -- it forces a
    # full set_const recompute -- so it fires on an interval for a subset of
    # environments rather than every step.
    events["payload_change"] = EventTermCfg(
      func=javis_events.change_payload_midepisode,
      mode="interval",
      interval_range_s=domain.payload.change_interval_s,
      params={"cfg": domain, "difficulty": difficulty},
    )

  return events


def _rewards(use_pi: bool) -> dict[str, RewardTermCfg]:
  rewards = {
    "track_linear_velocity": RewardTermCfg(
      func=velocity_mdp.track_linear_velocity,
      weight=1.5,
      params={"command_name": "twist", "std": math.sqrt(0.25)},
    ),
    "track_angular_velocity": RewardTermCfg(
      func=velocity_mdp.track_angular_velocity,
      weight=1.0,
      params={"command_name": "twist", "std": math.sqrt(0.5)},
    ),
    # Weak and wide, unlike velocity_task.py's 3.0 / sqrt(0.15). With the CoM
    # up to 0.12 m off-axis the balance point is a ~22 deg lean, and a strong
    # upright term would be penalizing the posture that keeps the robot alive.
    "upright": RewardTermCfg(
      func=velocity_mdp.upright,
      weight=1.0,
      params={"std": math.sqrt(0.5), "asset_cfg": _CHASSIS},
    ),
    "is_alive": RewardTermCfg(func=envs_mdp.is_alive, weight=0.5),
    "action_rate_l2": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.05),
    # Reward holding a lean steady rather than holding it vertical.
    "pitch_rate_l2": RewardTermCfg(
      func=javis_rewards.pitch_rate_l2, weight=-0.02, params={"asset_cfg": _CHASSIS}
    ),
    "wheel_slip": RewardTermCfg(func=javis_rewards.wheel_slip, weight=-0.05),
  }
  if use_pi:
    # Only meaningful with the PI actuator, which is the only path that exposes
    # the torque it actually commanded.
    rewards["torque_saturation"] = RewardTermCfg(
      func=javis_rewards.torque_saturation,
      weight=-0.02,
      params={"action_name": "wheel_vel"},
    )
  return rewards


def _make_env_cfg(rough: bool = False, play: bool = False) -> ManagerBasedRlEnvCfg:
  domain = JavisDomainCfg()
  # enabled=False + start_level=1.0 pins every DR range (mass/CoM/payload;
  # terrain is pinned separately, see _terrain_cfg) at its "hard" bound from
  # iteration 0 and disables the promote/demote logic in
  # javis/mdp/curriculums.py entirely -- see CURRICULUM_ENABLED above.
  curriculum_cfg = (
    CurriculumCfg() if CURRICULUM_ENABLED else CurriculumCfg(enabled=False, start_level=1.0)
  )
  # One registry key per task variant, so flat and rough ramp independently.
  # Play mode pins the level at 1.0 regardless: when you are looking at a
  # trained policy you want the full envelope, not wherever training stopped.
  difficulty = DifficultyState(
    key=f"javis_payload_{'rough' if rough else 'flat'}{'_play' if play else ''}",
    initial=1.0 if play else curriculum_cfg.start_level,
  )
  if play:
    difficulty.reset(1.0)
  use_pi = domain.drivetrain.use_pi_actuator

  actions: dict[str, ActionTermCfg]
  if use_pi:
    actions = {
      "wheel_vel": javis_actions.OdriveVelocityActionCfg(
        entity_name="robot",
        actuator_names=WHEEL_JOINTS,
        scale=5.0,
        drivetrain=domain.drivetrain,
        difficulty=difficulty,
        effort_limit=WHEEL_EFFORT_LIMIT_NM,
      )
    }
  else:
    actions = {
      "wheel_vel": JointVelocityActionCfg(
        entity_name="robot",
        actuator_names=WHEEL_JOINTS,
        scale=5.0,
        use_default_offset=False,
      )
    }

  commands: dict[str, CommandTermCfg] = {
    # JavisVelocityCommandCfg, not mjlab's UniformVelocityCommandCfg directly:
    # same term, patched to not crash the Viser web viewer's joystick GUI when
    # an axis's range is zero-width -- see javis/mdp/commands.py.
    "twist": javis_commands.JavisVelocityCommandCfg(
      entity_name="robot",
      resampling_time_range=(3.0, 8.0),
      rel_standing_envs=0.2,
      heading_command=False,
      debug_vis=True,
      ranges=javis_commands.JavisVelocityCommandCfg.Ranges(
        lin_vel_x=(-0.5, 0.5),
        lin_vel_y=(0.0, 0.0),  # differential drive cannot strafe
        ang_vel_z=(-1.0, 1.0),
      ),
    )
  }

  terminations = {
    "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
    # 60 deg, not velocity_task.py's 45: a large lean can be the correct
    # equilibrium here. The contact test below is what actually defines falling.
    "fell_over": TerminationTermCfg(
      func=envs_mdp.bad_orientation,
      params={"limit_angle": math.radians(60.0)},
    ),
    "chassis_down": TerminationTermCfg(
      func=javis_rewards.chassis_grounded,
      params={"sensor_name": CONTACT_SENSOR_NAME},
    ),
  }

  curriculum: dict[str, CurriculumTermCfg] = {
    "payload_difficulty": CurriculumTermCfg(
      func=javis_curriculums.payload_difficulty,
      params={"difficulty": difficulty, "cfg": curriculum_cfg},
    ),
    "infeasible_frac": CurriculumTermCfg(func=javis_curriculums.infeasible_fraction),
    "total_mass_kg": CurriculumTermCfg(func=javis_curriculums.current_total_mass),
  }
  # terrain_levels moves individual envs between easier/harder terrain rows.
  # With CURRICULUM_ENABLED=False every row is already pinned to max
  # difficulty (_terrain_cfg), so which row an env sits on no longer matters
  # -- the term would just spend cycles computing promotions with no effect.
  if rough and CURRICULUM_ENABLED:
    curriculum["terrain_levels"] = CurriculumTermCfg(
      func=velocity_mdp.terrain_levels_vel,
      params={"command_name": "twist"},
    )

  chassis_contact = ContactSensorCfg(
    name=CONTACT_SENSOR_NAME,
    primary=ContactMatch(mode="geom", pattern=CHASSIS_COLLISION_GEOM, entity="robot"),
    fields=("found",),
    reduce="netforce",
    num_slots=1,
  )

  cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=_terrain_cfg(domain, rough),
      entities={"robot": get_javis_robot_cfg(torque_actuators=use_pi)},
      sensors=(chassis_contact,),
      num_envs=1 if play else 4096,
      env_spacing=2.0,
    ),
    observations=_observations(domain),
    actions=actions,
    commands=commands,
    events=_events(domain, difficulty, rough),
    rewards=_rewards(use_pi),
    terminations=terminations,
    curriculum=curriculum,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name=CHASSIS_BODY_NAME,
      distance=1.8,
      elevation=-20.0,
      azimuth=90.0,
    ),
    # See CONTROL_HZ / PHYSICS_TIMESTEP_S above for the reasoning: 400 Hz
    # physics (required for the retuned drivetrain gain to integrate cleanly)
    # decimated down to 100 Hz control (the real hardware's measured ceiling).
    sim=SimulationCfg(mujoco=MujocoCfg(timestep=PHYSICS_TIMESTEP_S)),
    decimation=DECIMATION,
    episode_length_s=20.0,
  )

  if play:
    cfg.episode_length_s = 1e10
    cfg.observations["actor"].enable_corruption = False
  return cfg


def javis_balance_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return _make_env_cfg(rough=False, play=play)


def javis_balance_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return _make_env_cfg(rough=True, play=play)


def javis_balance_ppo_runner_cfg(experiment: str) -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      # Wider than velocity_task.py's (128, 128, 64): the observation is 12x
      # longer here, and inferring inertia from a response history is a harder
      # function than reacting to one frame.
      hidden_dims=(256, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(256, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name=experiment,
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=3000,
  )
