"""Velocity-tracking + balance task for the JAVIS rover.

Framed the same way mjlab/Isaac Lab frame legged locomotion (a manager-based
RL env: a command-conditioned policy tracking (lin_vel_x, ang_vel_z) while
staying upright), but for a 2-actuator differential-drive base instead of a
12+ joint legged robot. Structurally this is closer to mjlab's own
`tasks/cartpole` (single balance problem, no feet/contact sensors) than to
`tasks/velocity` (written for legged robots), but it reuses `tasks/velocity`'s
generic, non-leg-specific pieces directly: `UniformVelocityCommandCfg`,
`track_linear_velocity`, `track_angular_velocity`, and `upright` are all
already robot-agnostic (see mjlab/tasks/velocity/mdp/rewards.py) -- the
leg-specific parts of that task (feet contact/air-time/pose rewards) are
simply left out here rather than stripped from a copy.

Why velocity actions, not torque actions: a wheeled-balance problem (Segway-
style) is normally solved in sim with direct torque control, which is easier
for RL to learn from. But the real drivetrain is an ODrive running in
*velocity* mode (see javis/robot_constants.py, WHEEL_ACTUATOR_CFG), so a
torque-action policy would need a control-mode change on the real hardware to
deploy -- a real sim-to-real gap. Using JointVelocityActionCfg here instead
means the policy's action *is* exactly what gets sent to the real ODrive.
"""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointVelocityActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity import mdp as velocity_mdp
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from .mdp import commands as javis_commands
from .robot_constants import WHEEL_COLLISION_GEOMS, get_javis_robot_cfg

_ROBOT = SceneEntityCfg("robot")


def _make_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  ##
  # Observations.
  #
  # No joint_pos term (unlike legged robots): a continuous wheel joint's
  # absolute angle carries no information about posture or balance, only
  # its velocity does.
  ##

  actor_terms = {
    "base_lin_vel": ObservationTermCfg(
      func=envs_mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
      noise=Unoise(n_min=-0.1, n_max=0.1),
    ),
    "base_ang_vel": ObservationTermCfg(
      func=envs_mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.1, n_max=0.1),
    ),
    "projected_gravity": ObservationTermCfg(
      func=envs_mdp.projected_gravity,
      noise=Unoise(n_min=-0.03, n_max=0.03),
    ),
    "wheel_vel": ObservationTermCfg(
      func=envs_mdp.joint_vel_rel,
      noise=Unoise(n_min=-0.5, n_max=0.5),
    ),
    "actions": ObservationTermCfg(func=envs_mdp.last_action),
    "command": ObservationTermCfg(
      func=envs_mdp.generated_commands,
      params={"command_name": "twist"},
    ),
  }

  observations = {
    "actor": ObservationGroupCfg(terms=actor_terms, enable_corruption=True),
    "critic": ObservationGroupCfg(terms={**actor_terms}),
  }

  ##
  # Actions: wheel *velocity* targets (see module docstring for why not
  # torque). The real practical top speed is now known -- WHEEL_ACTUATOR_CFG
  # (robot_constants.py) derives ~45 rad/s from the hub motor's 15-18 km/h
  # datasheet claim -- but scale is still a placeholder rather than just set
  # to that: an untrained policy samples actions from a ~unit-variance
  # Gaussian (see javis_ppo_runner_cfg's init_std=1.0), so scale is really
  # "how fast should a barely-trying policy move", not the hardware ceiling.
  # Setting it near the physical max would make early random exploration
  # dangerously fast; retune this (probably down) once training is actually
  # observed, not just once the physical limit is known.
  ##

  actions: dict[str, ActionTermCfg] = {
    "wheel_vel": JointVelocityActionCfg(
      entity_name="robot",
      actuator_names=("left_wheel", "right_wheel"),
      scale=5.0,
      use_default_offset=False,
    )
  }

  ##
  # Commands: forward/turn only -- a differential-drive base is
  # non-holonomic and cannot strafe, so lin_vel_y is fixed at 0 (unlike
  # legged robots, where mjlab's default twist command allows lateral
  # velocity).
  ##

  commands: dict[str, CommandTermCfg] = {
    # JavisVelocityCommandCfg, not mjlab's UniformVelocityCommandCfg directly:
    # same term, patched to not crash the Viser web viewer's joystick GUI when
    # an axis's range is zero-width -- see javis/mdp/commands.py.
    "twist": javis_commands.JavisVelocityCommandCfg(
      entity_name="robot",
      resampling_time_range=(3.0, 8.0),
      rel_standing_envs=0.2,  # some envs must learn to just balance in place
      heading_command=False,
      debug_vis=True,
      ranges=javis_commands.JavisVelocityCommandCfg.Ranges(
        lin_vel_x=(-0.5, 0.5),
        lin_vel_y=(0.0, 0.0),
        ang_vel_z=(-1.0, 1.0),
      ),
    )
  }

  ##
  # Events.
  ##

  events = {
    "reset_base": EventTermCfg(
      func=envs_mdp.reset_root_state_uniform,
      mode="reset",
      params={
        # Small roll/pitch so the policy sometimes starts already tipping,
        # not just perfectly balanced -- this rover falls over fast (see
        # README "known gaps"), so recovery needs to be trained explicitly,
        # not hoped for.
        "pose_range": {
          "x": (-0.5, 0.5),
          "y": (-0.5, 0.5),
          "z": (0.0, 0.02),
          "roll": (-0.2, 0.2),
          "pitch": (-0.2, 0.2),
          "yaw": (-math.pi, math.pi),
        },
        "velocity_range": {},
      },
    ),
    "reset_wheels": EventTermCfg(
      func=envs_mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (0.0, 0.0),
        "velocity_range": (-0.5, 0.5),
        "asset_cfg": SceneEntityCfg("robot", joint_names=("left_wheel", "right_wheel")),
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
    # Domain randomization standing in for measurements not taken yet (see
    # SIM2REAL.md secs 1 & 4) -- ranges are rough placeholders, not real
    # measured bounds. Narrow them once real numbers are known so the
    # policy isn't trained more conservatively than necessary.
    "wheel_friction": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=WHEEL_COLLISION_GEOMS),
        "operation": "abs",
        "ranges": (0.6, 1.4),
        "shared_random": True,
      },
    ),
    "chassis_com": EventTermCfg(
      mode="startup",
      func=dr.body_com_offset,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=("body",)),
        "operation": "add",
        "ranges": {0: (-0.02, 0.02), 1: (-0.02, 0.02), 2: (-0.03, 0.03)},
      },
    ),
  }

  ##
  # Rewards. `upright` is weighted well above tracking (unlike legged
  # robots, where it's a small auxiliary term) because staying upright is
  # this robot's core control problem, not a secondary constraint --
  # confirmed by scripts/view_robot.py: it falls over on its own within ~1s.
  ##

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
    "upright": RewardTermCfg(
      func=velocity_mdp.upright,
      weight=3.0,
      params={
        "std": math.sqrt(0.15),
        # Must be given explicitly: SceneEntityCfg.body_ids defaults to
        # slice(None) (all bodies), which is truthy, so `upright`'s
        # `if asset_cfg.body_ids` check silently selects every body's quat
        # (wrong shape) rather than falling back to the root -- passing the
        # chassis body by name is the "Set per-robot" override
        # mjlab/tasks/velocity/velocity_env_cfg.py leaves blank upstream.
        "asset_cfg": SceneEntityCfg("robot", body_names=("body",)),
      },
    ),
    "is_alive": RewardTermCfg(func=envs_mdp.is_alive, weight=0.5),
    "action_rate_l2": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.05),
  }

  ##
  # Terminations. limit_angle is tighter than mjlab's legged-robot default
  # (70 deg) -- a rover resting on its face (see README) is unrecoverable,
  # so episodes should end well before that, not after.
  ##

  terminations = {
    "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
    "fell_over": TerminationTermCfg(
      func=envs_mdp.bad_orientation,
      params={"limit_angle": math.radians(45.0)},
    ),
  }

  cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={"robot": get_javis_robot_cfg()},
      num_envs=1 if play else 4096,
      env_spacing=2.0,
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="body",
      distance=1.5,
      elevation=-20.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(mujoco=MujocoCfg(timestep=0.005)),
    decimation=4,  # 0.005s * 4 = 50 Hz control -- retune to match the real control loop rate (SIM2REAL.md sec 6)
    episode_length_s=20.0,
  )
  if play:
    cfg.episode_length_s = 1e10
    cfg.observations["actor"].enable_corruption = False
  return cfg


def javis_velocity_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return _make_env_cfg(play=play)


def javis_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(128, 128, 64),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(128, 128, 64),
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
    experiment_name="javis_velocity",
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=1000,
  )
