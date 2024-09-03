from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
)

from .rough_env_cfg import (
    DigitV3RewardsCfg,
    DigitV3TerminationsCfg,
    DigitV3ActionCfg,
    DigitV3EventCfg,
)
from .env_cfg.observation_cfg import TeacherObsCfg, StudentObsCfg


##
# Pre-defined configs
##
from omni.isaac.lab_assets.digit import DIGITV3_CFG  # isort: skip


@configclass
class L2TObservationsCfg:
    """Observation specifications for the MDP."""

    # observation groups, defined in observation_cfg.py
    teacher: TeacherObsCfg = TeacherObsCfg()
    student: StudentObsCfg = StudentObsCfg()


@configclass
class DigitV3L2TRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: DigitV3RewardsCfg = DigitV3RewardsCfg()
    terminations: DigitV3TerminationsCfg = DigitV3TerminationsCfg()
    actions: DigitV3ActionCfg = DigitV3ActionCfg()
    observations: L2TObservationsCfg = L2TObservationsCfg()
    events: DigitV3EventCfg = DigitV3EventCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.sim.dt = 0.001  # 0.001
        # self.sim.render_interval = 20
        self.decimation = 20
        self.sim.gravity = (0.0, 0.0, -9.806)
        self.sim.render_interval = self.decimation

        # Scene
        self.scene.robot = DIGITV3_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"

        # Rewards
        self.rewards.dof_acc_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip.*", ".*_knee"]
        )

        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee"]  # ".*toe_roll", ".*toe_pitch"
        )

        self.rewards.undesired_contacts = None

        self.rewards.track_lin_vel_xy_exp.weight = 2.0
        self.rewards.track_ang_vel_z_exp.weight = 2.0
        self.rewards.alive.weight = 0.01
        self.rewards.lin_vel_z_l2.weight = -0.2
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.dof_torques_l2.weight = -1.5e-7
        self.rewards.dof_acc_l2.weight = -1.25e-7
        self.rewards.action_rate_l2.weight = -0.005

        self.rewards.feet_air_time.weight = 1.0
        self.rewards.flat_orientation_l2.weight = -2.0

        self.rewards.dof_pos_limits.weight = -0.1
        self.rewards.termination_penalty.weight = -200.0
        self.rewards.feet_slide.weight = -0.25
        self.rewards.joint_deviation_hip.weight = -0.1
        self.rewards.joint_deviation_arms.weight = -0.2
        self.rewards.joint_deviation_torso.weight = -0.1

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (-0.1, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)


@configclass
class DigitV3L2TRoughEnvCfg_PLAY(DigitV3L2TRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        # disable randomization for play
        self.observations.teacher.enable_corruption = False
        self.observations.student.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None  # type: ignore
        self.events.push_robot = None  # type: ignore


@configclass
class DigitV3L2TFlatEnvCfg(DigitV3L2TRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # override rewards
        # self.rewards.flat_orientation_l2.weight = -5.0

        # self.rewards.dof_torques_l2.weight = -2.5e-5
        # self.rewards.feet_air_time.weight = 0.5

        self.rewards.track_ang_vel_z_exp.weight = 1.0
        self.rewards.lin_vel_z_l2.weight = -0.2
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.dof_acc_l2.weight = -1.0e-7
        self.rewards.feet_air_time.weight = 0.75
        self.rewards.feet_air_time.params["threshold"] = 0.4
        self.rewards.dof_torques_l2.weight = -2.0e-6
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee"]
        )

        # self.rewards.feet_air_time.weight = 1.0
        # self.rewards.feet_air_time.params["threshold"] = 0.6

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None  # type: ignore
        self.observations.teacher.height_scan = None  # type: ignore
        self.observations.student.height_scan = None  # type: ignore

        # no terrain curriculum
        self.curriculum.terrain_levels = None  # type: ignore


class DigitV3L2TFlatEnvCfg_PLAY(DigitV3L2TFlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.teacher.enable_corruption = False
        self.observations.student.enable_corruption = (
            False  # remove random pushing event
        )
        self.events.base_external_force_torque = None  # type: ignore
        self.events.push_robot = None  # type: ignore
