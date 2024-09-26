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
import omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp as mdp

##
# Pre-defined configs
##
from omni.isaac.lab_assets.digit import DIGITV3_CFG  # isort: skip


@configclass
class L2TDigitV3ActionCfg:
    """Action terms for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(  # type: ignore
        asset_name="robot",
        joint_names=[
            "left_hip_roll",
            "left_hip_yaw",
            "left_hip_pitch",
            "left_knee",
            "left_toe_A",
            "left_toe_B",
            "right_hip_roll",
            "right_hip_yaw",
            "right_hip_pitch",
            "right_knee",
            "right_toe_A",
            "right_toe_B",
            "left_shoulder_roll",
            "left_shoulder_pitch",
            "left_shoulder_yaw",
            "left_elbow",
            "right_shoulder_roll",
            "right_shoulder_pitch",
            "right_shoulder_yaw",
            "right_elbow",
        ],
        # scale=0.5,
        use_default_offset=False,
        preserve_order=True,
    )


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
    actions: L2TDigitV3ActionCfg = L2TDigitV3ActionCfg()
    observations: L2TObservationsCfg = L2TObservationsCfg()
    events: DigitV3EventCfg = DigitV3EventCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.scene.env_spacing = 5.0
        self.sim.dt = 0.0005
        self.decimation = 10
        self.sim.gravity = (0.0, 0.0, -9.806)
        self.sim.render_interval = self.decimation

        # Scene
        self.scene.robot = DIGITV3_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")  # type: ignore
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee"]
        )
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee"]
        )
        # Rewards
        self.rewards.dof_acc_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip.*", ".*_knee"]
        )

        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee"]  # ".*toe_roll", ".*toe_pitch"
        )

        self.rewards.undesired_contacts = None
        self.rewards.alive.weight = 0.1
        self.rewards.track_lin_vel_xy_exp.weight = 2.25
        self.rewards.track_ang_vel_z_exp.weight = 2.25
        self.rewards.lin_vel_z_l2.weight = -3.0
        self.rewards.ang_vel_xy_l2.weight = -0.75
        self.rewards.track_lin_vel_xy_exp.weight = 2.25
        self.rewards.track_ang_vel_z_exp.weight = 2.25
        self.rewards.lin_vel_z_l2.weight = -0.5
        self.rewards.ang_vel_xy_l2.weight = -0.075
        self.rewards.feet_air_time.weight = 5.0
        self.rewards.track_foot_height.weight = 5.0
        self.rewards.dof_pos_limits.weight = -0.1
        self.rewards.termination_penalty.weight = -200
        self.rewards.feet_slide.weight = -0.25
        self.rewards.joint_deviation_hip.weight = -0.2
        self.rewards.joint_deviation_torso.weight = -0.2
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.dof_torques_l2.weight = 0.0
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.dof_acc_l2.weight = -1.25e-7

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
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
        self.rewards.alive.weight = 20
        self.rewards.track_lin_vel_xy_exp.weight = 2.0
        self.rewards.track_ang_vel_z_exp.weight = 2.0
        self.rewards.lin_vel_z_l2.weight = -0.3
        self.rewards.ang_vel_xy_l2.weight = -0.2

        self.rewards.dof_torques_l2.weight = -2.0e-6
        self.rewards.dof_acc_l2.weight = -1.0e-7
        # self.rewards.dof_vel_l2.weight = -1.0e-7

        self.rewards.action_rate_l2.weight = -0.005

        # self.rewards.feet_air_time.weight = 1.25
        # self.rewards.foot_clearance.weight = 0.5
        self.rewards.flat_orientation_l2.weight = -5.0
        self.rewards.foot_contact.weight = 0.5
        # self.rewards.track_foot_height.weight = 0.5
        # self.rewards.feet_distance.weight = 0.01

        self.rewards.dof_pos_limits.weight = -0.1
        self.rewards.termination_penalty.weight = -200
        self.rewards.feet_slide.weight = -0.25
        self.rewards.joint_deviation_hip.weight = -0.2
        self.rewards.joint_deviation_arms.weight = -0.2
        self.rewards.joint_deviation_torso.weight = -0.2

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
