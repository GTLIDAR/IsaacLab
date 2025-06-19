from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from .env_cfg import (
    TeacherObsCfg,
    StudentObsCfg,
    DigitV3CommandsCfg,
    DigitV3TerminationsCfg,
    DigitV3RewardsCfg,
    DigitV3ActionCfg,
    DigitV3EventCfg,
)
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
)

##
# Pre-defined configs
##
from isaaclab_assets.robots.digit import DIGITV3_CFG  # isort: skip


@configclass
class L2TObservationsCfg:
    """Observation specifications for the MDP."""

    # observation groups, defined in observation_cfg.py
    teacher: TeacherObsCfg = TeacherObsCfg()
    student: StudentObsCfg = StudentObsCfg()
    policy: TeacherObsCfg = TeacherObsCfg()


@configclass 
class DigitV3RecorderCfg:
    """Recorder configurations for the MDP."""

    # recorder groups
    teacher: TeacherObsCfg = TeacherObsCfg()
    student: StudentObsCfg = StudentObsCfg()

@configclass
class DigitV3L2TRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: DigitV3RewardsCfg = DigitV3RewardsCfg()
    terminations: DigitV3TerminationsCfg = DigitV3TerminationsCfg()
    actions: DigitV3ActionCfg = DigitV3ActionCfg()
    observations: L2TObservationsCfg = L2TObservationsCfg()
    events: DigitV3EventCfg = DigitV3EventCfg()
    commands: DigitV3CommandsCfg = DigitV3CommandsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.scene.env_spacing = 5.0
        self.sim.dt = 0.001
        self.decimation = 20
        self.sim.gravity = (0.0, 0.0, -9.806)
        self.sim.render_interval = self.decimation
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**26
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 2**22
        
        # Scene
        self.scene.robot = DIGITV3_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")  # type: ignore
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot",
            joint_names=[
                ".*_hip_.*",
                ".*_knee",
                ".*_toe.*",
                ".*_shoulder.*",
                ".*_elbow",
            ],
        )
        # Rewards
        self.rewards.dof_acc_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot",
            joint_names=[
                ".*_hip_.*",
                ".*_knee",
                ".*_shoulder.*",
                ".*_elbow",
            ],
        )

        self.rewards.undesired_contacts = None  # type: ignore
        # self.rewards.alive.weight = 0.0
        self.rewards.track_lin_vel_xy_exp.weight = 0.5
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        self.rewards.ang_vel_xy_l2.weight = -0.1
        self.rewards.dof_pos_limits.weight = -0.5
        self.rewards.termination_penalty.weight = -200
        self.rewards.feet_slide.weight = -1.0
        self.rewards.joint_deviation_hip.weight = -5.0
        self.rewards.flat_orientation_l2.weight = -10.0
        self.rewards.dof_torques_l2.weight = -1.0e-5
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.dof_acc_l2.weight = -1.25e-7

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (-0.3, 1.2)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.heading_command = False


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

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None  # type: ignore

        # no terrain curriculum
        self.curriculum.terrain_levels = None  # type: ignore
        self.observations.teacher.height_scan = None  # type: ignore


class DigitV3L2TFlatEnvCfg_PLAY(DigitV3L2TFlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.teacher.enable_corruption = False
        # self.observations.student.enable_corruption = (
        #     False  # remove random pushing event
        # )
        self.events.base_external_force_torque = None  # type: ignore
        self.events.push_robot = None  # type: ignore
        self.events.robot_joint_stiffness_and_damping = None  # type: ignore
