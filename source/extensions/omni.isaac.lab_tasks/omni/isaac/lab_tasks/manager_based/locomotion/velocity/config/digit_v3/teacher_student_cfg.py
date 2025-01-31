from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils import configclass

from .env_cfg.observation_cfg import TeacherObsCfg, StudentObsCfg
from omni.isaac.lab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
)
from .env_cfg import (
    StudentObsCfg,
    DigitV3CommandsCfg,
    DigitV3TerminationsCfg,
    DigitV3RewardsCfg,
    DigitV3ActionCfg,
    DigitV3EventCfg,
)

##
# Pre-defined configs
##
from omni.isaac.lab_assets.digit import DIGITV3_CFG  # isort: skip


@configclass
class TeacherObservationsCfg:
    """Observation specifications for the MDP."""

    # observation groups, defined in observation_cfg.py
    policy: TeacherObsCfg = TeacherObsCfg()


@configclass
class StudentObservationsCfg:
    """Observation specifications for the MDP."""

    # observation groups, defined in observation_cfg.py
    teacher: TeacherObsCfg = TeacherObsCfg()
    student: StudentObsCfg = StudentObsCfg()


@configclass
class DigitV3TeacherRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: DigitV3RewardsCfg = DigitV3RewardsCfg()
    terminations: DigitV3TerminationsCfg = DigitV3TerminationsCfg()
    actions: DigitV3ActionCfg = DigitV3ActionCfg()
    observations: TeacherObservationsCfg = TeacherObservationsCfg()
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
        self.rewards.dof_torques_l2.weight = -2.0e-5
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.dof_acc_l2.weight = -1.25e-7

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (-0.3, 1.2)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.heading_command = False


@configclass
class DigitV3TeacherFlatEnvCfg(DigitV3TeacherRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None  # type: ignore

        # no terrain curriculum
        self.curriculum.terrain_levels = None  # type: ignore
        self.observations.policy.height_scan = None  # type: ignore


@configclass
class DigitV3StudentRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: DigitV3RewardsCfg = DigitV3RewardsCfg()
    terminations: L2TTerminationsCfg = L2TTerminationsCfg()
    actions: L2TDigitV3ActionCfg = L2TDigitV3ActionCfg()
    observations: TeacherObservationsCfg = TeacherObservationsCfg()
    events: DigitV3EventCfg = DigitV3EventCfg()
    commands: CommandsCfg = CommandsCfg()

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
        self.rewards.dof_torques_l2.weight = -2.0e-5
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.dof_acc_l2.weight = -1.25e-7

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (-0.3, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.heading_command = False


@configclass
class DigitV3StudentFlatEnvCfg(DigitV3StudentRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None  # type: ignore

        # no terrain curriculum
        self.curriculum.terrain_levels = None  # type: ignore
        self.observations.teacher.height_scan = None  # type: ignore
