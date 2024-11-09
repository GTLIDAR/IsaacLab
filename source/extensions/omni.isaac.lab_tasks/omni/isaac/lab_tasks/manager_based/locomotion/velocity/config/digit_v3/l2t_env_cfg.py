from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from .env_cfg.observation_cfg import TeacherObsCfg, StudentObsCfg
import omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp as mdp
import omni.isaac.lab_tasks.manager_based.locomotion.velocity.config.digit_v3.mdp as digit_mdp
from omni.isaac.lab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    RewardsCfg,
    EventCfg,
)

##
# Pre-defined configs
##
from omni.isaac.lab_assets.digit import DIGITV3_CFG  # isort: skip
import math


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
        ),
    )


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
        use_default_offset=True,
        preserve_order=True,
    )


@configclass
class L2TObservationsCfg:
    """Observation specifications for the MDP."""

    # observation groups, defined in observation_cfg.py
    teacher: TeacherObsCfg = TeacherObsCfg()
    student: StudentObsCfg = StudentObsCfg()


@configclass
class L2TTerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)  # type: ignore
    base_contact = DoneTerm(
        func=mdp.illegal_contact,  # type: ignore
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[".*base", ".*hip.*", ".*knee", ".*elbow"],
            ),
            "threshold": 1.0,
        },
    )

    base_too_low = DoneTerm(
        func=digit_mdp.root_height_below_minimum_adaptive,  # type: ignore
        params={
            "minimum_height": 0.8,
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=[
                    ".*toe_roll.*",
                ],
            ),
        },
    )

    # bad_orientation = DoneTerm(
    #     func=mdp.bad_orientation,  # type: ignore
    #     params={"limit_angle": 0.7},
    # )

    # arm_deviation = DoneTerm(
    #     func=digit_mdp.arm_deviation_too_much,  # type: ignore
    #     params={
    #         "threshold": 1.0,
    #         "asset_cfg": SceneEntityCfg(
    #             "robot",
    #             joint_names=[
    #                 ".*_shoulder_pitch",
    #                 ".*_shoulder_roll",
    #                 ".*_shoulder_yaw",
    #                 ".*_elbow",
    #             ],
    #         ),
    #     },
    # )

    # joint_pos_out_of_limit = DoneTerm(
    #     func=mdp.joint_pos_out_of_limit,  # type: ignore
    #     params={
    #         "asset_cfg": SceneEntityCfg(
    #             "robot",
    #             joint_names=[
    #                 ".*_hip_.*",
    #                 ".*_knee",
    #                 ".*_toe.*",
    #                 ".*_shoulder.*",
    #                 ".*_elbow",
    #             ],
    #         ),
    #     },
    # )


@configclass
class DigitV3RewardsCfg(RewardsCfg):
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)  # type: ignore

    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.015)
    dof_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-5e-4)

    lin_vel_z_l2 = None
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=0.5,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=0.5,
        params={"command_name": "base_velocity", "std": 0.5},
    )

    # feet_air_time = None
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.25,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["left_toe_roll", "right_toe_roll"]
            ),
            "threshold": 0.4,
        },
    )

    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*toe_roll"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*toe_roll"),
        },
    )

    # Penalize ankle joint limits
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,  # type: ignore
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
            )
        },
    )

    # Penalize deviation from default of the joints that are not essential for locomotion
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,  # type: ignore
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*_hip_yaw", ".*_hip_roll"]
            )
        },
    )

    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,  # type: ignore
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_pitch",
                    ".*_shoulder_roll",
                    ".*_shoulder_yaw",
                    ".*_elbow",
                ],
            )
        },
    )

    # joint_deviation_toes = RewTerm(
    #     func=mdp.joint_deviation_l1,  # type: ignore
    #     weight=-0.1,
    #     params={
    #         "asset_cfg": SceneEntityCfg(
    #             "robot",
    #             joint_names=[
    #                 ".*_toe_A",
    #                 ".*_toe_B",
    #                 ".*_toe_pitch",
    #                 ".*_toe_roll",
    #             ],
    #         )
    #     },
    # )

    # foot_contact = RewTerm(
    #     func=digit_v3_mdp.reward_feet_contact_number,
    #     weight=0.5,
    #     params={
    #         "sensor_cfg": SceneEntityCfg(
    #             "contact_forces",
    #             body_names=["left_toe_roll", "right_toe_roll"],
    #             preserve_order=True,
    #         ),
    #         "pos_rw": 0.3,
    #         "neg_rw": -0.0,
    #     },
    # )

    # track_foot_height = RewTerm(
    #     func=digit_mdp.track_foot_height,
    #     weight=0.5,
    #     params={
    #         "std": 0.05,
    #         "asset_cfg": SceneEntityCfg(
    #             "robot",
    #             body_names=["left_toe_roll", "right_toe_roll"],
    #             preserve_order=True,
    #         ),
    #         "sensor_cfg": SceneEntityCfg(
    #             "contact_forces",
    #             body_names=["left_toe_roll", "right_toe_roll"],
    #             preserve_order=True,
    #         ),
    #     },
    # )

    foot_clearance = RewTerm(
        func=digit_mdp.foot_clearance_reward,
        weight=0.5,
        params={
            "target_height": 0.2,
            "std": 0.5,
            "tanh_mult": 2.0,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_toe_roll"),
        },
    )

    foot_distance = RewTerm(
        func=digit_mdp.feet_distance_l1,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["left_toe_roll", "right_toe_roll"],
                preserve_order=True,
            ),
            "min_dist": 0.2,
            "max_dist": 0.65,
        },
    )


@configclass
class DigitV3EventCfg(EventCfg):
    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=[
                    "base",
                    "left_shoulder_roll",
                    "right_shoulder_roll",
                    "left_hip_roll",
                    "right_hip_roll",
                    "left_shoulder_pitch",
                    "right_shoulder_pitch",
                    "left_hip_yaw",
                    "right_hip_yaw",
                    "left_shoulder_yaw",
                    "right_shoulder_yaw",
                    "left_hip_pitch",
                    "right_hip_pitch",
                    "left_elbow",
                    "right_elbow",
                    "left_knee",
                    "right_knee",
                    "left_achilles_rod",
                    "right_achilles_rod",
                    "left_shin",
                    "right_shin",
                    "left_tarsus",
                    "right_tarsus",
                    "left_heel_spring",
                    "right_heel_spring",
                    "left_toe_A",
                    "left_toe_B",
                    "left_toe_pitch",
                    "right_toe_A",
                    "right_toe_B",
                    "right_toe_pitch",
                    "left_toe_A_rod",
                    "left_toe_B_rod",
                    "left_toe_roll",
                    "right_toe_A_rod",
                    "right_toe_B_rod",
                    "right_toe_roll",
                ],
            ),
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (0.5, 1.8),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    # reset
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=[
                    "base",
                ],
            ),
            "mass_distribution_params": (0.5, 1.5),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    reset_gravity = EventTerm(
        func=mdp.randomize_physics_scene_gravity,
        mode="reset",
        params={
            "gravity_distribution_params": ([0.0, 0.0, -0.1], [0.0, 0.0, 0.1]),
            "operation": "add",
            "distribution": "gaussian",
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
        },
    )

    reset_robot_joints_offset = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.01, 0.01),
            "velocity_range": (-0.0, 0.0),
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.2, 0.2),
            "velocity_range": (-0.1, 0.1),
        },
    )

    robot_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
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
                preserve_order=True,
            ),
            "stiffness_distribution_params": (0.7, 2.2),
            "damping_distribution_params": (0.7, 2.2),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )


@configclass
class DigitV3L2TRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: DigitV3RewardsCfg = DigitV3RewardsCfg()
    terminations: L2TTerminationsCfg = L2TTerminationsCfg()
    actions: L2TDigitV3ActionCfg = L2TDigitV3ActionCfg()
    observations: L2TObservationsCfg = L2TObservationsCfg()
    events: DigitV3EventCfg = DigitV3EventCfg()
    commands: CommandsCfg = CommandsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.scene.env_spacing = 5.0
        self.sim.dt = 0.005
        self.decimation = 4
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
            ],
        )
        # Rewards
        self.rewards.dof_acc_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot",
            joint_names=[
                ".*_hip_.*",
                ".*_knee",
            ],
        )
        self.rewards.undesired_contacts.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces",
            body_names=[
                ".*knee.*",
                ".*tarsus.*",
                ".*rod.*",
                ".*shin.*",
            ],
        )

        self.rewards.undesired_contacts = None  # type: ignore
        self.rewards.dof_torques_l2.weight = 0
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.dof_acc_l2.weight = -1.25e-7
        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (-0.2, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.1, 0.1)
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
