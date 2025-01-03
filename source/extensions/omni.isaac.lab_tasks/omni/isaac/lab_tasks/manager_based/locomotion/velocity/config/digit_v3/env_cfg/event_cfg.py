from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils import configclass

import omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp as mdp
import omni.isaac.lab_tasks.manager_based.locomotion.velocity.config.digit_v3.mdp as digit_mdp
from omni.isaac.lab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    EventCfg,
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
            "static_friction_range": (0.6, 1.5),
            "dynamic_friction_range": (0.4, 1.5),
            "restitution_range": (0.0, 0.4),
            "num_buckets": 64,
        },
    )

    reset_gravity = EventTerm(
        func=mdp.randomize_physics_scene_gravity,
        mode="reset",
        params={
            "gravity_distribution_params": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.67]),
            "operation": "add",
            "distribution": "uniform",
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

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.035, 0.035),
            "velocity_range": (-0.00, 0.00),
        },
    )

    # robot_joint_stiffness_and_damping = EventTerm(
    #     func=digit_mdp.randomize_actuator_gains,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg(
    #             "robot",
    #             joint_names=[
    #                 "left_hip_roll",
    #                 "left_hip_yaw",
    #                 "left_hip_pitch",
    #                 "left_knee",
    #                 "left_toe_A",
    #                 "left_toe_B",
    #                 "right_hip_roll",
    #                 "right_hip_yaw",
    #                 "right_hip_pitch",
    #                 "right_knee",
    #                 "right_toe_A",
    #                 "right_toe_B",
    #                 "left_shoulder_roll",
    #                 "left_shoulder_pitch",
    #                 "left_shoulder_yaw",
    #                 "left_elbow",
    #                 "right_shoulder_roll",
    #                 "right_shoulder_pitch",
    #                 "right_shoulder_yaw",
    #                 "right_elbow",
    #             ],
    #             preserve_order=True,
    #         ),
    #         "stiffness_distribution_params": (0.9, 1.1),
    #         "damping_distribution_params": (0.9, 1.1),
    #         "operation": "scale",
    #         "distribution": "log_uniform",
    #     },
    # )
