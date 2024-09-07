from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import RewardTermCfg as RewTerm

from omni.isaac.lab.utils import configclass

import omni.isaac.lab_tasks.manager_based.locomotion.velocity.config.digit_v3.mdp as digit_v3_mdp
import omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp as mdp
from omni.isaac.lab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    RewardsCfg,
)


@configclass
class DigitV3RewardsCfg(RewardsCfg):
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)  # type: ignore

    alive = RewTerm(
        func=mdp.is_alive,
        weight=0.01,
    )
    # lin_vel_z_l2 = None
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.25,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*toe_roll"),
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
        weight=-0.1,  # -1.0
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*toe_pitch", ".*toe_roll"]
            )
        },  # joint_names=".*toe_roll"v
    )
    # Penalize deviation from default of the joints that are not essential for locomotion
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,  # type: ignore
        weight=-0.1,  # -0.2
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*_hip_yaw", ".*_hip_roll"]
            )
        },
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,  # type: ignore
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_pitch",
                    ".*_shoulder_roll",
                    ".*_shoulder_yaw",
                    ".*_elbow",
                ],  # [".*_shoulder_.*", ".*_elbow"]
            )
        },
    )

    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1,  # type: ignore
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="base")},
    )

    foot_clearance = RewTerm(
        func=digit_v3_mdp.foot_clearance_reward,
        weight=0.5,
        params={
            "std": 0.05,
            "tanh_mult": 2.0,
            "target_height": 0.11,
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["left_toe_roll", "right_toe_roll"],
                preserve_order=True,
            ),
        },
    )

    foot_contact = RewTerm(
        func=digit_v3_mdp.reward_feet_contact_number,
        weight=0.5,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["left_toe_roll", "right_toe_roll"],
                preserve_order=True,
            ),
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["left_toe_roll", "right_toe_roll"],
                preserve_order=True,
            ),
        },
    )

    track_foot_height = RewTerm(
        func=digit_v3_mdp.track_foot_height_l1,
        weight=0.5,
        params={
            "std": 0.05,
            "tanh_mult": 2.0,
            "target_height": 0.11,
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["left_toe_roll", "right_toe_roll"],
                preserve_order=True,
            ),
        },
    )

    feet_distance = RewTerm(
        func=digit_v3_mdp.feet_distance,
        weight=0.5,
        params={
            # "sensor_cfg": SceneEntityCfg(
            #     "contact_forces",
            #     body_names=["left_toe_roll", "right_toe_roll"],
            #     preserve_order=True,
            # ),
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["left_toe_roll", "right_toe_roll"],
                preserve_order=True,
            ),
            "min_dist": 0.2,
            "max_dist": 0.5,
        },
    )
