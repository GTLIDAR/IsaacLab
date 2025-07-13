# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.unitree import G1_CFG
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from ...imitation_env_cfg import (
    ImitationLearningEnvCfg,
    RewardsCfg,
)
from .mdp import track_joint_reference, track_root_pos, track_root_ang


# --- Rewards ---
@configclass
class G1RewardsCfg:
    # Borrow all velocity task rewards, then add imitation-specific ones
    track_joint_reference = RewTerm(
        func=track_joint_reference,
        weight=0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "left_hip_pitch_joint",
                    "left_hip_roll_joint",
                    "left_hip_yaw_joint",
                    "left_knee_joint",
                    "left_ankle_pitch_joint",
                    "left_ankle_roll_joint",
                    "right_hip_pitch_joint",
                    "right_hip_roll_joint",
                    "right_hip_yaw_joint",
                    "right_knee_joint",
                    "right_ankle_pitch_joint",
                    "right_ankle_roll_joint",
                    "torso_joint",
                    "left_shoulder_pitch_joint",
                    "left_shoulder_roll_joint",
                    "left_shoulder_yaw_joint",
                    "left_elbow_pitch_joint",
                    "left_elbow_roll_joint",
                    "right_shoulder_pitch_joint",
                    "right_shoulder_roll_joint",
                    "right_shoulder_yaw_joint",
                    "right_elbow_pitch_joint",
                    "right_elbow_roll_joint",
                ],
            ),
            "sigma": 0.25,
        },
    )
    # track_root_pos = RewTerm(
    #     func=track_root_pos,
    #     weight=0.1,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "sigma": 0.1,
    #     },
    # )
    # track_root_ang = RewTerm(
    #     func=track_root_ang,
    #     weight=0.1,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "sigma": 0.1,
    #     },
    # )

    """Reward terms from locomotion velocity task."""

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=".*_ankle_roll_link"
            ),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )

    # Penalize ankle joint limits
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"]
            )
        },
    )
    # Penalize deviation from default of the joints that are not essential for locomotion
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"]
            )
        },
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_pitch_joint",
                    ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",
                    ".*_elbow_pitch_joint",
                    ".*_elbow_roll_joint",
                ],
            )
        },
    )
    joint_deviation_fingers = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_five_joint",
                    ".*_three_joint",
                    ".*_six_joint",
                    ".*_four_joint",
                    ".*_zero_joint",
                    ".*_one_joint",
                    ".*_two_joint",
                ],
            )
        },
    )
    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="torso_joint")},
    )

    # -- penalties
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*THIGH"),
            "threshold": 1.0,
        },
    )
    # -- optional penalties
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=0.0)


@configclass
class ImitationG1EnvCfg(ImitationLearningEnvCfg):
    # MDP settings
    rewards = G1RewardsCfg()  # type: ignore
    # Dataset and cache settings for ImitationRLEnv
    device: str = "cuda"  # Torch device
    loader_type: str = "loco_mujoco"  # Loader type (required if Zarr does not exist)
    loader_kwargs: dict = {
        "env_name": "UnitreeG1",
    }  # Loader kwargs (required if Zarr does not exist)
    dataset: dict = {"trajectories": {"default": ["walk"], "amass": [], "lafan1": []}}
    replay_reference: bool = True
    # Reference joint names for the robot from the reference qpos order (this is the order of G1 in loco-mujoco)
    reference_joint_names: list[str] = [
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "torso_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_pitch_joint",
        "left_elbow_roll_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_pitch_joint",
        "right_elbow_roll_joint",
    ]

    # target joint names
    target_joint_names: list[str] = [
        "left_hip_pitch_joint",
        "right_hip_pitch_joint",
        "torso_joint",
        "left_hip_roll_joint",
        "right_hip_roll_joint",
        "left_shoulder_pitch_joint",
        "right_shoulder_pitch_joint",
        "left_hip_yaw_joint",
        "right_hip_yaw_joint",
        "left_shoulder_roll_joint",
        "right_shoulder_roll_joint",
        "left_knee_joint",
        "right_knee_joint",
        "left_shoulder_yaw_joint",
        "right_shoulder_yaw_joint",
        "left_ankle_pitch_joint",
        "right_ankle_pitch_joint",
        "left_elbow_pitch_joint",
        "right_elbow_pitch_joint",
        "left_ankle_roll_joint",
        "right_ankle_roll_joint",
        "left_elbow_roll_joint",
        "right_elbow_roll_joint",
        "left_five_joint",
        "left_three_joint",
        "left_zero_joint",
        "right_five_joint",
        "right_three_joint",
        "right_zero_joint",
        "left_six_joint",
        "left_four_joint",
        "left_one_joint",
        "right_six_joint",
        "right_four_joint",
        "right_one_joint",
        "left_two_joint",
        "right_two_joint",
    ]

    # n substep, unitree g1 has dt 0.001 in mujoco, and we have sim.dt * decimation = 0.02
    n_substeps: int = 20

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # post init of parent
        super().__post_init__()  # type: ignore
        # Scene
        self.scene.robot = G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")  # type: ignore

        # Randomization
        # Randomization
        self.events.push_robot = None
        self.events.add_base_mass = None
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.base_external_force_torque.params["asset_cfg"].body_names = [
            "torso_link"
        ]
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        self.events.base_com = None

        # Rewards
        self.rewards.lin_vel_z_l2.weight = 0.0
        self.rewards.undesired_contacts = None  # type: ignore
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.dof_acc_l2.weight = -1.25e-7
        self.rewards.dof_acc_l2.params["asset_cfg"] = SceneEntityCfg(  # type: ignore
            "robot",
            joint_names=[".*_hip_.*", ".*_knee_joint"],  # type: ignore
        )
        self.rewards.dof_torques_l2.weight = -1.5e-7
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(  # type: ignore
            "robot",
            joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"],  # type: ignore
        )

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = "torso_link"
