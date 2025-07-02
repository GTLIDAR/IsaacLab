# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.unitree import G1_CFG

from ...imitation_env_cfg import (
    ImitationLearningEnvCfg,
    RewardsCfg,
)
from .mdp import track_joint_reference, track_root_pos, track_root_ang


# --- Rewards ---
@configclass
class G1RewardsCfg(RewardsCfg):
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
    track_root_pos = RewTerm(
        func=track_root_pos,
        weight=0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "sigma": 0.1,
        },
    )
    track_root_ang = RewTerm(
        func=track_root_ang,
        weight=0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "sigma": 0.1,
        },
    )


@configclass
class ImitationG1EnvCfg(ImitationLearningEnvCfg):
    # MDP settings
    rewards: RewardsCfg = G1RewardsCfg()
    # Dataset settings
    dataset_type: str = "zarr"
    # Dataset and cache settings for ImitationRLEnv
    dataset_path: str = "/tmp/iltools_zarr"
    window_size: int = 64  # Window size for per-env cache
    batch_size: int = 1  # Batch size for Zarr prefetching
    device: str = "cuda"  # Torch device
    loader_type: str = "loco_mujoco"  # Loader type (required if Zarr does not exist)
    loader_kwargs: dict = {
        "env_name": "UnitreeG1",
        "task": "walk",
    }  # Loader kwargs (required if Zarr does not exist)
    replay_reference: bool = True

    # debug timing
    debug_timing: bool = True

    # Reference joint names for the robot from the reference qpos order (this is the order of G1 in loco-mujoco)
    reference_joint_names: list[str] = [
        "root_x",
        "root_y",
        "root_z",
        "root_qw",
        "root_qx",
        "root_qy",
        "root_qz",
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

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # post init of parent
        super().__post_init__()  # type: ignore
        # Scene
        self.scene.robot = G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")  # type: ignore

        # Randomization
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)

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
        self.terminations.base_contact.params["sensor_cfg"].body_names = "pelvis"
        self.events.add_base_mass.params["asset_cfg"].body_names = "pelvis"
        self.events.base_com.params["asset_cfg"].body_names = "pelvis"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "pelvis"
