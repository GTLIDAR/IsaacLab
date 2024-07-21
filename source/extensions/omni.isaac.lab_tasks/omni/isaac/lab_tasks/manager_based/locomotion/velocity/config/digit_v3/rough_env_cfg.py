# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch

import omni.isaac.lab.envs.mdp as mdp
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.envs import ManagerBasedEnv
from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from omni.isaac.lab.utils.noise import AdditiveGaussianNoiseCfg as Gnoise
from omni.isaac.lab.utils import configclass

import omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp as mdp
from omni.isaac.lab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    RewardsCfg,
)

##
# Pre-defined configs
##
from omni.isaac.lab_assets.digit import DIGITV3_CFG  # isort: skip


@configclass
class DigitV3Rewards(RewardsCfg):
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

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
        func=mdp.joint_pos_limits,
        weight=-0.1, # -1.0
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*toe_pitch", ".*toe_roll"])}, # joint_names=".*toe_roll"v
    )
    # Penalize deviation from default of the joints that are not essential for locomotion
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1, #-0.2
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*_hip_yaw", ".*_hip_roll"] 
            )
        },
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[
                    ".*_shoulder_pitch",
                    ".*_shoulder_roll",
                    ".*_shoulder_yaw",
                    ".*_elbow",
                ] #[".*_shoulder_.*", ".*_elbow"]
            )
        },
    )
    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="base")},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*base"),
            "threshold": 10.0,
        },
    )


@configclass
class ActionCfg:
    """Action terms for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
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
        # use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Gnoise(mean=0.0, std=0.15, operation="add"))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Gnoise(mean=0.0, std=0.15, operation="add"))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos, noise=Gnoise(mean=0.0, std=0.175, operation="add"),
                            params={
                                "asset_cfg": SceneEntityCfg(
                                "robot", joint_names=[
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
                                                "left_shin",
                                                "left_tarsus",
                                                "left_toe_pitch",
                                                "left_toe_roll",
                                                "left_heel_spring",
                                                "right_shin",
                                                "right_tarsus",
                                                "right_toe_pitch",
                                                "right_toe_roll",
                                                "right_heel_spring"],
                                            preserve_order = True
                                                )
                                        })
        
        
        
        joint_vel = ObsTerm(func=mdp.joint_vel, noise=Gnoise(mean=0.0, std=0.15, operation="add"),
                            params={
                                "asset_cfg": SceneEntityCfg(
                                "robot", joint_names=[
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
                                                "left_shin",
                                                "left_tarsus",
                                                "left_toe_pitch",
                                                "left_toe_roll",
                                                "left_heel_spring",
                                                "right_shin",
                                                "right_tarsus",
                                                "right_toe_pitch",
                                                "right_toe_roll",
                                                "right_heel_spring",],
                                            preserve_order = True
                                            )
                                    })
        actions = ObsTerm(func=mdp.last_action)



        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()



@configclass
class DigitV3RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: DigitV3Rewards = DigitV3Rewards()
    terminations: TerminationsCfg = TerminationsCfg()
    actions: ActionCfg = ActionCfg()
    observations: ObservationsCfg = ObservationsCfg()


    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.sim.dt = 0.001
        self.decimation = 20
        # Scene
        self.scene.robot = DIGITV3_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"

        # Randomization
        self.events.push_robot.params["asset_cfg"].body_names = [
            ".*base"
        ]
        self.events.add_base_mass.params["asset_cfg"].body_names = [
            ".*base"
        ]
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.base_external_force_torque.params["asset_cfg"].body_names = [
            ".*base"
        ]
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-0.5, 0.5)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        }

        # Terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = [".*base", ".*hip.*",
                                                                           ".*knee", ".*elbow"]

        # # Rewards
        self.rewards.undesired_contacts = None
        self.rewards.flat_orientation_l2.weight = -1.0
        # self.rewards.dof_torques_l2.weight = 0.0
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.dof_acc_l2.weight = -1.25e-7
        # Rewards
        self.rewards.lin_vel_z_l2.weight = 0.0


        self.rewards.dof_acc_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip.*", ".*_knee"]
        )
        self.rewards.dof_torques_l2.weight = -1.5e-7
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee", ".*toe_roll", ".*toe_pitch"]
        )

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)


@configclass
class DigitV3RoughEnvCfg_PLAY(DigitV3RoughEnvCfg):
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
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None
