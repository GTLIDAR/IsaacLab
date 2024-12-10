# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import math

from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from omni.isaac.lab.utils.noise import AdditiveGaussianNoiseCfg as Gnoise
from omni.isaac.lab.utils import configclass
from .env_cfg.observation_cfg import TeacherObsCfg, StudentObsCfg

import omni.isaac.lab_tasks.manager_based.locomotion.velocity.config.digit_v3.mdp as digit_mdp
import omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp as mdp
from omni.isaac.lab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
)

from .env_cfg import DigitV3RewardsCfg, DigitV3EventCfg

from omni.isaac.lab_assets.digit import DIGITV3_CFG  # isort: skip


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
class DigitV3TerminationsCfg:
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
            "minimum_height": 0.4,
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=[
                    ".*toe_roll.*",
                ],
            ),
        },
    )


@configclass
class DigitV3ActionCfg:
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
        scale={
            "left_hip_roll": 1.0,
            "left_hip_yaw": 1.0,
            "left_hip_pitch": 1.0,
            "left_knee": 1.0,
            "left_toe_A": 0.0,
            "left_toe_B": 0.0,
            "right_hip_roll": 1.0,
            "right_hip_yaw": 1.0,
            "right_hip_pitch": 1.0,
            "right_knee": 1.0,
            "right_toe_A": 0.0,
            "right_toe_B": 0.0,
            "left_shoulder_roll": 1.0,
            "left_shoulder_pitch": 1.0,
            "left_shoulder_yaw": 1.0,
            "left_elbow": 1.0,
            "right_shoulder_roll": 1.0,
            "right_shoulder_pitch": 1.0,
            "right_shoulder_yaw": 1.0,
            "right_elbow": 1.0,
        },
        use_default_offset=True,
        preserve_order=True,
    )


@configclass
class DigitV3ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        clock = ObsTerm(
            func=digit_mdp.clock,
        )
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            scale=1,
            noise=Gnoise(mean=0.0, std=0.15),
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            scale=1,
            noise=Gnoise(mean=0.0, std=0.15),
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Gnoise(mean=0.0, std=0.075),
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            scale=1,
            params={"command_name": "base_velocity"},
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos,
            scale=1,
            noise=Gnoise(mean=0.0, std=0.175),
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
                        "left_shin",
                        "left_tarsus",
                        "left_toe_pitch",
                        "left_toe_roll",
                        "left_heel_spring",
                        "right_shin",
                        "right_tarsus",
                        "right_toe_pitch",
                        "right_toe_roll",
                        "right_heel_spring",
                    ],
                    preserve_order=True,
                )
            },
        )

        joint_vel = ObsTerm(
            func=mdp.joint_vel,
            scale=1,
            noise=Gnoise(mean=0.0, std=0.175),
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
                        "left_shin",
                        "left_tarsus",
                        "left_toe_pitch",
                        "left_toe_roll",
                        "left_heel_spring",
                        "right_shin",
                        "right_tarsus",
                        "right_toe_pitch",
                        "right_toe_roll",
                        "right_heel_spring",
                    ],
                    preserve_order=True,
                )
            },
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class DigitV3RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    observations: DigitV3ObservationsCfg = DigitV3ObservationsCfg()
    actions: DigitV3ActionCfg = DigitV3ActionCfg()
    rewards: DigitV3RewardsCfg = DigitV3RewardsCfg()
    terminations: DigitV3TerminationsCfg = DigitV3TerminationsCfg()
    events: DigitV3EventCfg = DigitV3EventCfg()

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
        self.commands.base_velocity.ranges.lin_vel_x = (-0.3, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.heading_command = False


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

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None  # type: ignore
        self.events.push_robot = None  # type: ignore

        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
