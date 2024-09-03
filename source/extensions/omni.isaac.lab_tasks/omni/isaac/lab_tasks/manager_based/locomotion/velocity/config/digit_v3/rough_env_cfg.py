# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch

from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm
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
    EventCfg,
)


##
# Pre-defined configs
##
from omni.isaac.lab_assets.digit import DIGITV3_CFG  # isort: skip


@configclass
class DigitV3EventCfg(EventCfg):
    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (0.5, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    # reset
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.5, 1.5),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    reset_gravity = EventTerm(
        func=mdp.randomize_physics_scene_gravity,
        mode="reset",
        params={
            "gravity_distribution_params": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.67]),
            "operation": "add",
            "distribution": "gaussian",
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-0.5, -0.5)},
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
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.9, 1.1),
            "velocity_range": (0.0, 0.0),
        },
    )

    # interval
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="interval",
        interval_range_s=(0.0, 20.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (0.0, 5.0),
            "torque_range": (-5.0, 5.0),
        },
    )

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(0.0, 20.0),
        params={
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.1, 0.5),
            }
        },
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


@configclass
class DigitV3TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)  # type: ignore
    base_contact = DoneTerm(
        func=mdp.illegal_contact,  # type: ignore
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    ".*base",
                    ".*hip.*",
                    ".*knee",
                    ".*elbow",
                ],
            ),
            "threshold": 10.0,
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
        # scale=0.5,
        use_default_offset=False,
        preserve_order=True,
    )


@configclass
class DigitV3ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            scale=1,
            noise=Gnoise(mean=0.0, std=0.10, operation="add"),
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            scale=1,
            noise=Gnoise(mean=0.0, std=0.10, operation="add"),
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            scale=1,
            params={"command_name": "base_velocity"},
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos,
            scale=1,
            noise=Gnoise(mean=0.0, std=0.10, operation="add"),
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
            noise=Gnoise(mean=0.0, std=0.1, operation="add"),
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
    # Basic settings
    observations: DigitV3ObservationsCfg = DigitV3ObservationsCfg()
    actions: DigitV3ActionCfg = DigitV3ActionCfg()

    # MDP settings
    rewards: DigitV3RewardsCfg = DigitV3RewardsCfg()
    terminations: DigitV3TerminationsCfg = DigitV3TerminationsCfg()
    events: DigitV3EventCfg = DigitV3EventCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.sim.dt = 0.001  # 0.001
        # self.sim.render_interval = 20
        self.decimation = 20
        self.sim.gravity = (0.0, 0.0, -9.806)
        self.sim.render_interval = self.decimation

        # Scene
        self.scene.robot = DIGITV3_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"

        # Rewards
        self.rewards.dof_acc_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip.*", ".*_knee"]
        )

        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee"]  # ".*toe_roll", ".*toe_pitch"
        )

        self.rewards.undesired_contacts = None

        self.rewards.track_lin_vel_xy_exp.weight = 2.0
        self.rewards.track_ang_vel_z_exp.weight = 2.0
        self.rewards.alive.weight = 0.01
        self.rewards.lin_vel_z_l2.weight = -0.2
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.dof_torques_l2.weight = -1.5e-7
        self.rewards.dof_acc_l2.weight = -1.25e-7
        self.rewards.action_rate_l2.weight = -0.005

        self.rewards.feet_air_time.weight = 1.0
        self.rewards.flat_orientation_l2.weight = -2.0

        self.rewards.dof_pos_limits.weight = -0.1
        self.rewards.termination_penalty.weight = -200.0
        self.rewards.feet_slide.weight = -0.25
        self.rewards.joint_deviation_hip.weight = -0.1
        self.rewards.joint_deviation_arms.weight = -0.2
        self.rewards.joint_deviation_torso.weight = -0.1

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (-0.1, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        # self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)
        # self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        # self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)


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
        self.events.base_external_force_torque = None
        self.events.push_robot = None
