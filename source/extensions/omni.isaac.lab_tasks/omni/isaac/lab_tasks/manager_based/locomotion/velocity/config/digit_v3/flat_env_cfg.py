# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils import configclass


from .rough_env_cfg import DigitV3RoughEnvCfg


@configclass
class DigitV3FlatEnvCfg(DigitV3RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # override rewards
        # self.rewards.feet_air_time.params["threshold"] = 0.4

        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee"]
        )

        self.rewards.alive.weight = 0.01
        self.rewards.track_lin_vel_xy_exp.weight = 2.25
        self.rewards.track_ang_vel_z_exp.weight = 2.25
        self.rewards.lin_vel_z_l2.weight = -0.3
        self.rewards.ang_vel_xy_l2.weight = -0.05

        self.rewards.dof_torques_l2.weight = -2.0e-6
        self.rewards.dof_acc_l2.weight = -1.0e-7
        # self.rewards.dof_vel_l2.weight = -1.0e-7

        self.rewards.action_rate_l2.weight = -0.005

        # self.rewards.feet_air_time.weight = 1.25
        # self.rewards.foot_clearance.weight = 0.5
        self.rewards.flat_orientation_l2.weight = -5.0
        self.rewards.foot_contact.weight = 1.0
        self.rewards.track_foot_height.weight = 0.5
        self.rewards.feet_distance_l1.weight = -0.1

        self.rewards.dof_pos_limits.weight = -0.1
        self.rewards.termination_penalty.weight = -200
        self.rewards.feet_slide.weight = -0.25
        self.rewards.joint_deviation_hip.weight = -0.2
        self.rewards.joint_deviation_arms.weight = -0.2
        self.rewards.joint_deviation_torso.weight = -0.2

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None


class DigitV3FlatEnvCfg_PLAY(DigitV3FlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
