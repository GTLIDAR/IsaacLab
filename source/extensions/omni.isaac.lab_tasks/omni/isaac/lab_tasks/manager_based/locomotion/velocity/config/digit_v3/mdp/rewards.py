# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.managers import ManagerTermBase, SceneEntityCfg
from omni.isaac.lab.sensors import ContactSensor

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv
    from omni.isaac.lab.managers import RewardTermCfg


def reward_feet_contact_number(
    env, sensor_cfg: SceneEntityCfg, pos_rw: float, neg_rw: float
) -> torch.Tensor:
    """
    Calculates a reward based on the number of feet contacts aligning with the gait phase.
    Rewards or penalizes depending on whether the foot contact matches the expected gait phase.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
        .norm(dim=-1)
        .max(dim=1)[0]
        > 1.0
    )
    # print("contact", contacts.shape, contacts)
    phase = env.get_phase()
    sin_pos = torch.sin(2 * torch.pi * phase)
    stance_mask = torch.zeros((env.num_envs, 2), device=env.device)
    stance_mask[:, 0] = sin_pos >= 0
    stance_mask[:, 1] = sin_pos < 0
    stance_mask[torch.abs(sin_pos) < 0.1] = 1
    mask_2 = 1 - stance_mask
    mask_2[torch.abs(sin_pos) < 0.1] = 1
    # print("mask2", mask_2.shape, mask_2)
    # print("stance", stance_mask.shape, stance_mask)
    # print("")
    if torch.sum(contacts == stance_mask) > torch.sum(contacts == mask_2):
        # print("1")
        reward = torch.where(contacts == stance_mask, pos_rw, neg_rw)
        return torch.mean(reward, dim=1)

    # print("2")
    reward = torch.where(contacts == mask_2, pos_rw, neg_rw)
    return torch.mean(reward, dim=1)


def foot_clearance_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, std: float, tanh_mult: float
) -> torch.Tensor:
    """
    Reward the swinging feet for clearing a specified height off the ground
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_velocity_tanh = torch.tanh(
        tanh_mult
        * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)
    )
    reward = foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)


@torch.jit.script
def height_target(t: torch.Tensor) -> torch.Tensor:

    a5, a4, a3, a2, a1, a0 = [9.6, 12.0, -18.8, 5.0, 0.1, 0.0]
    return a5 * t**5 + a4 * t**4 + a3 * t**3 + a2 * t**2 + a1 * t + a0


def track_foot_height(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    std: float,
) -> torch.Tensor:
    """"""

    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
        .norm(dim=-1)
        .max(dim=1)[0]
        > 1.0
    )

    phase = env.get_phase()

    sin_pos = torch.sin(2 * torch.pi * phase)
    stance_mask = torch.zeros((env.num_envs, 2), device=env.device)
    stance_mask[:, 0] = sin_pos >= 0
    stance_mask[:, 1] = sin_pos < 0
    stance_mask[torch.abs(sin_pos) < 0.1] = 1
    # swing_mask = 1 - stance_mask
    mask_2 = 1 - stance_mask
    mask_2[torch.abs(sin_pos) < 0.1] = 1

    if torch.sum(contacts == stance_mask) > torch.sum(contacts == mask_2):
        swing_mask = 1 - stance_mask
    else:
        swing_mask = 1 - mask_2

    filt_foot = torch.where(swing_mask == 1, foot_z, torch.zeros_like(foot_z))

    phase_mod = torch.fmod(phase, 0.5)
    feet_z_target = height_target(phase_mod) + torch.min(filt_foot, dim=1).values
    feet_z_value = torch.max(filt_foot, dim=1).values

    error = torch.square(feet_z_value - feet_z_target)
    reward = torch.exp(-error / std**2)
    return reward


# def feet_distance( env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, min_dist: float, max_dist: float
# ) -> torch.Tensor:
#     """
#     Calculates the reward based on the distance between the feet. Penalize feet get close to each other or too far away.
#     """
#     asset: RigidObject = env.scene[asset_cfg.name]
#     foot_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :3]
#     foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)

#     d_min = torch.clamp(foot_dist - min_dist, -0.5, 0.)
#     # d_max = torch.clamp(foot_dist - max_dist, 0, 0.5)

#     return torch.exp(-torch.abs(d_min) * 100)


def feet_distance_l1(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, min_dist: float, max_dist: float
) -> torch.Tensor:
    """
    Calculates the reward based on the distance between the feet. Penalize feet get close to each other or too far away.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :3]
    foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)

    d_min = torch.clamp(foot_dist - min_dist, -0.5, 0.0)
    # d_max = torch.clamp(foot_dist - max_dist, 0, 0.5)

    return torch.exp(-torch.abs(d_min) * 100)


def joint_torques_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Penalize joint torques on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.linalg.norm((asset.data.applied_torque), dim=1)


def torque_applied_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, std: float
) -> torch.Tensor:
    """
    Calculates the L2 norm of the torques applied to the joints of the robot.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    torques = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.exp(-torch.norm(torques, dim=1) / std)


def center_of_mass_deviation_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, std: float
) -> torch.Tensor:
    """
    Calculates the L2 norm of the deviation of the center of mass from the center of two foot.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    com_xy = asset.data.root_pos_w[:, :2]
    foot_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :3]
    foot_center = torch.mean(
        foot_pos, dim=1
    ).squeeze()  # from (batch, 2, 3) to (batch, 3)
    foot_center_xy = foot_center[:, :2]
    return torch.exp(-torch.norm(com_xy - foot_center_xy, dim=1) / std)


def shoulder_center_deviation_foot_center_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, std: float
) -> torch.Tensor:
    """
    Calculates the L2 norm of the deviation of the center of the shoulder from the center of two foot.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    shoulder_foot_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :3]
    shoulder_pos = shoulder_foot_pos[:, :2, :]
    shoulder_pos_center_xy = torch.mean(shoulder_pos, dim=1).squeeze()[:, :2]
    foot_pos = shoulder_foot_pos[:, 2:, :]
    foot_center_xy = torch.mean(foot_pos, dim=1).squeeze()[:, :2]
    return torch.exp(-torch.norm(shoulder_pos_center_xy - foot_center_xy, dim=1) / std)
