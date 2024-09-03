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




def reward_feet_contact_number(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"))-> torch.Tensor:
    """
    Calculates a reward based on the number of feet contacts aligning with the gait phase. 
    Rewards or penalizes depending on whether the foot contact matches the expected gait phase.
    """
    # print("------------------------")
    # print("contacts:")
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0]
    # print(contacts)
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0

    # print(contacts)
    # print("\n")

    phase = env.get_phase()
    sin_pos = torch.sin(2 * torch.pi * phase)
    stance_mask = torch.zeros((env.num_envs, 2), device=env.device)
    stance_mask[:, 0] = sin_pos >= 0
    stance_mask[:, 1] = sin_pos < 0
    stance_mask[torch.abs(sin_pos) < 0.1] = 1

    # print("stance_mask:")
    # print(stance_mask)
    # print("\n")
    
    reward = torch.where(contacts == stance_mask, 1, -0.3)
    # print("reward:")
    # print(reward)
    # print("------------------------")
    # print("\n")

    return torch.mean(reward, dim=1)


# def reward_feet_clearance(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float, std: float, tanh_mult: float)-> torch.Tensor:
#     """
#     Calculates reward based on the clearance of the swing leg from the ground during movement.
#     Encourages appropriate lift of the feet during the swing phase of the gait.
#     """
#     asset: RigidObject = env.scene[asset_cfg.name]

#     # Compute swing mask
#     phase = env.get_phase()
#     sin_pos = torch.sin(2 * torch.pi * phase)
#     stance_mask = torch.zeros((env.num_envs, 2), device=env.device)
#     stance_mask[:, 0] = sin_pos >= 0
#     stance_mask[:, 1] = sin_pos < 0
#     stance_mask[torch.abs(sin_pos) < 0.1] = 1
#     swing_mask = 1 - stance_mask


#     foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
#     foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
#     reward = foot_z_target_error * foot_velocity_tanh 
#     return torch.exp(-torch.sum(reward, dim=1) / std)


def foot_clearance_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float, std: float, tanh_mult: float
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)