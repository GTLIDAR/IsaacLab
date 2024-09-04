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
    env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
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

    phase = env.get_phase()
    sin_pos = torch.sin(2 * torch.pi * phase)
    stance_mask = torch.zeros((env.num_envs, 2), device=env.device)
    stance_mask[:, 0] = sin_pos >= 0
    stance_mask[:, 1] = sin_pos < 0
    stance_mask[torch.abs(sin_pos) < 0.1] = 1

    reward = torch.where(contacts == stance_mask, 1, -0.3)
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
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    std: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(
        asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height
    )
    foot_velocity_tanh = torch.tanh(
        tanh_mult
        * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)
    )
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)


def track_foot_height_l1(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    std: float,
    tanh_mult: float,
) -> torch.Tensor:
    """"""

    def height_target(t: torch.Tensor):
        assert t.shape[0] == env.num_envs
        a5, a4, a3, a2, a1, a0 = [9.6, 12.0, -18.8, 5.0, 0.1, 0.0]
        return a5 * t**5 + a4 * t**4 + a3 * t**3 + a2 * t**2 + a1 * t + a0

    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]

    phase = env.get_phase()

    sin_pos = torch.sin(2 * torch.pi * phase)
    stance_mask = torch.zeros((env.num_envs, 2), device=env.device)
    stance_mask[:, 0] = sin_pos >= 0
    stance_mask[:, 1] = sin_pos < 0
    stance_mask[torch.abs(sin_pos) < 0.1] = 1
    swing_mask = 1 - stance_mask

    filt_foot = torch.where(swing_mask == 1, foot_z, torch.zeros_like(foot_z))

    phase_mod = torch.fmod(2 * torch.pi * phase, 0.5)
    feet_z_target = height_target(phase_mod)
    feet_z_value = torch.sum(filt_foot, dim=1)

    reward = torch.exp(-(torch.square(feet_z_value - feet_z_target)))
    # print(reward)
    return reward
