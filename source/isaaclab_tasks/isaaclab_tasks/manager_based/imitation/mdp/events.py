from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ImitationRLEnv


def reset_joints_to_reference(
    env: ImitationRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the robot joints to the reference joint positions and velocities."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # get reference joint state
    reference_joint_pos = env.current_reference["joint_pos"]
    reference_joint_vel = env.current_reference["joint_vel"]

    default_joint_pos = asset.data.default_joint_pos[env_ids]
    default_joint_vel = asset.data.default_joint_vel[env_ids]

    joint_pos = reference_joint_pos[env_ids].clone()
    joint_vel = reference_joint_vel[env_ids].clone()
    joint_pos_col_mask = torch.isnan(joint_pos).any(dim=0)
    joint_vel_col_mask = torch.isnan(joint_vel).any(dim=0)
    joint_pos[..., joint_pos_col_mask] = default_joint_pos[..., joint_pos_col_mask]
    joint_vel[..., joint_vel_col_mask] = default_joint_vel[..., joint_vel_col_mask]

    # set into the physics simulation
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
