from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers.command_manager import CommandTerm


def root_height_below_minimum_adaptive(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the asset's root height is below the minimum height.

    Note:
        This is currently only supported for flat terrains, i.e. the minimum height is in the world frame.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    min_foot_height = (
        (asset.data.body_pos_w[:, asset_cfg.body_ids, 2]).min(dim=1).values
    )

    # print(f"asset.data.root_pos_w[:, 2] {asset.data.root_pos_w[:5, 2]}")

    return asset.data.root_pos_w[:, 2] - min_foot_height < minimum_height


def arm_deviation_too_much(
    env: ManagerBasedRLEnv,
    threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the asset's arm is too far from the desired orientation limits.

    This is computed by checking the angle between the projected gravity vector and the z-axis.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return (
        torch.linalg.norm(
            asset.data.joint_pos[:, asset_cfg.joint_ids]
            - asset.data.default_joint_pos[:, asset_cfg.joint_ids],
            dim=1,
        )
        > threshold
    )


def has_nan(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the asset has NaN values."""
    obs_buf: dict[str, torch.Tensor | dict[str, torch.Tensor]] = env.obs_buf
    policy_obs = obs_buf["observation"]
    if isinstance(policy_obs, dict):
        return torch.isnan(policy_obs["joint_pos"]).sum(-1) > 0
    else:
        return torch.isnan(policy_obs).sum(-1) > 0
