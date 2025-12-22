from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ImitationRLEnv


def reference_joint_pos(
    env: ImitationRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """The joint positions of the asset.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    # extract the used quantities (to enable type-hinting)
    ref_joint_pos = env.current_reference.get("joint_pos")

    return ref_joint_pos[..., asset_cfg.joint_ids]  # type: ignore


def reference_joint_vel(
    env: ImitationRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """The joint positions of the asset.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    # extract the used quantities (to enable type-hinting)
    ref_joint_pos = env.current_reference.get("joint_pos")

    return ref_joint_pos[..., asset_cfg.joint_ids]  # type: ignore


def reference_root_lin_vel(
    env: ImitationRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """The root linear velocity of the asset.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    ref_root_lin_vel = env.current_reference.get("root_lin_vel")
    return ref_root_lin_vel


def reference_root_ang_vel(
    env: ImitationRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """The root angular velocity of the asset.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    ref_root_ang_vel = env.current_reference.get("root_ang_vel")
    return ref_root_ang_vel


def reference_root_pos(
    env: ImitationRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """The root position of the asset.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    ref_root_pos = env.current_reference.get("root_pos")
    return ref_root_pos


def reference_root_quat(
    env: ImitationRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """The root quaternion of the asset.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    ref_root_quat = env.current_reference.get("root_quat")
    return ref_root_quat
