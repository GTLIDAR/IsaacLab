# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi, quat_error_magnitude, quat_mul, quat_inv

from isaaclab.envs import ImitationRLEnv


def joint_pos_target_l2(
    env: ImitationRLEnv, target: float, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # wrap the joint positions to (-pi, pi)
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    # compute the reward
    return torch.sum(torch.square(joint_pos - target), dim=1)


def track_joint_reference(
    env: ImitationRLEnv,
    asset_cfg: SceneEntityCfg | None = None,
    sigma: float = 0.25,
) -> torch.Tensor:
    """
    Reward for joint position imitation using a gaussian kernel.

    Args:
        env: The environment instance.
        asset_cfg: Scene entity configuration for the robot. If None, uses default robot config.
        sigma: Standard deviation for the gaussian kernel (controls reward sharpness).

    Returns:
        Tensor of shape (num_envs,) with the gaussian reward for each environment.
    """

    # Use default asset config if none provided
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")  # type: ignore

    # Get actual qpos from the robot (IsaacLab order)
    qpos_actual: torch.Tensor = env.scene[asset_cfg.name].data.joint_pos[
        ..., asset_cfg.joint_ids
    ]
    # Get reference qpos from the dataset (reference order)
    qpos_reference = env.get_reference_data(
        key="joint_pos", joint_indices=asset_cfg.joint_ids
    )
    assert qpos_reference.shape == qpos_actual.shape

    # Compute mapping between IsaacLab and reference order
    isaaclab_joint_names = env.scene[asset_cfg.name].joint_names
    reference_joint_names = asset_cfg.joint_names
    assert reference_joint_names is not None, (
        "reference_joint_names must be provided in asset_cfg"
    )

    # Find common joints and their indices in both orders
    common_names = [
        name for name in reference_joint_names if name in isaaclab_joint_names
    ]
    idx_actual = [isaaclab_joint_names.index(name) for name in common_names]
    idx_reference = [reference_joint_names.index(name) for name in common_names]

    # Select only the common joints
    qpos_actual_common = qpos_actual
    qpos_reference_common = qpos_reference
    # Compute squared L2 error
    squared_error = torch.sum((qpos_actual_common - qpos_reference_common) ** 2, dim=1)

    # Apply gaussian kernel: exp(-error^2 / (2 * sigma^2))
    gaussian_reward = torch.exp(-squared_error / (2 * sigma**2))

    return gaussian_reward


def track_root_pos(
    env: ImitationRLEnv, asset_cfg: SceneEntityCfg | None = None, sigma: float = 0.1
) -> torch.Tensor:
    """
    Reward for root position imitation using a gaussian kernel.

    Args:
        env: The environment instance.
        asset_cfg: Scene entity configuration for the robot. If None, uses default robot config.
        sigma: Standard deviation for the gaussian kernel (controls reward sharpness).

    Returns:
        Tensor of shape (num_envs,) with the gaussian reward for each environment.
    """
    pass
    # Use default asset config if none provided
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")  # type: ignore

    # Extract the robot
    asset: Articulation = env.scene[asset_cfg.name]

    # Get actual root position (typically the base/pelvis position)
    root_pos_actual = asset.data.root_pos_w[:, :3]  # x, y, z coordinates

    # remove the default root position
    root_pos_actual: torch.Tensor = (
        root_pos_actual - asset.data.default_root_state[..., :3]
    )

    # Get reference root position from the dataset
    root_pos_reference = env.get_reference_data(key="root_pos")

    assert root_pos_actual.shape == root_pos_reference.shape, (
        root_pos_actual.shape,
        root_pos_reference.shape,
    )

    # Compute squared L2 error between actual and reference root position
    squared_error = torch.sum((root_pos_actual - root_pos_reference) ** 2, dim=1)

    # Apply gaussian kernel: exp(-error^2 / (2 * sigma^2))
    gaussian_reward = torch.exp(-squared_error / (2 * sigma**2))

    return gaussian_reward


def track_root_ang(
    env: ImitationRLEnv, asset_cfg: SceneEntityCfg | None = None, sigma: float = 0.1
) -> torch.Tensor:
    """
    Reward for root orientation imitation using a gaussian kernel.

    Args:
        env: The environment instance.
        asset_cfg: Scene entity configuration for the robot. If None, uses default robot config.
        sigma: Standard deviation for the gaussian kernel (controls reward sharpness).

    Returns:
        Tensor of shape (num_envs,) with the gaussian reward for each environment.
    """
    # Use default asset config if none provided
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")  # type: ignore

    # Extract the robot
    asset: Articulation = env.scene[asset_cfg.name]

    # Get actual root orientation (quaternion in w,x,y,z format)
    root_quat_actual = asset.data.root_quat_w

    # Transform actual quaternion back to original reference frame
    # q_relative = q_default^-1 * q_actual
    root_quat_actual_relative = quat_mul(
        quat_inv(asset.data.default_root_state[..., 3:7]), root_quat_actual
    )

    # Get reference root orientation from the dataset (quaternion in w,x,y,z format)
    root_quat_reference = env.get_reference_data(key="root_quat")

    assert root_quat_actual_relative.shape == root_quat_reference.shape, (
        root_quat_actual_relative.shape,
        root_quat_reference.shape,
    )

    # Compute quaternion error magnitude (angular error in radians)
    angular_error = quat_error_magnitude(root_quat_actual_relative, root_quat_reference)

    # Apply gaussian kernel: exp(-error^2 / (2 * sigma^2))
    # Note: angular_error is already the magnitude, so we square it for the gaussian
    gaussian_reward = torch.exp(-(angular_error**2) / (2 * sigma**2))

    return gaussian_reward
