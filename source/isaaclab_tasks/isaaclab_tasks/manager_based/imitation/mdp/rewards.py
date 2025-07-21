from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import (
    wrap_to_pi,
    quat_error_magnitude,
    quat_mul,
    quat_inv,
    quat_apply,
)

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


def track_joint_pos(
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

    # Get actual qpos from the robot (IsaacLab order)
    qpos_actual: torch.Tensor = env.scene[asset_cfg.name].data.joint_pos[
        ..., asset_cfg.joint_ids
    ]
    # Get reference qpos from the dataset (reference order)
    qpos_reference = env.get_reference_data(
        key="joint_pos", joint_indices=asset_cfg.joint_ids
    )

    # Compute squared L2 error
    squared_error = torch.sum((qpos_actual - qpos_reference) ** 2, dim=1)

    # Apply gaussian kernel: exp(-error^2 / (2 * sigma^2))
    gaussian_reward = torch.exp(-squared_error / (2 * sigma**2))

    return gaussian_reward


def track_joint_vel(
    env: ImitationRLEnv,
    asset_cfg: SceneEntityCfg | None = None,
    sigma: float = 0.25,
) -> torch.Tensor:
    """
    Reward for joint velocity imitation using a gaussian kernel.

    Args:
        env: The environment instance.
        asset_cfg: Scene entity configuration for the robot. If None, uses default robot config.
        sigma: Standard deviation for the gaussian kernel (controls reward sharpness).

    Returns:
        Tensor of shape (num_envs,) with the gaussian reward for each environment.
    """

    # Get actual qpos from the robot (IsaacLab order)
    qvel_actual: torch.Tensor = env.scene[asset_cfg.name].data.joint_vel[
        ..., asset_cfg.joint_ids
    ]
    # Get reference qvel from the dataset (reference order)
    qvel_reference = env.get_reference_data(
        key="joint_vel", joint_indices=asset_cfg.joint_ids
    )

    # Compute squared L2 error
    squared_error = torch.sum((qvel_actual - qvel_reference) ** 2, dim=1)

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
    # Extract the robot
    asset: Articulation = env.scene[asset_cfg.name]

    # Get actual root position (typically the base/pelvis position)
    root_pos_actual = asset.data.root_pos_w[:, :3]  # x, y, z coordinates

    init_pos = env._init_root_pos
    init_quat = env._init_root_quat

    # Get reference root position from the dataset
    root_pos_reference = env.get_reference_data(key="root_pos")

    root_pos_reference = quat_apply(init_quat, root_pos_reference)
    root_pos_reference = root_pos_reference + init_pos

    # Compute squared L2 error between actual and reference root position
    # only penalize the x and y position
    squared_error_xy = torch.sum(
        (root_pos_actual[..., :2] - root_pos_reference[..., :2]) ** 2, dim=1
    )

    # Apply gaussian kernel: exp(-error^2 / (2 * sigma^2))
    gaussian_reward = torch.exp(-squared_error_xy / (2 * sigma**2))

    return gaussian_reward


def track_root_quat(
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
    # Extract the robot
    asset: Articulation = env.scene[asset_cfg.name]

    # Get actual root orientation (quaternion in w,x,y,z format)
    root_quat_actual = asset.data.root_quat_w

    # Transform actual quaternion back to original reference frame
    # q_relative = q_default^-1 * q_actual
    root_quat_actual_relative = quat_mul(
        quat_inv(env._init_root_quat), root_quat_actual
    )

    # Get reference root orientation from the dataset (quaternion in w,x,y,z format)
    root_quat_reference = env.get_reference_data(key="root_quat")

    # Compute quaternion error magnitude (angular error in radians)
    angular_error = quat_error_magnitude(root_quat_actual_relative, root_quat_reference)

    # Apply gaussian kernel: exp(-error^2 / (2 * sigma^2))
    # Note: angular_error is already the magnitude, so we square it for the gaussian
    gaussian_reward = torch.exp(-(angular_error**2) / (2 * sigma**2))

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

    # Compute quaternion error magnitude (angular error in radians)
    angular_error = quat_error_magnitude(root_quat_actual_relative, root_quat_reference)

    # Apply gaussian kernel: exp(-error^2 / (2 * sigma^2))
    # Note: angular_error is already the magnitude, so we square it for the gaussian
    gaussian_reward = torch.exp(-(angular_error**2) / (2 * sigma**2))

    return gaussian_reward


def track_root_lin_vel(
    env: ImitationRLEnv, asset_cfg: SceneEntityCfg | None = None, sigma: float = 0.1
) -> torch.Tensor:
    """
    Reward for root linear velocity imitation using a gaussian kernel.

    Args:
        env: The environment instance.
        asset_cfg: Scene entity configuration for the robot. If None, uses default robot config.
        sigma: Standard deviation for the gaussian kernel (controls reward sharpness).

    Returns:
        Tensor of shape (num_envs,) with the gaussian reward for each environment.
    """
    # Extract the robot
    asset: Articulation = env.scene[asset_cfg.name]

    # Get actual root linear velocity (typically the base/pelvis position)
    root_lin_vel_actual = asset.data.root_lin_vel_w[:, :3]  # x, y, z coordinates

    init_pos = env._init_root_pos
    init_quat = env._init_root_quat

    # Get reference root linear velocity from the dataset
    root_lin_vel_reference = env.get_reference_data(key="root_lin_vel")

    root_lin_vel_reference = quat_apply(init_quat, root_lin_vel_reference)

    # Compute squared L2 error between actual and reference root position
    # only penalize the x and y position
    squared_error_xy = torch.sum(
        (root_lin_vel_actual[..., :2] - root_lin_vel_reference[..., :2]) ** 2, dim=-1
    )

    # Apply gaussian kernel: exp(-error^2 / (2 * sigma^2))
    gaussian_reward = torch.exp(-squared_error_xy / (2 * sigma**2))

    return gaussian_reward


def track_root_ang_vel(
    env: ImitationRLEnv, asset_cfg: SceneEntityCfg | None = None, sigma: float = 0.1
) -> torch.Tensor:
    """
    Reward for root angular velocity imitation using a gaussian kernel.

    Args:
        env: The environment instance.
        asset_cfg: Scene entity configuration for the robot. If None, uses default robot config.
        sigma: Standard deviation for the gaussian kernel (controls reward sharpness).

    Returns:
        Tensor of shape (num_envs,) with the gaussian reward for each environment.
    """
    # Extract the robot
    asset: Articulation = env.scene[asset_cfg.name]

    # Get actual root angular velocity (typically the base/pelvis position)
    root_ang_vel_actual = asset.data.root_ang_vel_w

    init_pos = env._init_root_pos
    init_quat = env._init_root_quat

    # Get reference root angular velocity from the dataset
    root_ang_vel_reference = env.get_reference_data(key="root_ang_vel")

    root_ang_vel_reference = quat_apply(init_quat, root_ang_vel_reference)

    # Compute squared L2 error between actual and reference root position
    # only penalize the x and y position
    ang_vel_error = quat_error_magnitude(root_ang_vel_actual, root_ang_vel_reference)

    # Apply gaussian kernel: exp(-error^2 / (2 * sigma^2))
    gaussian_reward = torch.exp(-(ang_vel_error**2) / (2 * sigma**2))

    return gaussian_reward
