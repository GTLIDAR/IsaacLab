from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ImitationRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import (
    quat_apply,
    quat_apply_inverse,
    quat_error_magnitude,
    quat_inv,
    quat_mul,
    wrap_to_pi,
)


def _print_tracking_debug(
    term_name: str,
    reward: torch.Tensor,
    error: torch.Tensor,
    actual: torch.Tensor,
    reference: torch.Tensor,
):
    """Compact debug print for tracking terms."""
    return
    if reward.numel() == 0:
        return
    env0 = 0
    print(
        f"{term_name}: reward(mean/min/max)=({reward.mean().item():.4f}, "
        f"{reward.min().item():.4f}, {reward.max().item():.4f}) "
        f"error(mean/max)=({error.mean().item():.6f}, {error.max().item():.6f})"
    )
    print(
        f"{term_name}: env0 actual={actual[env0].detach().cpu()} "
        f"reference={reference[env0].detach().cpu()} "
        f"delta={(actual[env0] - reference[env0]).detach().cpu()}"
    )


def _resolve_reference_body_indices(
    env: ImitationRLEnv, reference_body_names: Sequence[str], device: torch.device
) -> torch.Tensor:
    """Map reference body names to indices in the replayed trajectory body arrays."""
    all_reference_body_names = getattr(env, "reference_body_names", None) or []
    if len(all_reference_body_names) == 0:
        raise RuntimeError(
            "Reference body names are unavailable in the environment metadata. "
            "Ensure dataset zarr metadata contains `body_names`."
        )

    if not hasattr(env, "_reference_body_index_cache"):
        env._reference_body_index_cache = {}  # type: ignore[attr-defined]
    cache_key = tuple(reference_body_names)
    if cache_key in env._reference_body_index_cache:  # type: ignore[attr-defined]
        return env._reference_body_index_cache[cache_key]  # type: ignore[attr-defined]

    lookup = {name: idx for idx, name in enumerate(all_reference_body_names)}
    lookup_lower = {name.lower(): idx for idx, name in enumerate(all_reference_body_names)}

    def _find_one(name: str) -> int:
        if name in lookup:
            return lookup[name]
        if name.lower() in lookup_lower:
            return lookup_lower[name.lower()]
        simplified = name.replace("_link", "")
        if simplified in lookup:
            return lookup[simplified]
        if simplified.lower() in lookup_lower:
            return lookup_lower[simplified.lower()]
        raise KeyError(name)

    try:
        ref_indices = [_find_one(name) for name in reference_body_names]
    except KeyError as exc:
        missing_name = str(exc).strip("'")
        raise KeyError(
            f"Reference body '{missing_name}' not found in replay metadata. "
            f"First 20 available names: {all_reference_body_names[:20]}"
        ) from exc

    ref_indices_t = torch.tensor(ref_indices, dtype=torch.long, device=device)
    env._reference_body_index_cache[cache_key] = ref_indices_t  # type: ignore[attr-defined]
    return ref_indices_t


def _relative_pose_from_bodies(body_pos: torch.Tensor, body_quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute relative positions and quaternions against the first body in the list."""
    main_pos = body_pos[:, :1, :]
    rel_pos = body_pos[:, 1:, :] - main_pos

    main_quat = body_quat[:, :1, :].expand_as(body_quat[:, 1:, :]).reshape(-1, 4)
    child_quat = body_quat[:, 1:, :].reshape(-1, 4)
    rel_quat = quat_mul(quat_inv(main_quat), child_quat).reshape(body_quat.shape[0], -1, 4)
    return rel_pos, rel_quat


def _relative_velocity_from_bodies(
    body_quat: torch.Tensor, body_ang_vel: torch.Tensor, body_lin_vel: torch.Tensor
) -> torch.Tensor:
    """Compute relative 6D velocities (ang then lin) in the main-body local frame."""
    main_quat = body_quat[:, :1, :].expand_as(body_quat[:, 1:, :])
    main_ang = body_ang_vel[:, :1, :]
    main_lin = body_lin_vel[:, :1, :]
    child_ang = body_ang_vel[:, 1:, :]
    child_lin = body_lin_vel[:, 1:, :]

    main_quat_flat = main_quat.reshape(-1, 4)
    rel_lin = quat_apply_inverse(main_quat_flat, (main_lin - child_lin).reshape(-1, 3)).reshape(
        body_quat.shape[0], -1, 3
    )
    child_ang_main = quat_apply_inverse(main_quat_flat, child_ang.reshape(-1, 3)).reshape(body_quat.shape[0], -1, 3)
    main_ang_main = quat_apply_inverse(main_quat_flat, main_ang.expand_as(child_ang).reshape(-1, 3)).reshape(
        body_quat.shape[0], -1, 3
    )
    rel_ang = child_ang_main - main_ang_main
    return torch.cat([rel_ang, rel_lin], dim=-1)


def joint_pos_target_l2(env: ImitationRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
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
    qpos_actual: torch.Tensor = env.scene[asset_cfg.name].data.joint_pos[..., asset_cfg.joint_ids]
    # Get reference qpos from the dataset (reference order)
    qpos_reference = env.get_reference_data(key="joint_pos", joint_indices=asset_cfg.joint_ids)

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
    qvel_actual: torch.Tensor = env.scene[asset_cfg.name].data.joint_vel[..., asset_cfg.joint_ids]
    # Get reference qvel from the dataset (reference order)
    qvel_reference = env.get_reference_data(key="joint_vel", joint_indices=asset_cfg.joint_ids)

    # Compute squared L2 error
    squared_error = torch.sum((qvel_actual - qvel_reference) ** 2, dim=1)

    # Apply gaussian kernel: exp(-error^2 / (2 * sigma^2))
    gaussian_reward = torch.exp(-squared_error / (2 * sigma**2))
    return gaussian_reward


def track_root_pos(env: ImitationRLEnv, asset_cfg: SceneEntityCfg | None = None, sigma: float = 0.1) -> torch.Tensor:
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

    root_state_actual = asset.data.root_state_w
    root_pos_actual = root_state_actual[:, :3]  # x, y, z coordinates

    init_pos = env._init_root_pos
    init_quat = env._init_root_quat

    # Get reference root position from the dataset
    root_pos_reference = env.get_reference_data(key="root_pos")

    root_pos_reference = quat_apply(init_quat, root_pos_reference)
    root_pos_reference = root_pos_reference + init_pos

    # Compute squared L2 error between actual and reference root position
    # only penalize the x and y position
    squared_error_xy = torch.sum((root_pos_actual[..., :2] - root_pos_reference[..., :2]) ** 2, dim=1)

    # Apply gaussian kernel: exp(-error^2 / (2 * sigma^2))
    gaussian_reward = torch.exp(-squared_error_xy / (2 * sigma**2))
    _print_tracking_debug(
        term_name="track root pos",
        reward=gaussian_reward,
        error=squared_error_xy,
        actual=root_pos_actual[..., :2],
        reference=root_pos_reference[..., :2],
    )
    return gaussian_reward


def track_root_quat(env: ImitationRLEnv, asset_cfg: SceneEntityCfg | None = None, sigma: float = 0.1) -> torch.Tensor:
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

    # See note in track_root_pos: use root_state_w to avoid stale root buffers.
    root_state_actual = asset.data.root_state_w
    root_quat_actual = root_state_actual[:, 3:7]

    # Transform actual quaternion back to original reference frame
    # q_relative = q_default^-1 * q_actual
    root_quat_actual_relative = quat_mul(quat_inv(env._init_root_quat), root_quat_actual)

    # Get reference root orientation from the dataset (quaternion in w,x,y,z format)
    root_quat_reference = env.get_reference_data(key="root_quat")

    # Compute quaternion error magnitude (angular error in radians)
    angular_error = quat_error_magnitude(root_quat_actual_relative, root_quat_reference)

    # Apply gaussian kernel: exp(-error^2 / (2 * sigma^2))
    # Note: angular_error is already the magnitude, so we square it for the gaussian
    gaussian_reward = torch.exp(-(angular_error**2) / (2 * sigma**2))
    _print_tracking_debug(
        term_name="track root quat",
        reward=gaussian_reward,
        error=angular_error,
        actual=root_quat_actual_relative,
        reference=root_quat_reference,
    )
    return gaussian_reward


def track_root_ang(env: ImitationRLEnv, asset_cfg: SceneEntityCfg | None = None, sigma: float = 0.1) -> torch.Tensor:
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
    root_quat_actual_relative = quat_mul(quat_inv(asset.data.default_root_state[..., 3:7]), root_quat_actual)

    # Get reference root orientation from the dataset (quaternion in w,x,y,z format)
    root_quat_reference = env.get_reference_data(key="root_quat")

    # Compute quaternion error magnitude (angular error in radians)
    angular_error = quat_error_magnitude(root_quat_actual_relative, root_quat_reference)

    # Apply gaussian kernel: exp(-error^2 / (2 * sigma^2))
    # Note: angular_error is already the magnitude, so we square it for the gaussian
    gaussian_reward = torch.exp(-(angular_error**2) / (2 * sigma**2))

    print(f"track root ang vel: gaussian_reward: {gaussian_reward.shape, gaussian_reward}")
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

    # See note in track_root_pos: use root_state_w to avoid stale root buffers.
    root_state_actual = asset.data.root_state_w
    root_quat_actual = root_state_actual[:, 3:7]
    root_lin_vel_actual_w = root_state_actual[:, 7:10]
    root_lin_vel_actual_b = quat_apply_inverse(root_quat_actual, root_lin_vel_actual_w)

    init_quat = env._init_root_quat

    # Get reference root linear velocity from the dataset
    root_lin_vel_reference = env.get_reference_data(key="root_lin_vel")
    root_lin_vel_reference_w = quat_apply(init_quat, root_lin_vel_reference)
    # Compare in robot body frame to avoid coupling this term with random reset yaw.
    root_lin_vel_reference_b = quat_apply_inverse(root_quat_actual, root_lin_vel_reference_w)

    # Track horizontal velocity only (xy). Vertical velocity is already regularized by lin_vel_z_l2.
    squared_error = torch.sum((root_lin_vel_actual_b[..., :2] - root_lin_vel_reference_b[..., :2]) ** 2, dim=-1)

    # Apply gaussian kernel: exp(-error^2 / (2 * sigma^2))
    gaussian_reward = torch.exp(-squared_error / (2 * sigma**2))

    _print_tracking_debug(
        term_name="track root lin vel",
        reward=gaussian_reward,
        error=squared_error,
        actual=root_lin_vel_actual_b[..., :2],
        reference=root_lin_vel_reference_b[..., :2],
    )
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

    # See note in track_root_pos: use root_state_w to avoid stale root buffers.
    root_state_actual = asset.data.root_state_w
    root_ang_vel_actual = root_state_actual[:, 10:13]

    init_quat = env._init_root_quat

    # Get reference root angular velocity from the dataset
    root_ang_vel_reference = env.get_reference_data(key="root_ang_vel")

    root_ang_vel_reference = quat_apply(init_quat, root_ang_vel_reference)

    # Angular velocity is a 3D vector, so compare with L2 distance (not quaternion distance).
    squared_error = torch.sum((root_ang_vel_actual - root_ang_vel_reference) ** 2, dim=-1)

    # Apply gaussian kernel: exp(-error^2 / (2 * sigma^2))
    gaussian_reward = torch.exp(-squared_error / (2 * sigma**2))

    _print_tracking_debug(
        term_name="track root ang vel",
        reward=gaussian_reward,
        error=squared_error,
        actual=root_ang_vel_actual,
        reference=root_ang_vel_reference,
    )
    return gaussian_reward


def track_relative_body_pos(
    env: ImitationRLEnv,
    asset_cfg: SceneEntityCfg | None = None,
    reference_body_names: Sequence[str] = (),
    sigma: float = 0.1,
) -> torch.Tensor:
    """Track relative body positions against reference `xpos` (loco-style rpos term)."""
    asset: Articulation = env.scene[asset_cfg.name]
    body_ids = torch.as_tensor(asset_cfg.body_ids, dtype=torch.long, device=asset.data.body_link_pos_w.device)
    if body_ids.numel() < 2:
        raise ValueError("track_relative_body_pos requires at least 2 body ids.")
    if len(reference_body_names) != int(body_ids.numel()):
        raise ValueError("reference_body_names must match the number of selected body names.")

    ref_body_ids = _resolve_reference_body_indices(env, reference_body_names, body_ids.device)
    actual_pos = asset.data.body_link_pos_w[:, body_ids, :]
    ref_pos = env.get_reference_data(key="xpos")[..., ref_body_ids, :]

    actual_rel_pos, _ = _relative_pose_from_bodies(actual_pos, asset.data.body_link_quat_w[:, body_ids, :])
    ref_quat = env.get_reference_data(key="xquat")[..., ref_body_ids, :]
    ref_rel_pos, _ = _relative_pose_from_bodies(ref_pos, ref_quat)

    squared_error = torch.mean((actual_rel_pos - ref_rel_pos) ** 2, dim=(1, 2))
    return torch.exp(-squared_error / (2 * sigma**2))


def track_relative_body_quat(
    env: ImitationRLEnv,
    asset_cfg: SceneEntityCfg | None = None,
    reference_body_names: Sequence[str] = (),
    sigma: float = 0.1,
) -> torch.Tensor:
    """Track relative body orientations against reference `xquat` (loco-style rquat term)."""
    asset: Articulation = env.scene[asset_cfg.name]
    body_ids = torch.as_tensor(asset_cfg.body_ids, dtype=torch.long, device=asset.data.body_link_pos_w.device)
    if body_ids.numel() < 2:
        raise ValueError("track_relative_body_quat requires at least 2 body ids.")
    if len(reference_body_names) != int(body_ids.numel()):
        raise ValueError("reference_body_names must match the number of selected body names.")

    ref_body_ids = _resolve_reference_body_indices(env, reference_body_names, body_ids.device)
    actual_quat = asset.data.body_link_quat_w[:, body_ids, :]
    ref_quat = env.get_reference_data(key="xquat")[..., ref_body_ids, :]

    _, actual_rel_quat = _relative_pose_from_bodies(asset.data.body_link_pos_w[:, body_ids, :], actual_quat)
    _, ref_rel_quat = _relative_pose_from_bodies(env.get_reference_data(key="xpos")[..., ref_body_ids, :], ref_quat)

    ang_err = quat_error_magnitude(actual_rel_quat.reshape(-1, 4), ref_rel_quat.reshape(-1, 4)).reshape(
        actual_rel_quat.shape[0], -1
    )
    squared_error = torch.mean(ang_err**2, dim=1)
    return torch.exp(-squared_error / (2 * sigma**2))


def track_relative_body_vel(
    env: ImitationRLEnv,
    asset_cfg: SceneEntityCfg | None = None,
    reference_body_names: Sequence[str] = (),
    sigma: float = 0.2,
) -> torch.Tensor:
    """Track relative body 6D velocity against reference `cvel` (loco-style rvel term)."""
    asset: Articulation = env.scene[asset_cfg.name]
    body_ids = torch.as_tensor(asset_cfg.body_ids, dtype=torch.long, device=asset.data.body_link_pos_w.device)
    if body_ids.numel() < 2:
        raise ValueError("track_relative_body_vel requires at least 2 body ids.")
    if len(reference_body_names) != int(body_ids.numel()):
        raise ValueError("reference_body_names must match the number of selected body names.")

    ref_body_ids = _resolve_reference_body_indices(env, reference_body_names, body_ids.device)

    actual_quat = asset.data.body_link_quat_w[:, body_ids, :]
    actual_ang_vel = asset.data.body_ang_vel_w[:, body_ids, :]
    actual_lin_vel = asset.data.body_lin_vel_w[:, body_ids, :]
    actual_rel_vel = _relative_velocity_from_bodies(actual_quat, actual_ang_vel, actual_lin_vel)

    ref_xquat = env.get_reference_data(key="xquat")[..., ref_body_ids, :]
    ref_cvel = env.get_reference_data(key="cvel")[..., ref_body_ids, :]
    ref_ang_vel = ref_cvel[..., :3]
    ref_lin_vel = ref_cvel[..., 3:]
    ref_rel_vel = _relative_velocity_from_bodies(ref_xquat, ref_ang_vel, ref_lin_vel)

    squared_error = torch.mean((actual_rel_vel - ref_rel_vel) ** 2, dim=(1, 2))
    return torch.exp(-squared_error / (2 * sigma**2))
