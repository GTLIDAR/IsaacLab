from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ImitationRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import (
    matrix_from_quat,
    quat_apply,
    quat_mul,
    subtract_frame_transforms,
)


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
    ref_joint_vel = env.current_reference.get("joint_vel")

    return ref_joint_vel[..., asset_cfg.joint_ids]  # type: ignore


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


def _resolve_reference_body_indices(
    env: ImitationRLEnv, reference_body_names: Sequence[str], device: torch.device
) -> torch.Tensor:
    """Map reference body names to indices in replay metadata."""
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


def _reference_body_pose_w(
    env: ImitationRLEnv, reference_body_names: Sequence[str]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return reference body positions/quaternions in world frame."""
    device = env.device
    ref_body_ids = _resolve_reference_body_indices(env, reference_body_names, device)
    ref_pos = env.get_reference_data(key="xpos")[..., ref_body_ids, :]
    ref_quat = env.get_reference_data(key="xquat")[..., ref_body_ids, :]

    num_envs, num_bodies = ref_pos.shape[0], ref_pos.shape[1]
    init_quat = env._init_root_quat.unsqueeze(1).expand(-1, num_bodies, -1).reshape(-1, 4)

    ref_pos_w = quat_apply(init_quat, ref_pos.reshape(-1, 3)).reshape(num_envs, num_bodies, 3)
    ref_pos_w = ref_pos_w + env._init_root_pos.unsqueeze(1)
    ref_quat_w = quat_mul(init_quat, ref_quat.reshape(-1, 4)).reshape(num_envs, num_bodies, 4)
    return ref_pos_w, ref_quat_w


def reference_motion_command(
    env: ImitationRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reference joint position/velocity command vector (qpos || qvel)."""
    ref_joint_pos = env.get_reference_data(key="joint_pos", joint_indices=asset_cfg.joint_ids)
    ref_joint_vel = env.get_reference_data(key="joint_vel", joint_indices=asset_cfg.joint_ids)
    return torch.cat([ref_joint_pos, ref_joint_vel], dim=-1)


def reference_anchor_pos_b(
    env: ImitationRLEnv,
    anchor_body_name: str = "torso_link",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reference anchor position expressed in the robot-anchor frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    anchor_body_idx = asset.body_names.index(anchor_body_name)
    robot_anchor_pos_w = asset.data.body_link_pos_w[:, anchor_body_idx]
    robot_anchor_quat_w = asset.data.body_link_quat_w[:, anchor_body_idx]

    ref_anchor_pos_w, ref_anchor_quat_w = _reference_body_pose_w(env, [anchor_body_name])
    ref_anchor_pos_w = ref_anchor_pos_w[:, 0, :]
    ref_anchor_quat_w = ref_anchor_quat_w[:, 0, :]

    anchor_pos_b, _ = subtract_frame_transforms(
        robot_anchor_pos_w, robot_anchor_quat_w, ref_anchor_pos_w, ref_anchor_quat_w
    )
    return anchor_pos_b


def reference_anchor_ori_b(
    env: ImitationRLEnv,
    anchor_body_name: str = "torso_link",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reference anchor orientation expressed in the robot-anchor frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    anchor_body_idx = asset.body_names.index(anchor_body_name)
    robot_anchor_pos_w = asset.data.body_link_pos_w[:, anchor_body_idx]
    robot_anchor_quat_w = asset.data.body_link_quat_w[:, anchor_body_idx]

    ref_anchor_pos_w, ref_anchor_quat_w = _reference_body_pose_w(env, [anchor_body_name])
    ref_anchor_pos_w = ref_anchor_pos_w[:, 0, :]
    ref_anchor_quat_w = ref_anchor_quat_w[:, 0, :]

    _, anchor_ori_b = subtract_frame_transforms(
        robot_anchor_pos_w, robot_anchor_quat_w, ref_anchor_pos_w, ref_anchor_quat_w
    )
    anchor_ori_b_mat = matrix_from_quat(anchor_ori_b)
    return anchor_ori_b_mat[..., :2].reshape(anchor_ori_b_mat.shape[0], -1)


def robot_body_pos_b(
    env: ImitationRLEnv,
    anchor_body_name: str = "torso_link",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Robot body positions in the robot-anchor frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    anchor_body_idx = asset.body_names.index(anchor_body_name)
    robot_anchor_pos_w = asset.data.body_link_pos_w[:, anchor_body_idx]
    robot_anchor_quat_w = asset.data.body_link_quat_w[:, anchor_body_idx]

    body_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids]
    body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids]
    num_bodies = body_pos_w.shape[1]

    body_pos_b, _ = subtract_frame_transforms(
        robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        body_pos_w,
        body_quat_w,
    )
    return body_pos_b.reshape(env.num_envs, -1)


def robot_body_ori_b(
    env: ImitationRLEnv,
    anchor_body_name: str = "torso_link",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Robot body orientations in the robot-anchor frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    anchor_body_idx = asset.body_names.index(anchor_body_name)
    robot_anchor_pos_w = asset.data.body_link_pos_w[:, anchor_body_idx]
    robot_anchor_quat_w = asset.data.body_link_quat_w[:, anchor_body_idx]

    body_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids]
    body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids]
    num_bodies = body_pos_w.shape[1]

    _, body_ori_b = subtract_frame_transforms(
        robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        body_pos_w,
        body_quat_w,
    )
    body_ori_b_mat = matrix_from_quat(body_ori_b)
    return body_ori_b_mat[..., :2].reshape(body_ori_b_mat.shape[0], -1)
