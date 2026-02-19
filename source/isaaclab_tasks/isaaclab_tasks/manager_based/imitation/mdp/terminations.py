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
)


def reference_joint_pos_deviation_too_much(
    env: ImitationRLEnv,
    threshold: float = 0.75,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when average joint-position tracking error is too large."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos_actual = asset.data.joint_pos[:, asset_cfg.joint_ids]
    joint_pos_reference = env.get_reference_data(key="joint_pos", joint_indices=asset_cfg.joint_ids)
    rms_joint_error = torch.sqrt(torch.mean((joint_pos_actual - joint_pos_reference) ** 2, dim=1))
    return rms_joint_error > threshold


def reference_root_position_xy_deviation_too_much(
    env: ImitationRLEnv,
    threshold: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when root XY position diverges from reference trajectory."""
    asset: Articulation = env.scene[asset_cfg.name]
    root_pos_actual = asset.data.root_state_w[:, :3]

    root_pos_reference = env.get_reference_data(key="root_pos")
    root_pos_reference = quat_apply(env._init_root_quat, root_pos_reference) + env._init_root_pos

    xy_error = torch.linalg.norm(root_pos_actual[:, :2] - root_pos_reference[:, :2], dim=1)
    return xy_error > threshold


def reference_root_quat_deviation_too_much(
    env: ImitationRLEnv,
    threshold: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when root orientation error to reference exceeds threshold (radians)."""
    asset: Articulation = env.scene[asset_cfg.name]
    root_quat_actual = asset.data.root_state_w[:, 3:7]
    root_quat_actual_relative = quat_mul(quat_inv(env._init_root_quat), root_quat_actual)
    root_quat_reference = env.get_reference_data(key="root_quat")

    angular_error = quat_error_magnitude(root_quat_actual_relative, root_quat_reference)
    return angular_error > threshold


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


def bad_anchor_pos_z_only(
    env: ImitationRLEnv,
    threshold: float = 0.25,
    anchor_body_name: str = "torso_link",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when anchor z-tracking error exceeds threshold."""
    asset: Articulation = env.scene[asset_cfg.name]
    anchor_idx = asset.body_names.index(anchor_body_name)
    robot_anchor_pos_w = asset.data.body_link_pos_w[:, anchor_idx]

    ref_anchor_id = _resolve_reference_body_indices(env, [anchor_body_name], robot_anchor_pos_w.device)
    ref_anchor_pos = env.get_reference_data(key="xpos")[..., ref_anchor_id, :][:, 0, :]
    ref_anchor_pos_w = quat_apply(env._init_root_quat, ref_anchor_pos) + env._init_root_pos
    return torch.abs(ref_anchor_pos_w[:, 2] - robot_anchor_pos_w[:, 2]) > threshold


def bad_anchor_ori(
    env: ImitationRLEnv,
    threshold: float = 0.8,
    anchor_body_name: str = "torso_link",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when anchor orientation mismatch exceeds threshold."""
    asset: Articulation = env.scene[asset_cfg.name]
    anchor_idx = asset.body_names.index(anchor_body_name)
    robot_anchor_quat_w = asset.data.body_link_quat_w[:, anchor_idx]

    ref_anchor_id = _resolve_reference_body_indices(env, [anchor_body_name], robot_anchor_quat_w.device)
    ref_anchor_quat = env.get_reference_data(key="xquat")[..., ref_anchor_id, :][:, 0, :]
    ref_anchor_quat_w = quat_mul(env._init_root_quat, ref_anchor_quat)

    reference_projected_gravity_b = quat_apply_inverse(ref_anchor_quat_w, asset.data.GRAVITY_VEC_W)
    robot_projected_gravity_b = quat_apply_inverse(robot_anchor_quat_w, asset.data.GRAVITY_VEC_W)
    return (reference_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]).abs() > threshold


def bad_reference_body_pos_z_only(
    env: ImitationRLEnv,
    threshold: float = 0.25,
    reference_body_names: Sequence[str] = (),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when any selected body z error to reference exceeds threshold."""
    asset: Articulation = env.scene[asset_cfg.name]
    body_ids = torch.as_tensor(asset_cfg.body_ids, dtype=torch.long, device=asset.data.body_link_pos_w.device)
    if len(reference_body_names) != int(body_ids.numel()):
        raise ValueError("reference_body_names must match the number of selected body names.")

    ref_body_ids = _resolve_reference_body_indices(env, reference_body_names, body_ids.device)
    ref_pos = env.get_reference_data(key="xpos")[..., ref_body_ids, :]
    num_envs = ref_pos.shape[0]
    num_bodies = ref_pos.shape[1]
    init_quat = env._init_root_quat.unsqueeze(1).expand(-1, num_bodies, -1).reshape(-1, 4)
    ref_pos_w = quat_apply(init_quat, ref_pos.reshape(-1, 3)).reshape(num_envs, num_bodies, 3)
    ref_pos_w = ref_pos_w + env._init_root_pos.unsqueeze(1)

    body_pos_actual = asset.data.body_link_pos_w[:, body_ids, :]
    z_error = torch.abs(ref_pos_w[..., 2] - body_pos_actual[..., 2])
    return torch.any(z_error > threshold, dim=-1)
