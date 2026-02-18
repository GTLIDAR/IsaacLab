# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Replay reference trajectories and optionally record video."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay imitation reference data.")
parser.add_argument(
    "--video", action="store_true", default=False, help="Record videos during replay."
)
parser.add_argument(
    "--video_length",
    type=int,
    default=200,
    help="Length of the recorded video (in steps).",
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--num_envs", type=int, default=None, help="Number of environments to simulate."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent",
    type=str,
    default="rsl_rl_cfg_entry_point",
    help="Agent config entry point (unused).",
)
parser.add_argument(
    "--seed", type=int, default=None, help="Seed used for the environment"
)
parser.add_argument(
    "--real-time",
    action="store_true",
    default=False,
    help="Run in real-time, if possible.",
)
parser.add_argument(
    "--max_steps",
    type=int,
    default=1000,
    help="Maximum replay steps when not recording video.",
)
parser.add_argument(
    "--save_torchrl_rb",
    action="store_true",
    default=False,
    help="Save replay transitions to a TorchRL TensorDict replay buffer (memmap).",
)
parser.add_argument(
    "--torchrl_rb_dir",
    type=str,
    default=None,
    help="Output directory for the TorchRL replay buffer (defaults to <log_dir>/torchrl_rb).",
)
parser.add_argument(
    "--lerobot_dir",
    type=str,
    default=None,
    help="Optional output directory for a LeRobot dataset export.",
)
parser.add_argument(
    "--save_lerobot",
    action="store_true",
    default=False,
    help="Save replay data as a LeRobot dataset (defaults to <log_dir>/lerobot).",
)
parser.add_argument(
    "--debug_reference_match",
    action="store_true",
    default=False,
    help=(
        "Debug replay correctness by comparing replayed robot root/joint state against "
        "the transformed reference at runtime."
    ),
)
parser.add_argument(
    "--debug_reference_interval",
    type=int,
    default=50,
    help="Print debug replay-match stats every N steps.",
)
parser.add_argument(
    "--debug_reference_max_envs",
    type=int,
    default=4,
    help="Maximum number of failing envs to print per debug report.",
)
parser.add_argument(
    "--debug_reference_pos_tol",
    type=float,
    default=5.0e-4,
    help="Failure threshold for root position error norm (meters).",
)
parser.add_argument(
    "--debug_reference_joint_tol",
    type=float,
    default=1.0e-4,
    help="Failure threshold for max absolute joint position error (radians).",
)
parser.add_argument(
    "--debug_reference_quat_tol_deg",
    type=float,
    default=0.1,
    help="Failure threshold for root orientation error (degrees).",
)
parser.add_argument(
    "--debug_reference_xpos_tol",
    type=float,
    default=1.0e-2,
    help="Failure threshold for absolute body xpos error norm (meters).",
)
parser.add_argument(
    "--debug_reference_xpos_rel_tol",
    type=float,
    default=1.0e-2,
    help="Failure threshold for relative body xpos error norm (meters).",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import json
import numpy as np
import os
import time
import torch
from datetime import datetime
from typing import Any, Dict

import isaaclab_tasks  # noqa: F401
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_error_magnitude, quat_mul
from isaaclab.utils.dict import print_dict
from isaaclab_tasks.utils.hydra import hydra_task_config

from tensordict import TensorDict, TensorDictBase

try:
    from torchrl.data import TensorDictReplayBuffer
    from torchrl.data.replay_buffers.storages import LazyMemmapStorage
except ImportError:  # pragma: no cover - optional dependency for dataset export
    TensorDictReplayBuffer = None
    LazyMemmapStorage = None

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError:  # pragma: no cover - optional dependency for dataset export
    LeRobotDataset = None


def _as_tensordict(data: Any, batch_size: list[int]) -> TensorDictBase:
    if isinstance(data, TensorDictBase):
        return data
    return TensorDict(data, batch_size=batch_size)


def _to_cpu(data: Any) -> Any:
    if isinstance(data, TensorDictBase):
        return data.detach().to("cpu")
    if isinstance(data, dict):
        return {key: _to_cpu(value) for key, value in data.items()}
    if torch.is_tensor(data):
        return data.detach().cpu()
    return data


def _flatten_obs(obs: dict) -> dict:
    """Flatten one level of nested observation dicts.

    With ``concatenate_terms=False``, IsaacLab produces
    ``{"policy": {"joint_pos": tensor, ...}}``.
    This hoists leaf tensors to top-level: ``{"joint_pos": tensor, ...}``.
    Groups that are already tensors (``concatenate_terms=True``) stay as-is.
    """
    flat: dict = {}
    for k, v in obs.items():
        if isinstance(v, dict):
            flat.update(v)
        else:
            flat[k] = v
    return flat


def _init_replay_buffer(rb_dir: str, max_size: int) -> "TensorDictReplayBuffer":
    if TensorDictReplayBuffer is None or LazyMemmapStorage is None:
        raise ImportError(
            "torchrl is required for --save_torchrl_rb. Install torchrl to enable it."
        )
    os.makedirs(rb_dir, exist_ok=True)
    storage = LazyMemmapStorage(max_size=max_size, scratch_dir=rb_dir)
    return TensorDictReplayBuffer(storage=storage)


def _write_rb_metadata(rb_dir: str, metadata: Dict[str, Any]) -> str:
    metadata_path = os.path.join(rb_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    return metadata_path


def _index_data(data: Any, env_id: int) -> Any:
    if isinstance(data, TensorDictBase):
        return data[env_id]
    if isinstance(data, dict):
        return {key: _index_data(value, env_id) for key, value in data.items()}
    if torch.is_tensor(data):
        return data[env_id]
    return data


def _normalize_array(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    else:
        value = np.asarray(value)
    if value.shape == ():
        value = value.reshape(1)
    return value


def _flatten_nested(prefix: str, data: Any, out: Dict[str, Any]) -> None:
    if isinstance(data, TensorDictBase):
        items = data.items()
    elif isinstance(data, dict):
        items = data.items()
    else:
        out[prefix] = data
        return

    for key, value in items:
        if isinstance(key, tuple):
            key_str = ".".join(str(part) for part in key)
        else:
            key_str = str(key)
        next_prefix = f"{prefix}.{key_str}" if prefix else key_str
        _flatten_nested(next_prefix, value, out)


def _is_image_feature(name: str, value: np.ndarray) -> bool:
    if value.ndim != 3:
        return False
    if "image" in name or "rgb" in name or "camera" in name:
        return True
    return value.shape[0] in (1, 3, 4) or value.shape[-1] in (1, 3, 4)


def _infer_lerobot_features(sample_frame: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    features: Dict[str, Dict[str, Any]] = {}
    for key, value in sample_frame.items():
        if key == "task":
            continue
        value_np = _normalize_array(value)
        if _is_image_feature(key, value_np):
            features[key] = {"dtype": "image", "shape": value_np.shape, "names": None}
        else:
            features[key] = {
                "dtype": np.dtype(value_np.dtype).name,
                "shape": value_np.shape,
                "names": None,
            }
    return features


def _stringify_key(key: Any) -> str:
    if isinstance(key, tuple):
        return ".".join(str(part) for part in key)
    return str(key)


def _collect_field_info(data: Any, prefix: str = "") -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    if isinstance(data, TensorDictBase):
        items = data.items()
    elif isinstance(data, dict):
        items = data.items()
    else:
        key = prefix or "value"
        if torch.is_tensor(data):
            info[key] = {"shape": tuple(data.shape), "dtype": str(data.dtype)}
        elif isinstance(data, np.ndarray):
            info[key] = {"shape": data.shape, "dtype": str(data.dtype)}
        else:
            info[key] = {"type": type(data).__name__}
        return info

    for key, value in items:
        key_str = _stringify_key(key)
        next_prefix = f"{prefix}.{key_str}" if prefix else key_str
        if isinstance(value, (TensorDictBase, dict)):
            info.update(_collect_field_info(value, next_prefix))
        else:
            if torch.is_tensor(value):
                info[next_prefix] = {
                    "shape": tuple(value.shape),
                    "dtype": str(value.dtype),
                }
            elif isinstance(value, np.ndarray):
                info[next_prefix] = {
                    "shape": value.shape,
                    "dtype": str(value.dtype),
                }
            else:
                info[next_prefix] = {"type": type(value).__name__}
    return info


def _quat_apply_batched(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Apply quaternion rotation to a batched vector tensor of shape [N, K, 3]."""
    n_envs, n_items = vec.shape[0], vec.shape[1]
    quat_expanded = quat.unsqueeze(1).expand(n_envs, n_items, 4).reshape(-1, 4)
    vec_flat = vec.reshape(-1, 3)
    return quat_apply(quat_expanded, vec_flat).reshape(n_envs, n_items, 3)


def _quat_apply_inverse_batched(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Apply inverse quaternion rotation to a batched vector tensor of shape [N, K, 3]."""
    n_envs, n_items = vec.shape[0], vec.shape[1]
    quat_expanded = quat.unsqueeze(1).expand(n_envs, n_items, 4).reshape(-1, 4)
    vec_flat = vec.reshape(-1, 3)
    return quat_apply_inverse(quat_expanded, vec_flat).reshape(n_envs, n_items, 3)


def _resolve_reference_to_asset_body_map(env) -> tuple[torch.Tensor, torch.Tensor, list[tuple[str, str]]]:
    """Resolve reference body indices to asset body indices using conservative name matching."""
    reference_names = list(getattr(env.unwrapped, "reference_body_names", []) or [])
    asset_names = list(getattr(env.unwrapped.robot.data, "body_names", []) or [])
    if len(reference_names) == 0 or len(asset_names) == 0:
        return (
            torch.empty(0, dtype=torch.long, device=env.unwrapped.device),
            torch.empty(0, dtype=torch.long, device=env.unwrapped.device),
            [],
        )

    asset_lookup = {name: idx for idx, name in enumerate(asset_names)}
    asset_lookup_lower = {name.lower(): idx for idx, name in enumerate(asset_names)}

    special_alias = {
        "left_wrist_roll_rubber_hand": "left_palm_link",
        "right_wrist_roll_rubber_hand": "right_palm_link",
        "left_elbow_link": "left_elbow_pitch_link",
        "right_elbow_link": "right_elbow_pitch_link",
    }

    ref_ids: list[int] = []
    asset_ids: list[int] = []
    name_pairs: list[tuple[str, str]] = []

    for ref_idx, ref_name in enumerate(reference_names):
        asset_idx = None
        matched_asset_name = None
        if ref_name in asset_lookup:
            asset_idx = asset_lookup[ref_name]
            matched_asset_name = ref_name
        elif ref_name.lower() in asset_lookup_lower:
            asset_idx = asset_lookup_lower[ref_name.lower()]
            matched_asset_name = asset_names[asset_idx]
        else:
            alias_name = special_alias.get(ref_name)
            if alias_name is not None and alias_name in asset_lookup:
                asset_idx = asset_lookup[alias_name]
                matched_asset_name = alias_name

        if asset_idx is None or matched_asset_name is None:
            continue

        ref_ids.append(ref_idx)
        asset_ids.append(asset_idx)
        name_pairs.append((ref_name, matched_asset_name))

    device = env.unwrapped.device
    return (
        torch.tensor(ref_ids, dtype=torch.long, device=device),
        torch.tensor(asset_ids, dtype=torch.long, device=device),
        name_pairs,
    )


def _compute_replay_reference_match_errors(env, reference: TensorDictBase) -> dict[str, torch.Tensor]:
    """Compute replay-vs-reference errors for root pose and joint positions."""
    unwrapped = env.unwrapped
    robot = unwrapped.robot

    init_pos = unwrapped._init_root_pos
    init_quat = unwrapped._init_root_quat

    # Expected root state follows the same transform path used in ImitationRLEnv._replay_reference.
    expected_root_pos = quat_apply(init_quat, reference["root_pos"])
    expected_root_pos[:, :2] += init_pos[:, :2]
    expected_root_pos[:, 2] = init_pos[:, 2]
    expected_root_quat = quat_mul(init_quat, reference["root_quat"])

    actual_root_state = robot.data.root_state_w
    actual_root_pos = actual_root_state[:, :3]
    actual_root_quat = actual_root_state[:, 3:7]

    root_pos_err = torch.linalg.norm(actual_root_pos - expected_root_pos, dim=-1)
    root_pos_xy_err = torch.linalg.norm(actual_root_pos[:, :2] - expected_root_pos[:, :2], dim=-1)
    root_pos_z_err = torch.abs(actual_root_pos[:, 2] - expected_root_pos[:, 2])
    root_quat_err_rad = quat_error_magnitude(actual_root_quat, expected_root_quat)
    root_quat_err_deg = torch.rad2deg(root_quat_err_rad)

    # Expected joint positions also follow _replay_reference (NaNs replaced by defaults).
    ref_joint_pos = reference["joint_pos"]
    expected_joint_pos = torch.where(
        torch.isnan(ref_joint_pos), robot.data.default_joint_pos, ref_joint_pos
    )
    actual_joint_pos = robot.data.joint_pos
    joint_abs_err = torch.abs(actual_joint_pos - expected_joint_pos)
    joint_linf_err = torch.max(joint_abs_err, dim=-1).values
    joint_l2_err = torch.linalg.norm(actual_joint_pos - expected_joint_pos, dim=-1)

    # Optional body xpos check (reference `xpos` vs actual body_link_pos_w) under the same
    # initial rigid transform used by replay.
    xpos_abs_err = torch.empty(0, device=unwrapped.device)
    xpos_rel_err = torch.empty(0, device=unwrapped.device)
    xpos_num_bodies = torch.tensor(0, device=unwrapped.device, dtype=torch.int64)
    if ("xpos" in reference.keys()) and hasattr(unwrapped, "trajectory_manager"):
        ref_body_ids, asset_body_ids, _ = _resolve_reference_to_asset_body_map(env)
        if ref_body_ids.numel() > 0:
            tm = unwrapped.trajectory_manager
            traj_ranks = tm.env_traj_rank.to(dtype=torch.int64)
            start_idx = tm._start[traj_ranks]
            start_td = tm.rb[start_idx]
            start_td = start_td.to(unwrapped.device)
            start_qpos = start_td.get("qpos")
            start_root_pos = start_qpos[:, 0:3]
            start_root_quat = start_qpos[:, 3:7]

            ref_xpos = reference["xpos"][:, ref_body_ids, :]
            ref_xpos_rel_start = _quat_apply_inverse_batched(
                start_root_quat, ref_xpos - start_root_pos.unsqueeze(1)
            )
            expected_xpos_w = _quat_apply_batched(unwrapped._init_root_quat, ref_xpos_rel_start)
            expected_xpos_w = expected_xpos_w + unwrapped._init_root_pos.unsqueeze(1)

            # Match the root z clamping behavior used in replay.
            rigid_root_unclamped = quat_apply(unwrapped._init_root_quat, reference["root_pos"])
            rigid_root_unclamped = rigid_root_unclamped + unwrapped._init_root_pos
            z_shift = rigid_root_unclamped[:, 2] - unwrapped._init_root_pos[:, 2]
            expected_xpos_w[:, :, 2] -= z_shift.unsqueeze(1)

            actual_xpos_w = robot.data.body_link_pos_w[:, asset_body_ids, :]
            xpos_abs_err_per_body = torch.linalg.norm(actual_xpos_w - expected_xpos_w, dim=-1)
            xpos_abs_err = torch.mean(xpos_abs_err_per_body, dim=-1)

            actual_rel = actual_xpos_w - actual_xpos_w[:, :1, :]
            expected_rel = expected_xpos_w - expected_xpos_w[:, :1, :]
            xpos_rel_err_per_body = torch.linalg.norm(actual_rel - expected_rel, dim=-1)
            xpos_rel_err = torch.mean(xpos_rel_err_per_body, dim=-1)
            xpos_num_bodies = torch.tensor(ref_body_ids.numel(), device=unwrapped.device, dtype=torch.int64)

    return {
        "root_pos_err": root_pos_err,
        "root_pos_xy_err": root_pos_xy_err,
        "root_pos_z_err": root_pos_z_err,
        "root_quat_err_deg": root_quat_err_deg,
        "joint_linf_err": joint_linf_err,
        "joint_l2_err": joint_l2_err,
        "xpos_abs_err": xpos_abs_err,
        "xpos_rel_err": xpos_rel_err,
        "xpos_num_bodies": xpos_num_bodies,
    }


def _print_replay_reference_match_debug(
    env,
    step: int,
    errors: dict[str, torch.Tensor],
    *,
    max_envs: int,
    pos_tol: float,
    joint_tol: float,
    quat_tol_deg: float,
    xpos_tol: float,
    xpos_rel_tol: float,
) -> dict[str, float]:
    """Print compact replay-vs-reference error stats and return max metrics."""
    root_pos_err = errors["root_pos_err"]
    root_pos_xy_err = errors["root_pos_xy_err"]
    root_pos_z_err = errors["root_pos_z_err"]
    root_quat_err_deg = errors["root_quat_err_deg"]
    joint_linf_err = errors["joint_linf_err"]
    joint_l2_err = errors["joint_l2_err"]
    xpos_abs_err = errors["xpos_abs_err"]
    xpos_rel_err = errors["xpos_rel_err"]
    xpos_num_bodies = int(errors["xpos_num_bodies"].item())

    pos_fail = root_pos_err > pos_tol
    joint_fail = joint_linf_err > joint_tol
    quat_fail = root_quat_err_deg > quat_tol_deg
    if xpos_abs_err.numel() == 0:
        xpos_fail = torch.zeros_like(pos_fail)
        xpos_rel_fail = torch.zeros_like(pos_fail)
    else:
        xpos_fail = xpos_abs_err > xpos_tol
        xpos_rel_fail = xpos_rel_err > xpos_rel_tol
    any_fail = pos_fail | joint_fail | quat_fail | xpos_fail | xpos_rel_fail

    n_envs = root_pos_err.shape[0]
    n_fail = int(any_fail.sum().item())
    xpos_abs_err_mean = float("nan")
    xpos_rel_err_mean = float("nan")
    xpos_stats = ""
    if xpos_abs_err.numel() > 0:
        xpos_abs_err_mean = xpos_abs_err.mean().item()
        xpos_rel_err_mean = xpos_rel_err.mean().item()
        xpos_stats = (
            f" xpos_abs(mean/max)={xpos_abs_err_mean:.3e}/{xpos_abs_err.max().item():.3e}"
            f" xpos_rel(mean/max)={xpos_rel_err_mean:.3e}/{xpos_rel_err.max().item():.3e}"
            f" xpos_bodies={xpos_num_bodies}"
        )
    print(
        "[DEBUG][reference_match] "
        f"step={step} "
        f"root_pos(mean/max)={root_pos_err.mean().item():.3e}/{root_pos_err.max().item():.3e} "
        f"root_xy(max)={root_pos_xy_err.max().item():.3e} "
        f"root_z(max)={root_pos_z_err.max().item():.3e} "
        f"root_quat_deg(mean/max)={root_quat_err_deg.mean().item():.3e}/{root_quat_err_deg.max().item():.3e} "
        f"joint_linf(mean/max)={joint_linf_err.mean().item():.3e}/{joint_linf_err.max().item():.3e} "
        f"joint_l2(mean/max)={joint_l2_err.mean().item():.3e}/{joint_l2_err.max().item():.3e} "
        f"{xpos_stats}"
        f"fails={n_fail}/{n_envs}"
    )

    if n_fail > 0:
        failing_ids = any_fail.nonzero(as_tuple=False).squeeze(-1).tolist()
        failing_ids = failing_ids[: max(0, max_envs)]
        tm = getattr(env.unwrapped, "trajectory_manager", None)
        for env_id in failing_ids:
            traj_msg = ""
            if tm is not None:
                dataset, motion, trajectory = tm.get_env_traj_info(env_id)
                rank = int(tm.env_traj_rank[env_id].item())
                local_step = int(tm.env_step[env_id].item())
                traj_msg = (
                    f" traj={dataset}/{motion}/{trajectory} rank={rank} local_step={local_step}"
                )
            print(
                "[DEBUG][reference_match][env] "
                f"env={env_id} "
                f"root_pos={root_pos_err[env_id].item():.3e} "
                f"root_quat_deg={root_quat_err_deg[env_id].item():.3e} "
                f"joint_linf={joint_linf_err[env_id].item():.3e} "
                f"xpos_abs={(xpos_abs_err[env_id].item() if xpos_abs_err.numel() > 0 else float('nan')):.3e} "
                f"xpos_rel={(xpos_rel_err[env_id].item() if xpos_rel_err.numel() > 0 else float('nan')):.3e}"
                f"{traj_msg}"
            )

    return {
        "root_pos_err_max": root_pos_err.max().item(),
        "root_pos_xy_err_max": root_pos_xy_err.max().item(),
        "root_pos_z_err_max": root_pos_z_err.max().item(),
        "root_quat_err_deg_max": root_quat_err_deg.max().item(),
        "joint_linf_err_max": joint_linf_err.max().item(),
        "joint_l2_err_max": joint_l2_err.max().item(),
        "xpos_abs_err_max": xpos_abs_err.max().item() if xpos_abs_err.numel() > 0 else 0.0,
        "xpos_rel_err_max": xpos_rel_err.max().item() if xpos_rel_err.numel() > 0 else 0.0,
        "xpos_abs_err_mean": xpos_abs_err_mean,
        "xpos_rel_err_mean": xpos_rel_err_mean,
        "xpos_available": float(xpos_abs_err.numel() > 0),
        "n_fail": float(n_fail),
    }


class LeRobotExporter:
    def __init__(
        self,
        output_dir: str,
        repo_id: str,
        task_name: str,
        fps: int,
        num_envs: int,
        obs_sample: Any,
        action_sample: Any,
        reward_sample: Any,
        done_sample: Any,
        reference_sample: Any,
    ) -> None:
        if LeRobotDataset is None:
            raise ImportError(
                "lerobot is required for --lerobot_dir. Install lerobot to enable it."
            )
        if os.path.exists(output_dir):
            raise FileExistsError(
                f"LeRobot output directory already exists: {output_dir}"
            )
        self.task_name = task_name
        self.num_envs = num_envs
        sample_frame = self._build_frame(
            obs_sample,
            action_sample,
            reward_sample,
            done_sample,
            reference_sample,
            env_id=0,
        )
        features = _infer_lerobot_features(sample_frame)
        self.dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps,
            features=features,
            root=output_dir,
            use_videos=False,
        )
        self.active_buffers = [
            self.dataset.create_episode_buffer(episode_index=env_id)
            for env_id in range(num_envs)
        ]
        self.next_episode_index = num_envs
        self.completed_buffers: list[dict] = []

    def _build_frame(
        self,
        obs: Any,
        action: Any,
        reward: Any,
        done: Any,
        reference: Any,
        env_id: int,
    ) -> Dict[str, Any]:
        frame: Dict[str, Any] = {"task": self.task_name}
        obs_env = _index_data(_flatten_obs(obs), env_id)
        ref_env = _index_data(reference, env_id)
        obs_flat: Dict[str, Any] = {}
        ref_flat: Dict[str, Any] = {}
        _flatten_nested("observation", obs_env, obs_flat)
        _flatten_nested("reference", ref_env, ref_flat)
        for key, value in {**obs_flat, **ref_flat}.items():
            frame[key] = _normalize_array(value)
        frame["action"] = _normalize_array(_index_data(action, env_id))
        frame["next.reward"] = _normalize_array(_index_data(reward, env_id))
        frame["next.done"] = _normalize_array(_index_data(done, env_id))
        return frame

    def add_step(
        self,
        obs: Any,
        action: Any,
        reward: Any,
        done: Any,
        reference: Any,
    ) -> None:
        done_cpu = done.detach().cpu() if torch.is_tensor(done) else done
        for env_id in range(self.num_envs):
            frame = self._build_frame(obs, action, reward, done, reference, env_id)
            self.dataset.episode_buffer = self.active_buffers[env_id]
            self.dataset.add_frame(frame)
            self.active_buffers[env_id] = self.dataset.episode_buffer
            if bool(done_cpu[env_id]):
                self.completed_buffers.append(self.active_buffers[env_id])
                self.active_buffers[env_id] = self.dataset.create_episode_buffer(
                    episode_index=self.next_episode_index
                )
                self.next_episode_index += 1

    def finalize(self) -> None:
        for buffer in self.active_buffers:
            if buffer["size"] > 0:
                self.completed_buffers.append(buffer)
        self.completed_buffers.sort(key=lambda buf: buf["episode_index"])
        for buffer in self.completed_buffers:
            self.dataset.save_episode(episode_data=buffer)
        self.dataset.stop_image_writer()


class ReplayExportTester:
    @staticmethod
    def inspect_torchrl_rb_buffer(
        rb: "TensorDictReplayBuffer", sample_size: int = 4
    ) -> Dict[str, Any]:
        sample = rb.sample(sample_size)
        return {
            "size": len(rb),
            "sample_fields": _collect_field_info(sample),
        }

    @staticmethod
    def inspect_torchrl_rb(rb_dir: str, sample_size: int = 4) -> Dict[str, Any]:
        if TensorDictReplayBuffer is None or LazyMemmapStorage is None:
            raise ImportError("torchrl is required to inspect the replay buffer.")
        if hasattr(TensorDictReplayBuffer, "load"):
            rb = TensorDictReplayBuffer.load(rb_dir)
            return ReplayExportTester.inspect_torchrl_rb_buffer(rb, sample_size)
        storage = LazyMemmapStorage(max_size=1)
        if not hasattr(storage, "load"):
            raise RuntimeError("TorchRL storage does not support load; update torchrl.")
        storage.load(rb_dir)
        rb = TensorDictReplayBuffer(storage=storage)
        return ReplayExportTester.inspect_torchrl_rb_buffer(rb, sample_size)

    @staticmethod
    def inspect_lerobot_dataset(lerobot_dir: str, sample_index: int = 0) -> Dict[str, Any]:
        if LeRobotDataset is None:
            raise ImportError("lerobot is required to inspect the dataset.")
        dataset = LeRobotDataset(repo_id="local", root=lerobot_dir)
        sample = dataset[sample_index]
        info = {}
        if hasattr(dataset, "meta") and hasattr(dataset.meta, "info"):
            info = dataset.meta.info
        return {
            "num_frames": len(dataset),
            "info": info,
            "sample_fields": _collect_field_info(sample),
        }


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):  # noqa: ARG001
    """Replay reference data."""
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
    env_cfg.sim.device = (
        args_cli.device if args_cli.device is not None else env_cfg.sim.device
    )

    # force reference replay
    env_cfg.replay_reference = True
    env_cfg.replay_only = True

    task_name = args_cli.task.split(":")[-1]
    log_root_path = os.path.abspath(os.path.join("logs", "reference_replay", task_name))
    print(f"[INFO] Logging replay in directory: {log_root_path}")
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(log_root_path, log_dir)
    os.makedirs(log_dir, exist_ok=True)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(
        args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None
    )

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "replay"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during reference replay.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # reset environment
    obs, _ = env.reset()

    dt = env.unwrapped.step_dt
    max_steps = args_cli.video_length if args_cli.video else args_cli.max_steps
    num_envs = env.unwrapped.num_envs

    rb = None
    rb_dir = None
    if args_cli.save_torchrl_rb:
        if max_steps is None:
            raise ValueError("--save_torchrl_rb requires a finite --max_steps.")
        rb_dir = (
            args_cli.torchrl_rb_dir
            if args_cli.torchrl_rb_dir is not None
            else os.path.join(log_dir, "torchrl_rb")
        )
        rb = _init_replay_buffer(rb_dir, max_size=max_steps * num_envs)
        _write_rb_metadata(
            rb_dir,
            {
                "task": args_cli.task,
                "num_envs": num_envs,
                "max_steps": max_steps,
                "device": str(env.unwrapped.device),
                "log_dir": log_dir,
            },
        )

    # prepare a zero action with the right shape
    action = torch.as_tensor(env.action_space.sample(), device=env.unwrapped.device)
    action = torch.zeros_like(action)
    reference = env.unwrapped.get_reference_data()
    debug_interval = max(1, int(args_cli.debug_reference_interval))
    debug_agg_max: dict[str, float] = {
        "root_pos_err_max": 0.0,
        "root_pos_xy_err_max": 0.0,
        "root_pos_z_err_max": 0.0,
        "root_quat_err_deg_max": 0.0,
        "joint_linf_err_max": 0.0,
        "joint_l2_err_max": 0.0,
        "xpos_abs_err_max": 0.0,
        "xpos_rel_err_max": 0.0,
    }
    debug_fail_steps = 0
    debug_eval_steps = 0
    debug_xpos_eval_steps = 0
    debug_xpos_abs_time_sum = 0.0
    debug_xpos_rel_time_sum = 0.0
    if args_cli.debug_reference_match:
        ref_body_ids_dbg, asset_body_ids_dbg, name_pairs_dbg = _resolve_reference_to_asset_body_map(env)
        print(
            "[INFO] Replay-reference debug checks enabled:"
            f" interval={debug_interval},"
            f" pos_tol={args_cli.debug_reference_pos_tol:.3e} m,"
            f" joint_tol={args_cli.debug_reference_joint_tol:.3e} rad,"
            f" quat_tol={args_cli.debug_reference_quat_tol_deg:.3e} deg,"
            f" xpos_tol={args_cli.debug_reference_xpos_tol:.3e} m,"
            f" xpos_rel_tol={args_cli.debug_reference_xpos_rel_tol:.3e} m"
        )
        print(
            "[INFO] Replay-reference xpos debug mapping:"
            f" mapped_reference_bodies={int(ref_body_ids_dbg.numel())}/"
            f"{len(getattr(env.unwrapped, 'reference_body_names', []))}, "
            f"mapped_asset_bodies={int(asset_body_ids_dbg.numel())}/"
            f"{len(env.unwrapped.robot.data.body_names)}"
        )
        if len(name_pairs_dbg) > 0:
            preview = ", ".join([f"{r}->{a}" for r, a in name_pairs_dbg[:8]])
            print(f"[INFO] Replay-reference xpos mapping preview: {preview}")
        else:
            print("[WARN] Replay-reference xpos mapping found no overlapping bodies; xpos errors disabled.")

    lerobot_exporter = None
    lerobot_dir = args_cli.lerobot_dir
    if args_cli.save_lerobot and lerobot_dir is None:
        lerobot_dir = os.path.join(log_dir, "lerobot")
    if lerobot_dir is not None:
        fps = int(round(1.0 / dt)) if dt > 0 else 1
        reward_sample = torch.zeros(
            (num_envs,), device=env.unwrapped.device, dtype=torch.float32
        )
        done_sample = torch.zeros(
            (num_envs,), device=env.unwrapped.device, dtype=torch.bool
        )
        lerobot_exporter = LeRobotExporter(
            output_dir=lerobot_dir,
            repo_id=task_name.replace(":", "_"),
            task_name=task_name,
            fps=fps,
            num_envs=num_envs,
            obs_sample=obs,
            action_sample=action,
            reward_sample=reward_sample,
            done_sample=done_sample,
            reference_sample=reference,
        )

    timestep = 0
    while simulation_app.is_running():
        start_time = time.time()
        next_obs, reward, terminated, truncated, extras = env.step(action)
        next_reference = env.unwrapped.get_reference_data()
        if args_cli.debug_reference_match and (timestep % debug_interval == 0):
            errors = _compute_replay_reference_match_errors(env, next_reference)
            debug_report = _print_replay_reference_match_debug(
                env,
                timestep,
                errors,
                max_envs=args_cli.debug_reference_max_envs,
                pos_tol=args_cli.debug_reference_pos_tol,
                joint_tol=args_cli.debug_reference_joint_tol,
                quat_tol_deg=args_cli.debug_reference_quat_tol_deg,
                xpos_tol=args_cli.debug_reference_xpos_tol,
                xpos_rel_tol=args_cli.debug_reference_xpos_rel_tol,
            )
            debug_eval_steps += 1
            for key in debug_agg_max:
                debug_agg_max[key] = max(debug_agg_max[key], float(debug_report[key]))
            if int(debug_report["xpos_available"]) > 0:
                debug_xpos_eval_steps += 1
                debug_xpos_abs_time_sum += float(debug_report["xpos_abs_err_mean"])
                debug_xpos_rel_time_sum += float(debug_report["xpos_rel_err_mean"])
            if int(debug_report["n_fail"]) > 0:
                debug_fail_steps += 1
        if rb is not None:
            done = terminated | truncated

            # Flatten observation groups to top-level keys (handles both
            # concatenate_terms=True and False) and move to CPU.
            flat_obs = _to_cpu(_flatten_obs(obs))
            flat_next_obs = _to_cpu(_flatten_obs(next_obs))

            transition = TensorDict({}, batch_size=[num_envs])
            for key, val in flat_obs.items():
                transition.set(key, val)
            transition.set("reference", _as_tensordict(_to_cpu(reference), batch_size=[num_envs]))
            transition.set("action", _to_cpu(action))
            transition.set("reward", _to_cpu(reward))
            transition.set("terminated", _to_cpu(terminated))
            transition.set("truncated", _to_cpu(truncated))
            transition.set("done", _to_cpu(done))
            for key, val in flat_next_obs.items():
                transition.set(("next", key), val)
            transition.set(("next", "reference"), _as_tensordict(_to_cpu(next_reference), batch_size=[num_envs]))
            transition.set(("next", "reward"), _to_cpu(reward))
            transition.set(("next", "done"), _to_cpu(done))

            # Overwrite next-obs for done envs with their true terminal obs.
            # IsaacLab stores final_obs as np.ndarray(num_envs, dtype=object)
            # where each element is None (not done) or a per-env obs dict.
            final_obs_arr = extras.get("final_obs")
            if final_obs_arr is not None:
                done_ids = done.nonzero(as_tuple=False).squeeze(-1).tolist()
                if isinstance(done_ids, int):
                    done_ids = [done_ids]
                for env_id in done_ids:
                    if final_obs_arr[env_id] is None:
                        continue
                    flat_final = _flatten_obs(final_obs_arr[env_id])
                    for key, val in flat_final.items():
                        buf = transition.get(("next", key), default=None)
                        if buf is None:
                            continue
                        if torch.is_tensor(val):
                            buf[env_id] = val.cpu()
                        else:
                            buf[env_id] = torch.as_tensor(val)

            rb.extend(transition)

        if lerobot_exporter is not None:
            done = terminated | truncated
            lerobot_exporter.add_step(obs, action, reward, done, reference)

        obs = next_obs
        reference = next_reference
        timestep += 1
        if max_steps is not None and timestep >= max_steps:
            break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()

    if args_cli.debug_reference_match:
        xpos_abs_time_avg = None
        xpos_rel_time_avg = None
        if debug_xpos_eval_steps > 0:
            xpos_abs_time_avg = debug_xpos_abs_time_sum / float(debug_xpos_eval_steps)
            xpos_rel_time_avg = debug_xpos_rel_time_sum / float(debug_xpos_eval_steps)
        print("[INFO] Replay-reference debug summary:")
        print(json.dumps(
            {
                **debug_agg_max,
                "fail_steps": debug_fail_steps,
                "debug_eval_steps": debug_eval_steps,
                "xpos_eval_steps": debug_xpos_eval_steps,
                "xpos_abs_err_time_avg": xpos_abs_time_avg,
                "xpos_rel_err_time_avg": xpos_rel_time_avg,
                "evaluated_steps": timestep,
            },
            indent=2,
            sort_keys=True,
        ))

    if rb is not None and rb_dir is not None:
        if hasattr(rb, "dump"):
            rb.dump(rb_dir)
        try:
            rb_shapes = ReplayExportTester.inspect_torchrl_rb_buffer(rb)
            print("[INFO] TorchRL replay buffer inspection:")
            print(json.dumps(rb_shapes, indent=2, sort_keys=True, default=str))
        except Exception as exc:
            print(f"[WARN] Failed to inspect TorchRL replay buffer: {exc}")
    if lerobot_exporter is not None:
        lerobot_exporter.finalize()
        if lerobot_dir is not None:
            try:
                lerobot_shapes = ReplayExportTester.inspect_lerobot_dataset(lerobot_dir)
                print("[INFO] LeRobot dataset inspection:")
                print(json.dumps(lerobot_shapes, indent=2, sort_keys=True, default=str))
            except Exception as exc:
                print(f"[WARN] Failed to inspect LeRobot dataset: {exc}")


if __name__ == "__main__":
    main()
    simulation_app.close()
