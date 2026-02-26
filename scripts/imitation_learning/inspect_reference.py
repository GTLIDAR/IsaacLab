# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Inspect imitation reference metadata, manager terms, and loco-mujoco body/site maps.

Examples:
    ./isaaclab.sh -p scripts/imitation_learning/inspect_reference.py --task Isaac-Imitation-G1-v0
    ./isaaclab.sh -p scripts/imitation_learning/inspect_reference.py --task Isaac-Imitation-G1-v0 --motion walk
    ./isaaclab.sh -p scripts/imitation_learning/inspect_reference.py --task Isaac-Imitation-G1-v0 --loco_only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Inspect imitation reference metadata.")
parser.add_argument("--task", type=str, default="Isaac-Imitation-G1-v0", help="Task name to inspect.")
parser.add_argument(
    "--agent",
    type=str,
    default="rsl_rl_cfg_entry_point",
    help="Agent config entry point (used by hydra registration).",
)
parser.add_argument("--num_envs", type=int, default=None, help="Override number of environments.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--dataset_path", type=str, default=None, help="Override imitation dataset path.")
parser.add_argument("--dataset_type", type=str, default="default", choices=["default", "lafan1", "amass"])
parser.add_argument("--motion", type=str, default="walk", help="Comma-separated motion names.")
parser.add_argument(
    "--env_name",
    type=str,
    default=None,
    help="Override loco-mujoco environment name in loader kwargs (e.g. UnitreeG1).",
)
parser.add_argument("--n_substeps", type=int, default=None, help="Override loader n_substeps.")
parser.add_argument("--preview_envs", type=int, default=4, help="How many env assignments to print.")
parser.add_argument(
    "--max_list_items",
    type=int,
    default=256,
    help="Maximum body/site/index rows to print per section.",
)
parser.add_argument(
    "--loco_only",
    action="store_true",
    default=False,
    help="Inspect loco-mujoco trajectory info directly without creating IsaacLab gym env.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from tensordict import TensorDictBase

import isaaclab_tasks  # noqa: F401
from isaaclab.managers import SceneEntityCfg
from isaaclab_tasks.manager_based.imitation.config.g1.imitation_g1_env_cfg import (
    G1_WBT_TRACKED_ASSET_BODY_NAMES,
    G1_WBT_TRACKED_BODY_NAMES,
    G1_WBT_TRACKED_REFERENCE_BODY_NAMES,
)
from isaaclab_tasks.utils.hydra import hydra_task_config


def _section(title: str) -> None:
    print("\n" + "=" * 96)
    print(title)
    print("=" * 96)


def _motions_from_arg(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _summarize_value(value: Any, max_str_len: int = 120) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if torch.is_tensor(value):
        return {"shape": tuple(value.shape), "dtype": str(value.dtype), "device": str(value.device)}
    if isinstance(value, TensorDictBase):
        return {"type": "TensorDict", "keys": [str(k) for k in value.keys()]}
    if isinstance(value, SceneEntityCfg):
        return {
            "name": value.name,
            "joint_names": value.joint_names,
            "joint_ids": value.joint_ids,
            "body_names": value.body_names,
            "body_ids": value.body_ids,
            "preserve_order": value.preserve_order,
        }
    if isinstance(value, (list, tuple)):
        return [_summarize_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _summarize_value(v) for k, v in value.items()}
    text = repr(value)
    return text if len(text) <= max_str_len else text[: max_str_len - 3] + "..."


def _print_manager_terms(env) -> None:
    _section("Manager Metadata")

    obs_mgr = env.unwrapped.observation_manager
    print("[ObservationManager]")
    print("  Active groups:", list(obs_mgr.active_terms.keys()))
    for group_name, term_names in obs_mgr.active_terms.items():
        print(f"  - Group '{group_name}'")
        print(f"    concatenate_terms={obs_mgr.group_obs_concatenate[group_name]}")
        print(f"    group_obs_dim={obs_mgr.group_obs_dim[group_name]}")
        print(f"    terms({len(term_names)}): {term_names}")

    rew_mgr = env.unwrapped.reward_manager
    print("\n[RewardManager]")
    print("  Active terms:", rew_mgr.active_terms)
    for name in rew_mgr.active_terms:
        cfg = rew_mgr.get_term_cfg(name)
        func_name = getattr(cfg.func, "__name__", cfg.func.__class__.__name__)
        payload = {
            "func": func_name,
            "weight": cfg.weight,
            "params": _summarize_value(cfg.params),
        }
        print(f"  - {name}: {json.dumps(payload, default=str)}")

    event_mgr = env.unwrapped.event_manager
    print("\n[EventManager]")
    print("  Modes:", event_mgr.available_modes)
    for mode, names in event_mgr.active_terms.items():
        print(f"  - Mode '{mode}' terms({len(names)}): {names}")
        for name in names:
            cfg = event_mgr.get_term_cfg(name)
            payload = {
                "func": getattr(cfg.func, "__name__", cfg.func.__class__.__name__),
                "interval_range_s": cfg.interval_range_s,
                "is_global_time": cfg.is_global_time,
                "min_step_count_between_reset": cfg.min_step_count_between_reset,
                "params": _summarize_value(cfg.params),
            }
            print(f"    * {name}: {json.dumps(payload, default=str)}")

    term_mgr = env.unwrapped.termination_manager
    print("\n[TerminationManager]")
    print("  Active terms:", term_mgr.active_terms)
    for name in term_mgr.active_terms:
        cfg = term_mgr.get_term_cfg(name)
        payload = {
            "func": getattr(cfg.func, "__name__", cfg.func.__class__.__name__),
            "time_out": cfg.time_out,
            "params": _summarize_value(cfg.params),
        }
        print(f"  - {name}: {json.dumps(payload, default=str)}")


def _print_trajectory_manager(env, preview_envs: int) -> None:
    _section("Trajectory Manager Metadata")
    tm = env.unwrapped.trajectory_manager
    traj_info = tm.traj_info
    ordered = traj_info.get("ordered_traj_list", [])
    starts = traj_info.get("start_index", [])
    ends = traj_info.get("end_index", [])
    print("num_trajectories:", tm.num_trajectories)
    print("reset_schedule:", tm.reset_schedule)
    print("wrap_steps:", tm.wrap_steps)
    print("traj_info keys:", list(traj_info.keys()))
    print("ordered_traj_list length:", len(ordered))

    for rank, item in enumerate(ordered[: min(10, len(ordered))]):
        length = int(ends[rank]) - int(starts[rank]) if rank < len(starts) and rank < len(ends) else -1
        print(f"  rank={rank:3d} trajectory={item} length={length}")

    n_env_print = min(max(preview_envs, 0), env.unwrapped.num_envs)
    print(f"\nFirst {n_env_print} env assignments:")
    for env_id in range(n_env_print):
        rank = int(tm.env_traj_rank[env_id].item())
        step = int(tm.env_step[env_id].item())
        ds, motion, traj = tm.get_env_traj_info(env_id)
        print(f"  env={env_id:3d} rank={rank:3d} step={step:6d} -> {ds}/{motion}/{traj}")


def _print_name_list(title: str, names: list[str], max_items: int) -> None:
    print(f"{title} ({len(names)}):")
    for idx, name in enumerate(names[:max_items]):
        print(f"  {idx:3d}: {name}")
    if len(names) > max_items:
        print(f"  ... truncated ({len(names) - max_items} more)")


def _print_reference_metadata(env, max_items: int) -> None:
    _section("Reference Metadata and Tensor Keys")
    reference = env.unwrapped.get_reference_data()
    body_names = list(getattr(env.unwrapped, "reference_body_names", []))
    site_names = list(getattr(env.unwrapped, "reference_site_names", []))
    _print_name_list("Reference body names", body_names, max_items=max_items)
    _print_name_list("Reference site names", site_names, max_items=max_items)

    print("\nReference TensorDict keys/shapes:")
    ref_keys = [str(k) for k in reference.keys()]
    for key in reference.keys():
        value = reference.get(key)
        if torch.is_tensor(value):
            print(f"  - {key}: shape={tuple(value.shape)} dtype={value.dtype} device={value.device}")
        else:
            print(f"  - {key}: type={type(value).__name__}")

    body_array_keys = ("xpos", "xquat", "cvel", "subtree_com")
    site_array_keys = ("site_xpos", "site_xmat")
    print("\nBody-array consistency:")
    for key in body_array_keys:
        if key not in ref_keys:
            print(f"  - {key}: unavailable")
            continue
        value = reference.get(key)
        if not torch.is_tensor(value):
            print(f"  - {key}: unavailable")
            continue
        n_bodies = value.shape[1] if value.ndim >= 3 else -1
        print(f"  - {key}: shape={tuple(value.shape)} body_dim={n_bodies} body_names={len(body_names)}")

    print("\nSite-array consistency:")
    for key in site_array_keys:
        if key not in ref_keys:
            print(f"  - {key}: unavailable")
            continue
        value = reference.get(key)
        if not torch.is_tensor(value):
            print(f"  - {key}: unavailable")
            continue
        n_sites = value.shape[1] if value.ndim >= 3 else -1
        print(f"  - {key}: shape={tuple(value.shape)} site_dim={n_sites} site_names={len(site_names)}")


def _print_reference_tracking_compatibility(env, max_items: int) -> None:
    _section("G1 Tracking Compatibility")
    asset_body_names = list(getattr(env.unwrapped.scene["robot"], "body_names", []))
    all_reference_body_names = list(getattr(env.unwrapped, "reference_body_names", []))
    if not asset_body_names:
        print("No robot asset body names found on env scene.")
        return
    if not all_reference_body_names:
        print("No reference body names found in metadata.")
        return

    print(
        f"Tracked pair lengths: asset={len(G1_WBT_TRACKED_ASSET_BODY_NAMES)} "
        f"reference={len(G1_WBT_TRACKED_REFERENCE_BODY_NAMES)}"
    )
    if len(G1_WBT_TRACKED_ASSET_BODY_NAMES) != len(G1_WBT_TRACKED_REFERENCE_BODY_NAMES):
        print("[WARN] Tracked asset/reference lists have different lengths.")

    asset_lookup = {name: idx for idx, name in enumerate(asset_body_names)}
    ref_lookup = {name: idx for idx, name in enumerate(all_reference_body_names)}

    missing_asset = [name for name in G1_WBT_TRACKED_ASSET_BODY_NAMES if name not in asset_lookup]
    missing_ref = [name for name in G1_WBT_TRACKED_REFERENCE_BODY_NAMES if name not in ref_lookup]

    print(f"Asset bodies present: {len(G1_WBT_TRACKED_ASSET_BODY_NAMES) - len(missing_asset)}/{len(G1_WBT_TRACKED_ASSET_BODY_NAMES)}")
    if missing_asset:
        for name in missing_asset:
            print(f"  - Missing asset body: {name}")

    print(
        f"Reference bodies present: {len(G1_WBT_TRACKED_REFERENCE_BODY_NAMES) - len(missing_ref)}/"
        f"{len(G1_WBT_TRACKED_REFERENCE_BODY_NAMES)}"
    )
    if missing_ref:
        for name in missing_ref:
            print(f"  - Missing reference body: {name}")

    print("\nTracked body pair mapping (asset -> reference):")
    for asset_name, ref_name in zip(
        G1_WBT_TRACKED_ASSET_BODY_NAMES[:max_items], G1_WBT_TRACKED_REFERENCE_BODY_NAMES[:max_items]
    ):
        asset_idx = asset_lookup.get(asset_name, -1)
        ref_idx = ref_lookup.get(ref_name, -1)
        print(f"  - {asset_name:24s} ({asset_idx:3d}) -> {ref_name:28s} ({ref_idx:3d})")
    if len(G1_WBT_TRACKED_ASSET_BODY_NAMES) > max_items:
        print(f"  ... truncated ({len(G1_WBT_TRACKED_ASSET_BODY_NAMES) - max_items} more)")

    # Legacy alias visibility for older scripts.
    print("\nLegacy alias G1_WBT_TRACKED_BODY_NAMES:")
    print(" ", G1_WBT_TRACKED_BODY_NAMES)


def _inspect_loco_mujoco_direct(env_cfg) -> None:
    _section("Direct Loco-MuJoCo Inspection")
    try:
        from loco_mujoco.task_factories import (
            AMASSDatasetConf,
            DefaultDatasetConf,
            ImitationFactory,
            LAFAN1DatasetConf,
        )
    except Exception as exc:
        print(f"Could not import loco_mujoco for direct inspection: {exc}")
        return

    loader_kwargs = dict(getattr(env_cfg, "loader_kwargs", {}) or {})
    env_name = args_cli.env_name or loader_kwargs.get("env_name", "UnitreeG1")
    n_substeps = args_cli.n_substeps if args_cli.n_substeps is not None else loader_kwargs.get("n_substeps", 4)
    sim_cfg = dict(loader_kwargs.get("sim", {}) or {})
    timestep = sim_cfg.get("dt", None)
    motions = _motions_from_arg(args_cli.motion)
    if len(motions) == 0:
        motions = ["walk"]

    if len(motions) > 1:
        print(f"Multiple motions passed; direct inspector currently uses the first one: {motions[0]}")
    motion = motions[0]

    factory_kwargs: dict[str, Any]
    if args_cli.dataset_type == "default":
        factory_kwargs = {"default_dataset_conf": DefaultDatasetConf(task=motion)}
    elif args_cli.dataset_type == "lafan1":
        factory_kwargs = {"lafan1_dataset_conf": LAFAN1DatasetConf(dataset_name=motion)}
    else:
        factory_kwargs = {"amass_dataset_conf": AMASSDatasetConf(rel_dataset_path=motion)}

    print(
        "Creating loco-mujoco env with "
        f"env_name={env_name}, dataset_type={args_cli.dataset_type}, motion={motion}, "
        f"n_substeps={n_substeps}, timestep={timestep}"
    )
    make_kwargs: dict[str, Any] = {"n_substeps": n_substeps, **factory_kwargs}
    if timestep is not None:
        make_kwargs["timestep"] = timestep
    loco_env = ImitationFactory.make(env_name, **make_kwargs)

    traj = loco_env.th.traj
    info = traj.info
    data = traj.data

    print("Trajectory info:")
    print(f"  joint_names={len(info.joint_names)}")
    print(f"  body_names={0 if info.body_names is None else len(info.body_names)}")
    print(f"  site_names={0 if info.site_names is None else len(info.site_names)}")
    print(f"  frequency={info.frequency}")
    if info.body_names is not None:
        _print_name_list("  body_names", list(info.body_names), max_items=min(args_cli.max_list_items, 64))
    if info.site_names is not None:
        _print_name_list("  site_names", list(info.site_names), max_items=min(args_cli.max_list_items, 64))

    print("\nTrajectory data shapes:")
    print(f"  split_points: shape={tuple(data.split_points.shape)}")
    print(f"  qpos:        shape={tuple(data.qpos.shape)}")
    print(f"  qvel:        shape={tuple(data.qvel.shape)}")
    print(f"  xpos:        shape={tuple(data.xpos.shape)}")
    print(f"  xquat:       shape={tuple(data.xquat.shape)}")
    print(f"  cvel:        shape={tuple(data.cvel.shape)}")
    print(f"  subtree_com: shape={tuple(data.subtree_com.shape)}")
    print(f"  site_xpos:   shape={tuple(data.site_xpos.shape)}")
    print(f"  site_xmat:   shape={tuple(data.site_xmat.shape)}")


def _apply_loader_selection_overrides(env_cfg) -> None:
    motions = _motions_from_arg(args_cli.motion)
    if len(motions) == 0:
        motions = ["walk"]

    loader_kwargs = dict(getattr(env_cfg, "loader_kwargs", {}) or {})
    dataset_cfg = dict(loader_kwargs.get("dataset", {}) or {})
    trajectories = dict(dataset_cfg.get("trajectories", {}) or {})
    trajectories["default"] = []
    trajectories["lafan1"] = []
    trajectories["amass"] = []
    trajectories[args_cli.dataset_type] = motions
    dataset_cfg["trajectories"] = trajectories
    loader_kwargs["dataset"] = dataset_cfg

    if args_cli.env_name is not None:
        loader_kwargs["env_name"] = args_cli.env_name
    if args_cli.n_substeps is not None:
        loader_kwargs["n_substeps"] = args_cli.n_substeps

    env_cfg.loader_kwargs = loader_kwargs

    if args_cli.dataset_path is not None:
        env_cfg.dataset_path = args_cli.dataset_path
        dataset_path = Path(args_cli.dataset_path)
        trajectories_zarr = dataset_path / "trajectories.zarr"
        if dataset_path.exists() or trajectories_zarr.exists():
            print(
                "[INFO] dataset_path already exists. Existing zarr will be reused; "
                "loader dataset/motion overrides only apply if a new zarr is built."
            )

    print("[INFO] Effective loader trajectory selection:")
    print(json.dumps(env_cfg.loader_kwargs.get("dataset", {}), indent=2))


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, _agent_cfg):  # noqa: ARG001
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    # We only need metadata inspection, no replay forcing.
    env_cfg.replay_reference = False
    env_cfg.replay_only = False

    _apply_loader_selection_overrides(env_cfg)

    if args_cli.loco_only:
        _inspect_loco_mujoco_direct(env_cfg)
        return

    _section("Creating IsaacLab Environment")
    env = gym.make(args_cli.task, cfg=env_cfg)
    obs, _ = env.reset()
    print("Reset complete.")
    if isinstance(obs, dict):
        print("Observation groups:", list(obs.keys()))
    else:
        print("Observation type:", type(obs).__name__)

    _print_manager_terms(env)
    _print_trajectory_manager(env, preview_envs=args_cli.preview_envs)
    _print_reference_metadata(env, max_items=args_cli.max_list_items)
    _print_reference_tracking_compatibility(env, max_items=args_cli.max_list_items)

    _section("Done")
    print("Inspection completed.")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
