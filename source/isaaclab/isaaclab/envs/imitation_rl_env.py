import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import zarr
from tensordict import TensorDict

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs.common import VecEnvStepReturn
from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import (
    FRAME_MARKER_CFG,
)

# Import the new manager and utilities
try:
    from iltools.datasets.lafan1.loader import Lafan1CsvLoader
    from iltools.datasets.loco_mujoco.loader import LocoMuJoCoLoader
    from iltools.datasets.manager import ParallelTrajectoryManager, ResetSchedule
    from iltools.datasets.utils import make_rb_from
except ImportError as e:
    raise ImportError(
        f"Failed to import required modules from iltools_datasets: {e}. Make sure ImitationLearningTools is installed."
    ) from e


class ImitationRLEnv(ManagerBasedRLEnv):
    """
    Simplified RL environment for imitation learning with clean dataset interface.

    Config attributes (cfg):
        dataset_path: str, path to Zarr dataset directory (or directory containing trajectories.zarr)
        reset_schedule: str, trajectory reset schedule ("random", "sequential", "round_robin", "custom")
        wrap_steps: bool, if True, wrap steps within trajectory (default: False)
        replay_only: bool, if True, ignore actions and force reference root/joint state each step
        loader_type: str, required if Zarr does not exist (e.g., "loco_mujoco")
        loader_kwargs: dict, required if Zarr does not exist (e.g., {"env_name": "UnitreeG1", "cfg": ...})
        reference_joint_names: list[str], joint names in reference data order
        target_joint_names: list[str], optional, joint names in target robot order (for mapping)
        datasets: str | list[str] | None, optional, dataset names to load from Zarr
        motions: str | list[str] | None, optional, motion names to load from Zarr
        trajectories: str | list[str] | None, optional, trajectory names to load from Zarr
        keys: str | list[str] | None, optional, keys to load from Zarr (default: all keys)
        refresh_zarr_dataset: bool, if True, delete existing zarr and rebuild it using the loader each run
        reference_start_frame: int, trajectory-local frame index used after each reset (default: 0)
        visualize_reference_arrows: bool, if True show reference velocity/position/heading arrows and
            desired/current frame markers for root and tracked bodies (default: False)

    Example config:
        dataset_path = '/path/to/zarr'
        reset_schedule = 'random'  # or 'sequential', 'round_robin', 'custom'
        wrap_steps = False
        loader_type = 'loco_mujoco'
        loader_kwargs = {'env_name': 'UnitreeG1', 'cfg': {...}}
        reference_joint_names = ['left_hip_pitch_joint', ...]
    """

    def __init__(self, cfg: Any, render_mode: str | None = None, **kwargs: Any) -> None:
        """Initialize the simplified ImitationRLEnv."""
        print(f"[ImitationRLEnv] Starting initialization with num_envs={cfg.scene.num_envs}")

        # Get device
        device = cfg.sim.device
        num_envs = cfg.scene.num_envs

        # Get dataset path and determine if we need to create it
        dataset_path = getattr(cfg, "dataset_path", None)
        loader_type = getattr(cfg, "loader_type", None)
        loader_kwargs = getattr(cfg, "loader_kwargs", {})
        refresh_zarr_dataset = bool(getattr(cfg, "refresh_zarr_dataset", False))

        # Build or load the replay buffer and trajectory info
        if dataset_path is not None:
            dataset_path = Path(dataset_path)
            # Check if it's a directory containing trajectories.zarr or the zarr itself
            if dataset_path.is_dir():
                zarr_path = dataset_path / "trajectories.zarr"
                if not zarr_path.exists():
                    zarr_path = dataset_path  # Assume the directory itself is the zarr
            else:
                zarr_path = dataset_path

            # For debugging, optionally force dataset refresh on every run.
            if refresh_zarr_dataset:
                if loader_type is None:
                    raise ValueError(
                        "refresh_zarr_dataset=True requires loader_type + loader_kwargs "
                        "so the zarr dataset can be rebuilt."
                    )
                if zarr_path.exists():
                    print(f"[ImitationRLEnv] refresh_zarr_dataset=True, removing existing zarr at {zarr_path}...")
                    if zarr_path.is_dir():
                        shutil.rmtree(zarr_path)
                    else:
                        zarr_path.unlink()

            # If zarr doesn't exist and loader is provided, create it
            if not zarr_path.exists() and loader_type is not None:
                print(f"[ImitationRLEnv] Zarr not found at {zarr_path}, creating with {loader_type} loader...")
                if loader_type == "loco_mujoco":
                    from omegaconf import DictConfig

                    loader_cfg = DictConfig(loader_kwargs)
                    print(f"[ImitationRLEnv] Loader cfg: {loader_cfg}")
                    _ = LocoMuJoCoLoader(
                        env_name=loader_kwargs["env_name"],
                        cfg=loader_cfg,
                        build_zarr_dataset=True,
                        zarr_path=str(zarr_path),
                    )
                elif loader_type == "lafan1_csv":
                    from omegaconf import DictConfig

                    loader_cfg_dict = loader_kwargs.get("cfg", loader_kwargs)
                    loader_cfg = DictConfig(loader_cfg_dict)
                    print(f"[ImitationRLEnv] Loader cfg: {loader_cfg}")
                    _ = Lafan1CsvLoader(
                        cfg=loader_cfg,
                        build_zarr_dataset=True,
                        zarr_path=str(zarr_path),
                    )
                else:
                    raise ValueError(f"Unsupported loader_type: {loader_type}")
                print(f"[ImitationRLEnv] Zarr created at {zarr_path}")

            # Load replay buffer from Zarr
            print(f"[ImitationRLEnv] Loading replay buffer from {zarr_path}...")
            datasets = getattr(cfg, "datasets", None)
            motions = getattr(cfg, "motions", None)
            traj_names = getattr(cfg, "trajectories", None)
            keys = getattr(cfg, "keys", None)

            rb, traj_info = make_rb_from(
                zarr_path=str(zarr_path),
                datasets=datasets,
                motions=motions,
                trajectories=traj_names,
                keys=keys,
                device="cpu",
                verbose_tree=False,
            )
        else:
            raise ValueError(
                "Either dataset_path must be provided, or loader_type + loader_kwargs "
                "must be provided to create a new dataset."
            )

        # Map assignment_strategy to reset_schedule (for backward compatibility)
        assignment_strategy = getattr(cfg, "assignment_strategy", None)
        reset_schedule = getattr(cfg, "reset_schedule", None)
        if reset_schedule is None and assignment_strategy is not None:
            # Map old assignment_strategy to new reset_schedule
            mapping = {
                "random": ResetSchedule.RANDOM,
                "sequential": ResetSchedule.SEQUENTIAL,
                "round_robin": ResetSchedule.ROUND_ROBIN,
            }
            reset_schedule = mapping.get(assignment_strategy, ResetSchedule.RANDOM)
            print(
                f"[ImitationRLEnv] Mapped assignment_strategy='{assignment_strategy}' "
                f"to reset_schedule='{reset_schedule}'"
            )
        if reset_schedule is None:
            reset_schedule = ResetSchedule.RANDOM
        print(f"[ImitationRLEnv] Reset schedule: {reset_schedule}")
        # Get other config options
        wrap_steps = getattr(cfg, "wrap_steps", False)
        reference_start_frame = int(getattr(cfg, "reference_start_frame", 0))
        if reference_start_frame < 0:
            raise ValueError("reference_start_frame must be >= 0.")
        reference_joint_names = getattr(cfg, "reference_joint_names", [])
        target_joint_names = getattr(cfg, "target_joint_names", [])

        assert len(reference_joint_names) > 0 and len(target_joint_names) > 0, (
            "Reference and target joint names must have the length greater than 0"
        )

        # Initialize the trajectory manager
        self.trajectory_manager = ParallelTrajectoryManager(
            rb=rb,
            traj_info=traj_info,
            num_envs=num_envs,
            reset_schedule=reset_schedule,
            reset_start_step=reference_start_frame,
            wrap_steps=wrap_steps,
            device=device,
            reference_joint_names=reference_joint_names,
            target_joint_names=target_joint_names,
        )

        # Get initial reference data (this also initializes env assignments)
        self.current_reference: TensorDict = self.trajectory_manager.sample(advance=False)

        # Store reference joint mapping
        self.reference_joint_names = reference_joint_names
        self.reference_body_names: list[str] = []
        self.reference_site_names: list[str] = []
        self._joint_mapping_cache: torch.Tensor | None = None
        self._reference_vel_vis_enabled = bool(
            getattr(cfg, "visualize_reference_arrows", getattr(cfg, "visualize_reference_velocity", False))
        )
        self._reference_vel_marker: VisualizationMarkers | None = None
        self._reference_pos_delta_marker: VisualizationMarkers | None = None
        self._initial_heading_marker: VisualizationMarkers | None = None
        self._goal_root_frame_marker: VisualizationMarkers | None = None
        self._current_root_frame_marker: VisualizationMarkers | None = None
        self._goal_body_frame_markers: list[VisualizationMarkers] = []
        self._current_body_frame_markers: list[VisualizationMarkers] = []
        self._vis_reference_body_ids: torch.Tensor | None = None
        self._vis_robot_body_ids: torch.Tensor | None = None
        self._vis_body_names: list[str] = []
        self._last_tracked_root_pos_w = torch.zeros((num_envs, 3), device=device)
        self._last_tracked_root_pos_valid = torch.zeros((num_envs,), device=device, dtype=torch.bool)
        self.replay_reference = getattr(cfg, "replay_reference", False)
        self.replay_only = getattr(cfg, "replay_only", False)
        if self.replay_only and not self.replay_reference:
            self.replay_reference = True
            print("[ImitationRLEnv] replay_only enabled; forcing replay_reference=True.")

        # Store initial poses for replay
        self._init_root_pos = torch.zeros((num_envs, 3), device=device)
        self._init_root_quat = torch.zeros((num_envs, 4), device=device)
        self._init_root_quat[:, 0] = 1.0
        # Reference root pose at reset frame for each env.
        # This aligns datasets whose xpos/xquat do not start near origin.
        self._reference_reset_root_pos = torch.zeros((num_envs, 3), device=device)
        self._reference_reset_root_quat = torch.zeros((num_envs, 4), device=device)
        self._reference_reset_root_quat[:, 0] = 1.0
        initial_reference_root_pos = self.current_reference.get("root_pos")
        initial_reference_root_quat = self.current_reference.get("root_quat")
        if initial_reference_root_pos is not None:
            self._reference_reset_root_pos.copy_(initial_reference_root_pos)
        if initial_reference_root_quat is not None:
            self._reference_reset_root_quat.copy_(initial_reference_root_quat)
        self._load_reference_metadata(zarr_path)

        # Initialize parent class
        super().__init__(cfg, render_mode, **kwargs)

        self.robot: Articulation = self.scene["robot"]
        self._setup_reference_velocity_visualizer()
        self._update_reference_velocity_visualizer()
        joint_names = self.robot.joint_names
        print("[ImitationRLEnv] G1 Joint names: ", joint_names)

        print("[ImitationRLEnv] Initialization complete")

    def _load_reference_metadata(self, zarr_path: Path) -> None:
        """Load reference body/site names from zarr metadata if available."""
        try:
            root = zarr.open(str(zarr_path), mode="r")
        except Exception as exc:
            print(f"[ImitationRLEnv] Could not open zarr metadata at {zarr_path}: {exc}")
            return

        dataset_group = None
        try:
            group_keys = list(root.group_keys())  # type: ignore[attr-defined]
            for key in group_keys:
                group = root[key]
                if "body_names" in group.attrs:
                    dataset_group = group
                    break
        except Exception:
            dataset_group = None

        if dataset_group is None:
            print("[ImitationRLEnv] No dataset group with body/site metadata found in zarr.")
            return

        body_names = dataset_group.attrs.get("body_names", [])
        site_names = dataset_group.attrs.get("site_names", [])
        self.reference_body_names = list(body_names) if body_names is not None else []
        self.reference_site_names = list(site_names) if site_names is not None else []
        print(
            f"[ImitationRLEnv] Loaded reference metadata: {len(self.reference_body_names)} bodies,"
            f" {len(self.reference_site_names)} sites"
        )

    @staticmethod
    def _normalize_body_name_for_matching(name: str) -> str:
        """Normalize body names for tolerant cross-dataset matching."""
        lowered = name.lower()
        if lowered.endswith("_link"):
            lowered = lowered[:-5]
        return lowered

    def _resolve_reference_body_visualization_pairs(self) -> tuple[torch.Tensor, torch.Tensor, list[str]] | None:
        """Resolve pairs of (reference body idx, robot body idx) to visualize."""
        if len(self.reference_body_names) == 0:
            print("[ImitationRLEnv] Skipping body visualization: no reference body metadata.")
            return None

        reference_body_pos = self.current_reference.get("xpos")
        reference_body_quat = self.current_reference.get("xquat")
        if reference_body_pos is None or reference_body_quat is None:
            print("[ImitationRLEnv] Skipping body visualization: reference keys `xpos`/`xquat` are unavailable.")
            return None

        robot_body_names = list(self.robot.body_names)
        robot_name_lookup = {name: idx for idx, name in enumerate(robot_body_names)}
        robot_name_lookup_lower = {name.lower(): idx for idx, name in enumerate(robot_body_names)}
        robot_normalized_lookup: dict[str, list[int]] = {}
        robot_normalized_names: list[str] = []

        for idx, body_name in enumerate(robot_body_names):
            normalized_name = self._normalize_body_name_for_matching(body_name)
            robot_normalized_names.append(normalized_name)
            robot_normalized_lookup.setdefault(normalized_name, []).append(idx)

        selected_ref_ids: list[int] = []
        selected_robot_ids: list[int] = []
        selected_names: list[str] = []
        unresolved_names: list[str] = []
        used_robot_ids: set[int] = set()

        for ref_id, ref_body_name in enumerate(self.reference_body_names):
            robot_id: int | None = None

            if ref_body_name in robot_name_lookup:
                robot_id = robot_name_lookup[ref_body_name]
            else:
                ref_body_name_lower = ref_body_name.lower()
                if ref_body_name_lower in robot_name_lookup_lower:
                    robot_id = robot_name_lookup_lower[ref_body_name_lower]
                else:
                    normalized_ref_name = self._normalize_body_name_for_matching(ref_body_name)
                    normalized_matches = robot_normalized_lookup.get(normalized_ref_name, [])
                    if len(normalized_matches) > 0:
                        robot_id = normalized_matches[0]
                    else:
                        prefix_matches = [
                            idx
                            for idx, normalized_robot_name in enumerate(robot_normalized_names)
                            if normalized_robot_name.startswith(normalized_ref_name)
                            or normalized_ref_name.startswith(normalized_robot_name)
                        ]
                        if len(prefix_matches) > 0:
                            robot_id = prefix_matches[0]

            if robot_id is None:
                unresolved_names.append(ref_body_name)
                continue
            if robot_id in used_robot_ids:
                continue

            used_robot_ids.add(robot_id)
            selected_ref_ids.append(ref_id)
            selected_robot_ids.append(robot_id)
            selected_names.append(ref_body_name)

        if len(selected_ref_ids) == 0:
            print("[ImitationRLEnv] Skipping body visualization: could not match reference bodies to robot bodies.")
            return None

        if len(unresolved_names) > 0:
            print(
                f"[ImitationRLEnv] Body visualization: unmatched reference bodies (first 10): {unresolved_names[:10]}"
            )

        print(
            f"[ImitationRLEnv] Body visualization: matched {len(selected_ref_ids)}/{len(self.reference_body_names)}"
            " reference bodies."
        )
        return (
            torch.tensor(selected_ref_ids, dtype=torch.long, device=self.device),
            torch.tensor(selected_robot_ids, dtype=torch.long, device=self.device),
            selected_names,
        )

    def _get_reference_alignment_transform(
        self, env_ids: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return per-env rigid transform from dataset world frame to simulation world frame.

        Alignment is yaw-only (rotation around z-axis), to avoid injecting small roll/pitch
        frame mismatches that can tilt reference motion upward/downward.
        """
        if env_ids is None:
            init_pos = self._init_root_pos
            init_quat = self._init_root_quat
            ref_reset_pos = self._reference_reset_root_pos
            ref_reset_quat = self._reference_reset_root_quat
        else:
            init_pos = self._init_root_pos[env_ids]
            init_quat = self._init_root_quat[env_ids]
            ref_reset_pos = self._reference_reset_root_pos[env_ids]
            ref_reset_quat = self._reference_reset_root_quat[env_ids]

        init_yaw_quat = math_utils.yaw_quat(init_quat)
        ref_reset_yaw_quat = math_utils.yaw_quat(ref_reset_quat)
        align_quat = math_utils.quat_mul(init_yaw_quat, math_utils.quat_inv(ref_reset_yaw_quat))
        align_pos = init_pos - math_utils.quat_apply(align_quat, ref_reset_pos)
        return align_quat, align_pos

    def _transform_reference_pose_to_world(
        self, ref_pos: torch.Tensor, ref_quat: torch.Tensor | None = None, env_ids: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply per-episode rigid transform from reference frame to world frame."""
        align_quat, align_pos = self._get_reference_alignment_transform(env_ids)

        if ref_pos.ndim == 2:
            pos_w = math_utils.quat_apply(align_quat, ref_pos) + align_pos
            if ref_quat is None:
                return pos_w, None
            quat_w = math_utils.quat_mul(align_quat, ref_quat)
            return pos_w, quat_w

        if ref_pos.ndim != 3:
            raise ValueError(f"Unsupported ref_pos shape for transform: {tuple(ref_pos.shape)}")

        num_envs, num_items = ref_pos.shape[0], ref_pos.shape[1]
        align_quat_expand = align_quat.unsqueeze(1).expand(-1, num_items, -1).reshape(-1, 4)
        pos_w = math_utils.quat_apply(align_quat_expand, ref_pos.reshape(-1, 3)).reshape(num_envs, num_items, 3)
        pos_w = pos_w + align_pos.unsqueeze(1)

        if ref_quat is None:
            return pos_w, None
        quat_w = math_utils.quat_mul(align_quat_expand, ref_quat.reshape(-1, 4)).reshape(num_envs, num_items, 4)
        return pos_w, quat_w

    def _transform_reference_body_pose_to_init_alignment(
        self, ref_pos: torch.Tensor, ref_quat: torch.Tensor | None = None, env_ids: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Map reference body pose with reset-time alignment so global motion is preserved."""
        return self._transform_reference_pose_to_world(ref_pos, ref_quat, env_ids=env_ids)

    def _reset_idx(self, env_ids: Sequence[int]):
        """Reset the specified environments."""

        if not isinstance(env_ids, torch.Tensor):
            env_ids_tensor = torch.tensor(env_ids, device=self.device, dtype=torch.int64)
        else:
            env_ids_tensor = env_ids.to(dtype=torch.int64)

        # Reset trajectory tracking (reassigns trajectories and resets steps)
        self.trajectory_manager.reset_envs(env_ids_tensor)

        # Get initial reference data for all envs (manager handles indexing).
        # IMPORTANT: advance=False here so that non-reset envs are NOT pushed
        # forward an extra frame.  The per-step advance happens once in step().
        self.current_reference = self.trajectory_manager.sample(advance=False)

        # Trigger the reset events
        result = super()._reset_idx(env_ids_tensor)  # type: ignore

        # Store initial poses for replay
        self._init_root_pos[env_ids_tensor] = self.robot.data.root_state_w[env_ids_tensor, 0:3]
        self._init_root_quat[env_ids_tensor] = self.robot.data.root_state_w[env_ids_tensor, 3:7]
        reference_root_pos = self.current_reference.get("root_pos")
        reference_root_quat = self.current_reference.get("root_quat")
        if reference_root_pos is not None:
            self._reference_reset_root_pos[env_ids_tensor] = reference_root_pos[env_ids_tensor]
        if reference_root_quat is not None:
            self._reference_reset_root_quat[env_ids_tensor] = reference_root_quat[env_ids_tensor]

        if self.replay_reference:
            self._replay_reference(env_ids_tensor)

        tracked_root_pos_w = self._get_tracked_reference_root_pos_w()
        if tracked_root_pos_w is not None:
            self._last_tracked_root_pos_w[env_ids_tensor] = tracked_root_pos_w[env_ids_tensor]
            self._last_tracked_root_pos_valid[env_ids_tensor] = True

        return result

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """Step the environment and update reference data."""
        # Standard RL stepping path.
        if not self.replay_only:
            # Get next reference data point (advance=True to move to next step)
            self.current_reference = self.trajectory_manager.sample(advance=True)
            step_return = super().step(action)
            self._update_reference_velocity_visualizer()
            self._update_env0_velocity_metrics()
            return step_return

        # Replay-only path: ignore physics stepping and evaluate rewards exactly
        # on the replayed reference state.
        self.action_manager.process_action(action.to(self.device))
        self.recorder_manager.record_pre_step()

        # Advance and replay the next reference frame.
        self.current_reference = self.trajectory_manager.sample(advance=True)
        self._replay_reference(reference=self.current_reference)
        self.scene.update(dt=0.0)

        # post-step:
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)
        # -- check terminations
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        # -- reward computation
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        if len(self.recorder_manager.active_terms) > 0:
            # update observations for recording if needed
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        # -- reset envs that terminated/timed-out and log the episode information
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        # Clear any stale terminal info from previous steps.
        for key in ("final_obs", "final_info"):
            if key in self.extras:
                del self.extras[key]

        if len(reset_env_ids) > 0:
            reset_env_ids_list = reset_env_ids.tolist()
            # Populate Gymnasium-style terminal observation info for vector envs.
            # final_obs/final_info are object arrays with None for non-reset envs.
            final_obs = np.empty(self.num_envs, dtype=object)
            final_obs[:] = None
            final_info = np.empty(self.num_envs, dtype=object)
            final_info[:] = None

            def _slice_obs(obs: dict | torch.Tensor, env_id: int):
                if isinstance(obs, dict):
                    return {k: _slice_obs(v, env_id) for k, v in obs.items()}
                return obs[env_id].clone()

            for env_id in reset_env_ids_list:
                final_obs[env_id] = _slice_obs(self.obs_buf, env_id)
                final_info[env_id] = {}

            self.extras["final_obs"] = final_obs
            self.extras["final_info"] = final_info

            # trigger recorder terms for pre-reset calls
            self.recorder_manager.record_pre_reset(reset_env_ids_list)

            self._reset_idx(reset_env_ids_list)

            # if sensors are added to the scene, make sure we render to reflect changes in reset
            if self.sim.has_rtx_sensors() and self.cfg.num_rerenders_on_reset > 0:
                for _ in range(self.cfg.num_rerenders_on_reset):
                    self.sim.render()

            # trigger recorder terms for post-reset calls
            self.recorder_manager.record_post_reset(reset_env_ids_list)

        # -- update command
        self.command_manager.compute(dt=self.step_dt)
        # -- step interval events
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        # -- compute observations
        # note: done after reset to get the correct observations for reset envs
        self.obs_buf = self.observation_manager.compute(update_history=True)
        self._update_reference_velocity_visualizer()
        self._update_env0_velocity_metrics()
        # return observations, rewards, resets and extras
        return (
            self.obs_buf,
            self.reward_buf,
            self.reset_terminated,
            self.reset_time_outs,
            self.extras,
        )

    def get_reference_data(
        self, key: str | None = None, joint_indices: Sequence[int] | None = None
    ) -> TensorDict | torch.Tensor:
        """
        Get the current reference data.

        Args:
            key: Specific key to extract. If None, returns full TensorDict.

        Returns:
            Reference data for all environments
        """
        if self.current_reference is None:
            raise RuntimeError("No reference data available. Call reset() first.")

        if key is None:
            return self.current_reference

        if key not in self.current_reference:
            available_keys = [str(k) for k in self.current_reference.keys()]
            raise KeyError(f"Key '{key}' not found. Available keys: {available_keys}")

        data = self.current_reference[key]
        if joint_indices is not None:
            if isinstance(data, torch.Tensor):
                return data[..., joint_indices]
            else:
                # Handle TensorDict case - data should be a Tensor
                return data[..., joint_indices]  # type: ignore[return-value]
        else:
            return data  # type: ignore[return-value]

    def _replay_reference(self, env_ids: torch.Tensor | None = None, reference: TensorDict | None = None):
        """Replay the reference data. If env_ids is provided, only replay the reference data for the given environments.
        If env_ids is not provided, replay the reference data for all environments."""

        if env_ids is None:
            ref = self.current_reference if reference is None else reference
            defaults_pos = self.robot.data.default_joint_pos
            defaults_vel = self.robot.data.default_joint_vel
        else:
            env_ids_tensor = env_ids
            full_reference = self.current_reference if reference is None else reference
            ref = full_reference[env_ids_tensor]
            defaults_pos = self.robot.data.default_joint_pos[env_ids_tensor]
            defaults_vel = self.robot.data.default_joint_vel[env_ids_tensor]

        root_pos, root_quat_opt = self._transform_reference_pose_to_world(
            ref["root_pos"], ref["root_quat"], env_ids=env_ids
        )
        if root_quat_opt is None:
            raise RuntimeError("Failed to transform reference root quaternion for replay.")
        root_quat = root_quat_opt
        align_quat, _ = self._get_reference_alignment_transform(env_ids)
        root_lin_vel = self._estimate_reference_root_lin_vel_w_from_pos(ref["root_pos"], env_ids=env_ids)
        root_ang_vel = math_utils.quat_apply(align_quat, ref["root_ang_vel"])
        root_pose = torch.cat([root_pos, root_quat], dim=-1)
        root_vel = torch.cat([root_lin_vel, root_ang_vel], dim=-1)
        # Extract joint data from reference TensorDict
        # ref is a TensorDict, so accessing keys returns tensors
        joint_pos_raw = ref["joint_pos"]  # type: ignore[assignment]
        joint_vel_raw = ref["joint_vel"]  # type: ignore[assignment]
        joint_pos = joint_pos_raw.clone()
        joint_vel = joint_vel_raw.clone()

        # Replace NaN positions with default values
        joint_pos = torch.where(torch.isnan(joint_pos), defaults_pos, joint_pos)
        joint_vel = torch.where(torch.isnan(joint_vel), defaults_vel, joint_vel)
        # Use link/com-specific writers so all articulation data buffers stay coherent.
        # `base_lin_vel` uses root_com_vel_w + root_link_quat_w internally.
        self.robot.write_root_link_pose_to_sim(root_pose, env_ids=env_ids)
        self.robot.write_root_com_velocity_to_sim(root_vel, env_ids=env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self.robot.write_data_to_sim()
        # Refresh cached kinematics buffers (e.g. root_lin_vel_b) after direct state writes.
        self.scene.update(dt=0.0)
        self.robot.update(dt=0.0)

    def _get_tracked_reference_root_pos_w(self) -> torch.Tensor | None:
        """Return tracked reference root positions in world frame for all environments."""
        if self.current_reference is None:
            return None

        reference_root_pos = self.current_reference.get("root_pos")
        if reference_root_pos is None:
            return None

        # Apply the full per-episode rigid transform (R, t) from reset frame to world frame.
        tracked_root_pos_w, _ = self._transform_reference_pose_to_world(reference_root_pos)
        return tracked_root_pos_w

    def _estimate_reference_root_lin_vel_w_from_pos(
        self,
        reference_root_pos: torch.Tensor,
        env_ids: torch.Tensor | None = None,
        update_cache: bool = False,
    ) -> torch.Tensor:
        """Estimate reference root linear velocity in world frame from finite differences of root position."""
        if env_ids is None:
            tracked_root_pos_w, _ = self._transform_reference_pose_to_world(reference_root_pos)
            previous_pos_w = self._last_tracked_root_pos_w
            previous_valid = self._last_tracked_root_pos_valid
        else:
            env_ids_tensor = env_ids.to(dtype=torch.int64)
            tracked_root_pos_w, _ = self._transform_reference_pose_to_world(reference_root_pos, env_ids=env_ids_tensor)
            previous_pos_w = self._last_tracked_root_pos_w[env_ids_tensor]
            previous_valid = self._last_tracked_root_pos_valid[env_ids_tensor]

        reference_root_lin_vel_w = torch.zeros_like(tracked_root_pos_w)
        dt = float(self.step_dt)
        if dt > 0.0:
            reference_root_lin_vel_w[previous_valid] = (
                tracked_root_pos_w[previous_valid] - previous_pos_w[previous_valid]
            ) / dt

        if update_cache:
            if env_ids is None:
                self._last_tracked_root_pos_w.copy_(tracked_root_pos_w)
                self._last_tracked_root_pos_valid.fill_(True)
            else:
                env_ids_tensor = env_ids.to(dtype=torch.int64)
                self._last_tracked_root_pos_w[env_ids_tensor] = tracked_root_pos_w
                self._last_tracked_root_pos_valid[env_ids_tensor] = True

        return reference_root_lin_vel_w

    def _setup_reference_velocity_visualizer(self) -> None:
        """Create desired/current frame markers for root and tracked bodies."""
        if not self._reference_vel_vis_enabled:
            return

        # Desired reference body (root) location and current robot root — frame markers like unitree_rl_lab
        goal_cfg = FRAME_MARKER_CFG.copy()
        goal_cfg.prim_path = "/Visuals/Imitation/reference_root_goal"
        goal_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)
        self._goal_root_frame_marker = VisualizationMarkers(goal_cfg)
        self._goal_root_frame_marker.set_visibility(True)
        current_cfg = FRAME_MARKER_CFG.copy()
        current_cfg.prim_path = "/Visuals/Imitation/current_root"
        current_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)
        self._current_root_frame_marker = VisualizationMarkers(current_cfg)
        self._current_root_frame_marker.set_visibility(True)

        body_id_pairs = self._resolve_reference_body_visualization_pairs()
        if body_id_pairs is None:
            return

        self._vis_reference_body_ids, self._vis_robot_body_ids, self._vis_body_names = body_id_pairs
        for body_name in self._vis_body_names:
            current_body_cfg = FRAME_MARKER_CFG.copy()
            current_body_cfg.prim_path = f"/Visuals/Imitation/current_body/{body_name}"
            current_body_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
            current_body_marker = VisualizationMarkers(current_body_cfg)
            current_body_marker.set_visibility(True)
            self._current_body_frame_markers.append(current_body_marker)

            goal_body_cfg = FRAME_MARKER_CFG.copy()
            goal_body_cfg.prim_path = f"/Visuals/Imitation/reference_body_goal/{body_name}"
            goal_body_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
            goal_body_marker = VisualizationMarkers(goal_body_cfg)
            goal_body_marker.set_visibility(True)
            self._goal_body_frame_markers.append(goal_body_marker)

    def _update_reference_velocity_visualizer(self) -> None:
        """Update marker pose/scale from current reference linear velocity."""
        if not self._reference_vel_vis_enabled:
            return
        if self.current_reference is None:
            return
        if not self.robot.is_initialized:
            return

        tracked_root_pos_w = self._get_tracked_reference_root_pos_w()

        # Desired reference body (root) location and current robot root — frame markers like unitree_rl_lab
        if self._goal_root_frame_marker is not None and self._current_root_frame_marker is not None:
            ref_root_pos_w = tracked_root_pos_w
            align_quat, _ = self._get_reference_alignment_transform()
            ref_root_quat_w = math_utils.quat_mul(align_quat, self.current_reference["root_quat"])
            if ref_root_pos_w is not None:
                self._goal_root_frame_marker.visualize(translations=ref_root_pos_w, orientations=ref_root_quat_w)
            self._current_root_frame_marker.visualize(
                translations=self.robot.data.root_pos_w, orientations=self.robot.data.root_quat_w
            )

        if (
            self._vis_reference_body_ids is not None
            and self._vis_robot_body_ids is not None
            and len(self._goal_body_frame_markers) == len(self._current_body_frame_markers)
            and len(self._goal_body_frame_markers) > 0
        ):
            reference_body_pos = self.current_reference.get("xpos")
            reference_body_quat = self.current_reference.get("xquat")
            if reference_body_pos is not None and reference_body_quat is not None:
                ref_body_pos = reference_body_pos[..., self._vis_reference_body_ids, :]
                ref_body_quat = reference_body_quat[..., self._vis_reference_body_ids, :]
                ref_body_pos_w, ref_body_quat_w_opt = self._transform_reference_body_pose_to_init_alignment(
                    ref_body_pos, ref_body_quat
                )
                if ref_body_quat_w_opt is None:
                    return
                ref_body_quat_w = ref_body_quat_w_opt
                num_bodies = ref_body_pos.shape[1]

                robot_body_pos_w = self.robot.data.body_pos_w[:, self._vis_robot_body_ids]
                robot_body_quat_w = self.robot.data.body_quat_w[:, self._vis_robot_body_ids]

                for body_index in range(num_bodies):
                    self._current_body_frame_markers[body_index].visualize(
                        robot_body_pos_w[:, body_index], robot_body_quat_w[:, body_index]
                    )
                    self._goal_body_frame_markers[body_index].visualize(
                        ref_body_pos_w[:, body_index], ref_body_quat_w[:, body_index]
                    )

    def _update_env0_velocity_metrics(self) -> None:
        """Expose env[0] velocity tracking metrics in extras for easy logging."""
        if self.current_reference is None or self.num_envs < 1:
            return
        reference_root_pos = self.current_reference.get("root_pos")
        if reference_root_pos is None:
            return

        reference_root_lin_vel_w = self._estimate_reference_root_lin_vel_w_from_pos(
            reference_root_pos, update_cache=True
        )
        reference_vel_env0 = reference_root_lin_vel_w[0]
        actual_vel_env0 = self.robot.data.root_lin_vel_w[0]
        diff_vel_env0 = actual_vel_env0 - reference_vel_env0

        self.extras["metrics/env0/reference_root_lin_vel_x"] = reference_vel_env0[0].item()
        self.extras["metrics/env0/reference_root_lin_vel_y"] = reference_vel_env0[1].item()
        self.extras["metrics/env0/reference_root_lin_vel_z"] = reference_vel_env0[2].item()
        self.extras["metrics/env0/actual_root_lin_vel_x"] = actual_vel_env0[0].item()
        self.extras["metrics/env0/actual_root_lin_vel_y"] = actual_vel_env0[1].item()
        self.extras["metrics/env0/actual_root_lin_vel_z"] = actual_vel_env0[2].item()
        self.extras["metrics/env0/root_lin_vel_diff_x"] = diff_vel_env0[0].item()
        self.extras["metrics/env0/root_lin_vel_diff_y"] = diff_vel_env0[1].item()
        self.extras["metrics/env0/root_lin_vel_diff_z"] = diff_vel_env0[2].item()
        self.extras["metrics/env0/root_lin_vel_diff_norm"] = torch.linalg.norm(diff_vel_env0).item()
