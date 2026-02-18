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
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG, RED_ARROW_X_MARKER_CFG

# Import the new manager and utilities
try:
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
        self._reference_vel_vis_enabled = bool(getattr(cfg, "visualize_reference_velocity", True))
        self._reference_vel_marker: VisualizationMarkers | None = None
        self._reference_pos_delta_marker: VisualizationMarkers | None = None
        self._initial_heading_marker: VisualizationMarkers | None = None
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

    def _reset_idx(self, env_ids: Sequence[int]):
        """Reset the specified environments."""

        if not isinstance(env_ids, torch.Tensor):
            env_ids_tensor = torch.tensor(env_ids, device=self.device, dtype=torch.int64)
        else:
            env_ids_tensor = env_ids.to(dtype=torch.int64)

        # Reset trajectory tracking (reassigns trajectories and resets steps)
        self.trajectory_manager.reset_envs(env_ids_tensor)

        # Get initial reference data for all envs (manager handles indexing)
        # Keep the reset frame as-is so replay starts from the first frame.
        self.current_reference = self.trajectory_manager.sample(advance=False)

        # Trigger the reset events
        result = super()._reset_idx(env_ids_tensor)  # type: ignore

        # Store initial poses for replay
        self._init_root_pos[env_ids_tensor] = self.robot.data.root_state_w[env_ids_tensor, 0:3]
        self._init_root_quat[env_ids_tensor] = self.robot.data.root_state_w[env_ids_tensor, 3:7]

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
            init_pos = self._init_root_pos
            init_quat = self._init_root_quat
            ref = self.current_reference if reference is None else reference
            defaults_pos = self.robot.data.default_joint_pos
            defaults_vel = self.robot.data.default_joint_vel
        else:
            env_ids_tensor = env_ids
            init_pos = self._init_root_pos[env_ids_tensor]
            init_quat = self._init_root_quat[env_ids_tensor]
            full_reference = self.current_reference if reference is None else reference
            ref = full_reference[env_ids_tensor]
            defaults_pos = self.robot.data.default_joint_pos[env_ids_tensor]
            defaults_vel = self.robot.data.default_joint_vel[env_ids_tensor]

        # Rotate reference root_pos by initial orientation, then translate by initial position
        root_pos = math_utils.quat_apply(init_quat, ref["root_pos"])
        root_pos[..., :2] += init_pos[..., :2]
        root_pos[..., 2] = init_pos[..., 2]
        root_quat = math_utils.quat_mul(init_quat, ref["root_quat"])
        root_lin_vel = math_utils.quat_apply(init_quat, ref["root_lin_vel"])
        root_ang_vel = math_utils.quat_apply(init_quat, ref["root_ang_vel"])
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

        tracked_root_pos_w = math_utils.quat_apply(self._init_root_quat, reference_root_pos)
        tracked_root_pos_w[:, :2] += self._init_root_pos[:, :2]
        tracked_root_pos_w[:, 2] = self._init_root_pos[:, 2]
        return tracked_root_pos_w

    def _setup_reference_velocity_visualizer(self) -> None:
        """Create the marker used to visualize reference linear velocity."""
        if not self._reference_vel_vis_enabled:
            return
        marker_cfg = RED_ARROW_X_MARKER_CFG.copy()
        marker_cfg.prim_path = "/Visuals/Imitation/reference_root_lin_vel"
        marker_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
        self._reference_vel_marker = VisualizationMarkers(marker_cfg)
        self._reference_vel_marker.set_visibility(True)

        pos_delta_cfg = BLUE_ARROW_X_MARKER_CFG.copy()
        pos_delta_cfg.prim_path = "/Visuals/Imitation/reference_root_pos_delta"
        pos_delta_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
        self._reference_pos_delta_marker = VisualizationMarkers(pos_delta_cfg)
        self._reference_pos_delta_marker.set_visibility(True)

        heading_cfg = GREEN_ARROW_X_MARKER_CFG.copy()
        heading_cfg.prim_path = "/Visuals/Imitation/reference_initial_heading"
        heading_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
        self._initial_heading_marker = VisualizationMarkers(heading_cfg)
        self._initial_heading_marker.set_visibility(True)

    def _update_reference_velocity_visualizer(self) -> None:
        """Update marker pose/scale from current reference linear velocity."""
        if not self._reference_vel_vis_enabled:
            return
        if self.current_reference is None:
            return
        if not self.robot.is_initialized:
            return

        marker_pos_w = self.robot.data.root_pos_w.clone()
        marker_pos_w[:, 2] += 0.5

        if self._reference_vel_marker is not None:
            reference_root_lin_vel = self.current_reference.get("root_lin_vel")
            if reference_root_lin_vel is not None:
                # Convert reference velocity to world frame using the reset-frame orientation.
                reference_root_lin_vel_w = math_utils.quat_apply(self._init_root_quat, reference_root_lin_vel)
                reference_root_lin_vel_xy_w = reference_root_lin_vel_w[:, :2]

                default_scale = self._reference_vel_marker.cfg.markers["arrow"].scale
                marker_scale = torch.tensor(default_scale, device=self.device).repeat(self.num_envs, 1)
                marker_scale[:, 0] *= torch.linalg.norm(reference_root_lin_vel_xy_w, dim=1) * 3.0

                heading_angle = torch.atan2(reference_root_lin_vel_xy_w[:, 1], reference_root_lin_vel_xy_w[:, 0])
                zeros = torch.zeros_like(heading_angle)
                marker_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
                self._reference_vel_marker.visualize(marker_pos_w, marker_quat, marker_scale)

        if self._reference_pos_delta_marker is not None:
            tracked_root_pos_w = self._get_tracked_reference_root_pos_w()
            if tracked_root_pos_w is not None:
                tracked_root_pos_delta_w = tracked_root_pos_w - self._last_tracked_root_pos_w
                tracked_root_pos_delta_w[~self._last_tracked_root_pos_valid] = 0.0
                tracked_root_pos_delta_xy_w = tracked_root_pos_delta_w[:, :2]

                delta_marker_pos_w = marker_pos_w.clone()
                delta_marker_pos_w[:, 2] += 0.2

                delta_default_scale = self._reference_pos_delta_marker.cfg.markers["arrow"].scale
                delta_marker_scale = torch.tensor(delta_default_scale, device=self.device).repeat(self.num_envs, 1)
                delta_scale_gain = 3.0 / max(float(self.step_dt), 1.0e-6)
                delta_marker_scale[:, 0] *= torch.linalg.norm(tracked_root_pos_delta_xy_w, dim=1) * delta_scale_gain

                delta_heading_angle = torch.atan2(tracked_root_pos_delta_xy_w[:, 1], tracked_root_pos_delta_xy_w[:, 0])
                zeros = torch.zeros_like(delta_heading_angle)
                delta_marker_quat = math_utils.quat_from_euler_xyz(zeros, zeros, delta_heading_angle)
                self._reference_pos_delta_marker.visualize(delta_marker_pos_w, delta_marker_quat, delta_marker_scale)

                self._last_tracked_root_pos_w.copy_(tracked_root_pos_w)
                self._last_tracked_root_pos_valid.fill_(True)

        if self._initial_heading_marker is not None:
            heading_marker_pos_w = self.robot.data.root_pos_w.clone()
            heading_marker_pos_w[:, 2] += 0.8
            heading_default_scale = self._initial_heading_marker.cfg.markers["arrow"].scale
            heading_marker_scale = torch.tensor(heading_default_scale, device=self.device).repeat(self.num_envs, 1)
            self._initial_heading_marker.visualize(heading_marker_pos_w, self._init_root_quat, heading_marker_scale)
