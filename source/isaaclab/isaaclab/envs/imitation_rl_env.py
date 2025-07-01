import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor

import torch

# Import dataset utilities from ImitationLearningTools
from iltools_datasets import ZarrBackedTrajectoryDataset, export_trajectories_to_zarr
from iltools_datasets.utils import ZarrTrajectoryWindowCache

from .common import VecEnvStepReturn
from .manager_based_rl_env import ManagerBasedRLEnv


class ImitationRLEnv(ManagerBasedRLEnv):
    """
    Generic RL environment for imitation learning with per-env trajectory tracking and efficient windowed access.
    Uses ZarrTrajectoryWindowCache for per-env rolling window caching, and async prefetching for speed.

    Config attributes (cfg):
        dataset_path: str, path to Zarr dataset directory
        window_size: int, optional, window size for sequence sampling
        batch_size: int, optional, batch size for prefetching
        device: str, optional, torch device (default: env device)
        loader_type: str, required if Zarr does not exist (e.g., 'loco_mujoco', 'amass', 'trajopt')
        loader_kwargs: dict, required if Zarr does not exist (kwargs for the loader)

    Example config:
        dataset_path = '/path/to/zarr'
        window_size = 5
        batch_size = 2
        device = 'cuda'
        loader_type = 'loco_mujoco'
        loader_kwargs = {'env_name': 'UnitreeG1', 'task': 'walk'}
    """

    _prefetch_futures: list[Future | None]

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        device = getattr(cfg, "device", self.device)
        window_size = getattr(cfg, "window_size", 64)
        batch_size = getattr(cfg, "batch_size", 1)
        dataset_path = getattr(cfg, "dataset_path", None)
        if dataset_path is None:
            raise ValueError(
                "dataset_path must be provided in the config for ImitationRLEnv."
            )
        zarr_dir = os.path.join(dataset_path, "trajectories.zarr")
        meta_file = os.path.join(dataset_path, "metadata.json")
        zarr_exists = os.path.exists(zarr_dir) and os.path.exists(meta_file)
        if not zarr_exists:
            loader_type = getattr(cfg, "loader_type", None)
            loader_kwargs = getattr(cfg, "loader_kwargs", None)
            if loader_type is None or loader_kwargs is None:
                raise RuntimeError(
                    "Zarr dataset not found and loader_type/loader_kwargs not provided in config. "
                    "Please specify how to generate the dataset."
                )
            # Import only the required loader
            if loader_type == "loco_mujoco":
                from iltools_datasets.loco_mujoco.loader import LocoMuJoCoLoader

                loader = LocoMuJoCoLoader(**loader_kwargs)
            elif loader_type == "amass":
                from iltools_datasets.amass.loader import AmassLoader

                loader = AmassLoader(**loader_kwargs)
            elif loader_type == "trajopt":
                from iltools_datasets.trajopt.loader import TrajoptLoader

                loader = TrajoptLoader(**loader_kwargs)
            else:
                raise ValueError(f"Unknown loader_type: {loader_type}")
            export_trajectories_to_zarr(
                loader,
                dataset_path,
                window_size=window_size,
                control_freq=1.0 / (self.cfg.sim.dt * self.cfg.decimation),
                desired_horizon_steps=int(
                    self.cfg.episode_length_s / (self.cfg.sim.dt * self.cfg.decimation)
                ),
                horizon_multiplier=2.0,
            )
        # Now load the Zarr dataset
        self.dataset = ZarrBackedTrajectoryDataset(
            dataset_path,
            window_size=window_size,
            device=device,
            batch_size=batch_size,
        )
        # --- Per-trajectory window counts ---
        self.traj_lengths = torch.tensor(
            self.dataset.lengths, device=device, dtype=torch.long
        )
        self.num_trajectories = len(self.traj_lengths)
        self.window_size = window_size
        # Compute number of windows per trajectory
        self.traj_num_windows = torch.tensor(
            [max(1, int(length) - window_size + 1) for length in self.traj_lengths],
            device=device,
            dtype=torch.long,
        )
        self.traj_window_offsets = torch.cat(
            [
                torch.tensor([0], device=device, dtype=torch.long),
                torch.cumsum(self.traj_num_windows, dim=0)[:-1],
            ]
        )
        self.total_windows = int(self.traj_num_windows.sum().item())
        # Assign each env a trajectory and a step within that trajectory
        self.env2traj = torch.empty(self.num_envs, dtype=torch.long, device=device)
        self.env2step = torch.zeros(self.num_envs, device=device, dtype=torch.long)
        self.assign_trajectories(torch.arange(self.num_envs, device=device))
        # Efficient per-env window cache
        self.cache = ZarrTrajectoryWindowCache(
            self.dataset, window_size=window_size, device=device
        )
        # Async prefetching: thread pool and per-env prefetch state
        self._prefetch_executor = ThreadPoolExecutor(max_workers=8)
        self._prefetch_futures = [None for _ in range(self.num_envs)]
        self._prefetch_lock = threading.Lock()
        # --- Add replay_reference option ---
        self.replay_reference = getattr(cfg, "replay_reference", False)
        # --- Initialize attributes for linter/type safety ---
        self.ref_qpos = None
        self.ref_root_pos = None  # NEW: reference root position
        self.ref_root_rot = None  # NEW: reference root orientation
        self._qpos_mapping: list[int] = []
        self._qpos_inv_mapping: list[int] = []
        self._root_pos_idx: list[int] = []
        self._root_rot_idx: list[int] = []
        # --- Debug flag for reference check ---
        self.debug_reference_check = (
            False  # Set to True to enable debug check in step()
        )
        self._debug_loader = (
            None  # Set this to your loader instance if you want to check
        )
        if self.debug_reference_check:
            loader_type = getattr(cfg, "loader_type", None)
            loader_kwargs = getattr(cfg, "loader_kwargs", None)
            # Import only the required loader
            if loader_type == "loco_mujoco":
                from iltools_datasets.loco_mujoco.loader import LocoMuJoCoLoader

                self._debug_loader = LocoMuJoCoLoader(**loader_kwargs)  # type: ignore
            elif loader_type == "amass":
                from iltools_datasets.amass.loader import AmassLoader

                self._debug_loader = AmassLoader(**loader_kwargs)  # type: ignore
            elif loader_type == "trajopt":
                from iltools_datasets.trajopt.loader import TrajoptLoader

                self._debug_loader = TrajoptLoader(**loader_kwargs)  # type: ignore
            else:
                raise ValueError(f"Unknown loader_type: {loader_type}")

    def assign_trajectories(self, env_ids):
        # Default: random assignment
        self.env2traj[env_ids] = torch.randint(
            0, self.num_trajectories, (len(env_ids),), device=self.device
        )
        # Optionally: implement round-robin, curriculum, or fixed assignment here

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        # Assign new trajectory to each reset env
        self.assign_trajectories(env_ids)
        self.env2step[env_ids] = 0
        # Cancel any outstanding prefetches for these envs
        with self._prefetch_lock:
            for env in env_ids:
                fut = self._prefetch_futures[env]
                if fut is not None and not fut.done():
                    fut.cancel()
                self._prefetch_futures[env] = None  # type: ignore
        # Optionally prefetch the first window for these envs
        for env in env_ids:
            self._maybe_prefetch(env, self.env2traj[env].item(), 0)

    def _get_window_index(self, traj_idx, step_idx):
        # Map (traj_idx, step_idx) to global window index in the Zarr dataset
        return self.traj_window_offsets[traj_idx] + step_idx

    def _maybe_prefetch(self, env_idx, traj_idx, step_idx):
        """
        If not already prefetched, prefetch the window for (traj_idx, step_idx) in a background thread.
        """

        def prefetch():
            # This will populate the cache for the next window
            self.cache.dataset[self._get_window_index(traj_idx, step_idx)]

        with self._prefetch_lock:
            fut = self._prefetch_futures[env_idx]
            if fut is not None and not fut.done():
                return  # Already prefetching
            self._prefetch_futures[env_idx] = self._prefetch_executor.submit(prefetch)

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        # Advance time for each env
        self.env2step += 1
        # Clamp to valid range for each env (0 to num_windows-1 for that trajectory)
        max_step = self.traj_num_windows[self.env2traj] - 1
        self.env2step = torch.minimum(self.env2step, max_step)
        # Async prefetch: if next step will move out of window, prefetch next window
        next_step = self.env2step + 1
        need_prefetch = next_step < self.traj_num_windows[self.env2traj]
        envs_to_prefetch = torch.where(need_prefetch)[0]
        for i in envs_to_prefetch:
            self._maybe_prefetch(i.item(), self.env2traj[i].item(), next_step[i].item())
        # Compute and store the current reference qpos for all envs
        self.ref_qpos = self.compute_reference(key="qpos")
        # Also extract and store reference root position and orientation for all envs
        if self._root_pos_idx:
            self.ref_root_pos = self.ref_qpos[:, self._root_pos_idx]
        else:
            self.ref_root_pos = None
        if self._root_rot_idx:
            self.ref_root_rot = self.ref_qpos[:, self._root_rot_idx]
        else:
            self.ref_root_rot = None

        # --- Replay reference if enabled ---
        if self.replay_reference:
            isaaclab_joint_names = self.scene["robot"].joint_names
            reference_joint_names = self.get_reference_joint_names()
            if not hasattr(self, "_qpos_mapping") or not self._qpos_mapping:
                self.compute_qpos_mapping(reference_joint_names)
            # --- Robust root pose construction ---
            canonical_root_order = [
                "root_x",
                "root_y",
                "root_z",
                "root_qw",
                "root_qx",
                "root_qy",
                "root_qz",
            ]
            root_pose_list = []
            for root_name in canonical_root_order:
                if root_name in reference_joint_names:
                    idx = reference_joint_names.index(root_name)
                    root_pose_list.append(
                        self.ref_qpos[:, idx : idx + 1]
                    )  # (num_envs, 1)
                else:
                    # Fill with sensible default
                    if root_name == "root_x":
                        # Use env origin x if available, else 0
                        default_val = (
                            self.scene.env_origins[:, 0:1]
                            if hasattr(self.scene, "env_origins")
                            else torch.zeros((self.num_envs, 1), device=self.device)
                        )
                    elif root_name == "root_y":
                        # Use env origin y if available, else 0
                        default_val = (
                            self.scene.env_origins[:, 1:2]
                            if hasattr(self.scene, "env_origins")
                            else torch.zeros((self.num_envs, 1), device=self.device)
                        )
                    elif root_name == "root_z":
                        # Use env origin z if available, else 0
                        default_val = (
                            self.scene.env_origins[:, 2:3]
                            if hasattr(self.scene, "env_origins")
                            else torch.zeros((self.num_envs, 1), device=self.device)
                        )
                    elif root_name == "root_qw":
                        default_val = torch.ones((self.num_envs, 1), device=self.device)
                    else:
                        default_val = torch.zeros(
                            (self.num_envs, 1), device=self.device
                        )
                    root_pose_list.append(default_val)
            root_pose = torch.cat(root_pose_list, dim=1)  # (num_envs, 7)
            default_root_state = self.scene["robot"].data.default_root_state.clone()
            default_root_state[..., :7] = root_pose
            default_root_state[..., :3] += self.scene.env_origins
            self.scene["robot"].write_root_state_to_sim(default_root_state)
            # --- Set joint positions from reference (excluding root states) ---
            # Find indices in reference_joint_names that are actual joints (not root states)
            root_names = [
                reference_joint_names[i]
                for i in (self._root_pos_idx + self._root_rot_idx)
            ]
            joint_indices_ref = [
                i for i, n in enumerate(reference_joint_names) if n not in root_names
            ]
            joint_names_ref = [reference_joint_names[i] for i in joint_indices_ref]
            # Map these to IsaacLab joint indices (only those present in isaaclab_joint_names)
            joint_indices_isaac = [
                isaaclab_joint_names.index(n)
                for n in joint_names_ref
                if n in isaaclab_joint_names
            ]
            # Prepare joint position array (num_envs, num_joints)
            qpos_joints = torch.zeros(
                (self.ref_qpos.shape[0], len(isaaclab_joint_names)),
                device=self.ref_qpos.device,
            )
            # Fill in the mapped joint positions
            for idx_ref, name in zip(joint_indices_ref, joint_names_ref):
                if name in isaaclab_joint_names:
                    idx_isaac = isaaclab_joint_names.index(name)
                    qpos_joints[:, idx_isaac] = self.ref_qpos[:, idx_ref]
            self.scene["robot"].write_joint_state_to_sim(
                qpos_joints,
                self.scene["robot"].data.default_joint_vel.clone(),
            )
            self.scene.write_data_to_sim()
        # Call parent step
        return super().step(action)

    def compute_reference(self, key="qpos"):
        """
        Fully vectorized: get the reference state for each env at the current time index using batch_get.
        Args:
            key: Which observation key to extract (e.g., 'qpos').
        Returns:
            refs: torch.Tensor of shape (num_envs, ...)
        """
        traj_indices = self.env2traj
        window_indices = self.env2step
        # Always use the first step in the window for standard imitation
        step_indices = torch.zeros_like(window_indices)
        # Use the underlying dataset's batch_get directly for (window_idx, step_in_window)
        batch = self.cache.dataset.batch_get(
            window_indices, step_indices, key=key, data_type="observations"
        )
        # Defensive: ensure output is always 2D (num_envs, qpos_dim)
        if batch.ndim == 1:
            batch = batch.unsqueeze(0)
        return batch

    def compute_qpos_mapping(self, reference_joint_names=None):
        """
        Compute mapping between IsaacLab qpos order and a reference qpos order (configurable).
        Also separates root position and orientation indices for convenience.
        Args:
            reference_joint_names: list of joint names in reference qpos order. If None, uses cfg.reference_joint_names or self.get_loco_joint_names().
        Returns:
            mapping: list of indices such that qpos_reference = qpos_isaaclab[mapping]
            inv_mapping: list of indices such that qpos_isaaclab = qpos_reference[inv_mapping]
        """
        isaaclab_joint_names = self.scene["robot"].joint_names
        # Allow joint name list to be set via config
        if reference_joint_names is None:
            reference_joint_names = getattr(self.cfg, "reference_joint_names", None)
        if reference_joint_names is None:
            reference_joint_names = self.get_loco_joint_names()
        # Build mapping: IsaacLab -> Reference
        mapping = []
        for name in reference_joint_names:
            if name not in isaaclab_joint_names or name not in reference_joint_names:
                continue
            mapping.append(isaaclab_joint_names.index(name))
        # Build inverse mapping: Reference -> IsaacLab
        inv_mapping = []
        for name in isaaclab_joint_names:
            if name not in reference_joint_names or name not in isaaclab_joint_names:
                continue
            inv_mapping.append(reference_joint_names.index(name))
        self._qpos_mapping = mapping
        self._qpos_inv_mapping = inv_mapping
        # Identify root position and orientation indices (assume standard naming)
        root_pos_names = ["root_x", "root_y", "root_z"]
        root_rot_names = ["root_qw", "root_qx", "root_qy", "root_qz"]
        self._root_pos_idx = [
            i for i, name in enumerate(reference_joint_names) if name in root_pos_names
        ]
        self._root_rot_idx = [
            i for i, name in enumerate(reference_joint_names) if name in root_rot_names
        ]
        # --- Debug printouts for mapping verification ---
        missing_in_isaac = [
            name for name in reference_joint_names if name not in isaaclab_joint_names
        ]
        missing_in_ref = [
            name for name in isaaclab_joint_names if name not in reference_joint_names
        ]
        # if missing_in_isaac:
        #     print(
        #         f"[WARNING] Joints in reference but not in IsaacLab: {missing_in_isaac}"
        #     )
        # if missing_in_ref:
        #     print(
        #         f"[WARNING] Joints in IsaacLab but not in reference: {missing_in_ref}"
        #     )
        # print(f"[DEBUG] Reference joint order: {reference_joint_names}")
        # print(f"[DEBUG] IsaacLab joint order: {isaaclab_joint_names}")
        # print(f"[DEBUG] Reference->IsaacLab mapping indices: {mapping}")
        # print(f"[DEBUG] IsaacLab->Reference mapping indices: {inv_mapping}")
        # ---
        return mapping, inv_mapping

    def get_loco_joint_names(self):
        """
        Return the list of joint names in loco-mujoco qpos order for this robot.
        Override or set this method/config per robot.
        """
        # Example for UnitreeG1 (should generalize for other robots)
        return [
            "root_z",
            "root_qx",
            "root_qy",
            "root_qz",
            "root_qw",
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
            "torso_joint",
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_pitch_joint",
            "left_elbow_roll_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_pitch_joint",
            "right_elbow_roll_joint",
        ]

    def get_reference_joint_names(self):
        """
        Return the list of joint names in the reference qpos order for this robot.
        Override or set this method/config per robot. By default, uses cfg.reference_joint_names or falls back to get_loco_joint_names().
        """
        joint_names = getattr(self.cfg, "reference_joint_names", None)
        if joint_names is not None:
            return joint_names
        return self.get_loco_joint_names()

    def get_qpos_in_loco_order(self, qpos: torch.Tensor | None = None) -> torch.Tensor:
        """
        Return IsaacLab qpos in loco-mujoco order using the mapping.
        Args:
            qpos: (optional) qpos array to convert. If None, uses self.qpos.
        Returns:
            qpos_loco_order: np.ndarray or torch.Tensor in loco-mujoco order.
        """
        if qpos is None:
            qpos = getattr(self, "qpos", None)
            if qpos is None:
                raise AttributeError(
                    "self.qpos is not set. Please set self.qpos before calling get_qpos_in_loco_order, or pass qpos explicitly."
                )
        if not hasattr(self, "_qpos_mapping"):
            self.compute_qpos_mapping()
        mapping = self._qpos_mapping
        return qpos[..., mapping]

    def get_qpos_in_isaaclab_order(
        self, qpos_loco: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Convert qpos from loco-mujoco order to IsaacLab order using the inverse mapping.
        Args:
            qpos_loco: (optional) qpos array in loco-mujoco order. If None, uses self.qpos_loco.
        Returns:
            qpos_isaaclab_order: np.ndarray or torch.Tensor in IsaacLab order.
        """
        if qpos_loco is None:
            qpos_loco = getattr(self, "qpos_loco", None)
            if qpos_loco is None:
                raise AttributeError(
                    "self.qpos_loco is not set. Please set self.qpos_loco before calling get_qpos_in_isaaclab_order, or pass qpos_loco explicitly."
                )
        if not hasattr(self, "_qpos_inv_mapping"):
            self.compute_qpos_mapping()
        inv_mapping = self._qpos_inv_mapping
        return qpos_loco[..., inv_mapping]

    def debug_reference_vs_loader(
        self, loader, num_envs_to_check=3, num_steps=10, key="qpos"
    ):
        """
        Debug utility to compare reference data fetched from the Zarr dataset (via env) and the original loader.
        Args:
            loader: The original trajectory loader (e.g., LocoMuJoCoLoader).
            num_envs_to_check: Number of environments to check.
            num_steps: Number of steps to check for each env.
            key: Which observation key to check (e.g., 'qpos').
        """
        print("=== Debug: Reference vs Loader ===")
        for env_id in range(min(num_envs_to_check, self.num_envs)):
            traj_idx = self.env2traj[env_id].item()
            print(f"\n[Env {env_id}] Trajectory index: {traj_idx}")
            # Get the original trajectory from the loader
            orig_traj = loader[traj_idx]
            orig_qpos = orig_traj.observations[key]
            orig_dt = orig_traj.dt
            print(f"  Loader qpos shape: {orig_qpos.shape}, dt: {orig_dt}")
            # For each step, compare the reference from env and the loader
            for step in range(
                min(num_steps, self.traj_num_windows[traj_idx].item())  # type: ignore
            ):
                # Set env2step for this env to the step we want to check
                self.env2step[env_id] = step
                ref_qpos = self.compute_reference(key=key)[env_id].cpu().numpy()
                # The Zarr window starts at 'step', so the first element should match orig_qpos[step]
                orig_qpos_step = orig_qpos[step]
                diff = ((ref_qpos - orig_qpos_step) ** 2).mean() ** 0.5
                print(f"    Step {step}: RMSE={diff:.6f}")
                if diff > 1e-5:
                    print(f"      [WARNING] Mismatch at step {step} (env {env_id})")
        print("=== End Debug ===")
