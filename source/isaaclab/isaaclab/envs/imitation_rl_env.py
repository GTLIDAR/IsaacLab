import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Optional, Union, Sequence

import torch

# Import dataset utilities from ImitationLearningTools
from iltools_datasets import (
    ZarrBackedTrajectoryDataset,
    export_trajectories_to_zarr,
    LocoMuJoCoLoader,
)
from iltools_datasets.amass.loader import AmassLoader
from iltools_datasets.trajopt.loader import TrajoptLoader
from iltools_datasets.utils import ZarrTrajectoryWindowCache

from .common import VecEnvStepReturn
from .manager_based_rl_env import ManagerBasedRLEnv


# === Type Aliases ===
LoaderType = Union["LocoMuJoCoLoader", "AmassLoader", "TrajoptLoader"]


# === Helper Functions for Loader and Dataset Management ===
def _get_loader(loader_type: str, loader_kwargs: dict[str, Any]) -> LoaderType:
    """
    Dynamically import and instantiate the correct loader based on loader_type.

    Args:
        loader_type: One of 'loco_mujoco', 'amass', 'trajopt'.
        loader_kwargs: Arguments for the loader.

    Returns:
        Loader instance.

    Example:
        loader = _get_loader('loco_mujoco', {'env_name': 'UnitreeG1', 'task': 'walk'})
    """
    if loader_type == "loco_mujoco":
        from iltools_datasets.loco_mujoco.loader import LocoMuJoCoLoader

        return LocoMuJoCoLoader(**loader_kwargs)
    elif loader_type == "amass":
        from iltools_datasets.amass.loader import AmassLoader

        return AmassLoader(**loader_kwargs)
    elif loader_type == "trajopt":
        from iltools_datasets.trajopt.loader import TrajoptLoader

        return TrajoptLoader(**loader_kwargs)
    else:
        raise ValueError(f"Unknown loader_type: {loader_type}")


def _check_zarr_exists(dataset_path: str) -> bool:
    """
    Check if both the Zarr directory and metadata file exist in the dataset path.

    Args:
        dataset_path: Path to the dataset directory.

    Returns:
        True if both exist, False otherwise.
    """
    zarr_dir = os.path.join(dataset_path, "trajectories.zarr")
    meta_file = os.path.join(dataset_path, "metadata.json")
    return os.path.exists(zarr_dir) and os.path.exists(meta_file)


def _get_control_freq(cfg: Any) -> float:
    """
    Compute the control frequency from the config.

    Returns:
        Control frequency (Hz).
    """
    return 1.0 / (cfg.sim.dt * cfg.decimation)


def _get_desired_horizon_steps(cfg: Any) -> int:
    """
    Compute the desired number of horizon steps from the config.

    Returns:
        Number of steps per episode.
    """
    return int(cfg.episode_length_s / (cfg.sim.dt * cfg.decimation))


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
        replay_reference: bool, optional, if True, applies reference pose to simulator
        reference_joint_names: list[str], optional, custom joint order for reference
        debug_timing: bool, optional, if True, enables step timing profiling

    Example config:
        dataset_path = '/path/to/zarr'
        window_size = 5
        batch_size = 2
        device = 'cuda'
        loader_type = 'loco_mujoco'
        loader_kwargs = {'env_name': 'UnitreeG1', 'task': 'walk'}
        replay_reference = True
        reference_joint_names = ['root_z', 'root_qx', ...]
        debug_timing = True  # Enable step timing profiling

    Example usage:
        env = ImitationRLEnv(cfg)
        obs = env.reset()

        # Optional: Enable timing debug if not set in config
        env.enable_timing_debug(True)

        for _ in range(100):
            action = ...
            obs, reward, done, info = env.step(action)

        # Print timing summary manually
        env.print_timing_summary()
    """

    def __init__(
        self, cfg: Any, render_mode: Optional[str] = None, **kwargs: Any
    ) -> None:
        """Initialize the ImitationRLEnv with proper attribute typing and structure."""
        init_start_time = time.perf_counter()
        init_timings: dict[str, float] = {}

        print(
            f"[INIT] Starting ImitationRLEnv initialization with num_envs={cfg.scene.num_envs}"
        )

        # === Parent Class Initialization ===
        t1 = time.perf_counter()
        super().__init__(cfg, render_mode, **kwargs)
        init_timings["parent_init"] = time.perf_counter() - t1
        print(
            f"[INIT] Parent class initialization took {init_timings['parent_init']:.3f}s"
        )

        # === Core Configuration ===
        t1 = time.perf_counter()
        self.dataset_path: str = self._validate_dataset_path(cfg)
        # self.device: torch.device = torch.device(getattr(cfg, "device", self.device))
        self.window_size: int = getattr(cfg, "window_size", 64)
        self.batch_size: int = getattr(cfg, "batch_size", 1)
        self.replay_reference: bool = getattr(cfg, "replay_reference", False)
        init_timings["config_setup"] = time.perf_counter() - t1

        # === Dataset and Cache ===
        t1 = time.perf_counter()
        self.dataset: ZarrBackedTrajectoryDataset = self._initialize_dataset(cfg)
        init_timings["dataset_init"] = time.perf_counter() - t1
        print(f"[INIT] Dataset initialization took {init_timings['dataset_init']:.3f}s")

        t1 = time.perf_counter()
        self.cache: ZarrTrajectoryWindowCache = ZarrTrajectoryWindowCache(
            self.dataset,
            window_size=self.window_size,
            device=self.device,
            max_envs=self.num_envs,
        )
        init_timings["cache_init"] = time.perf_counter() - t1
        print(f"[INIT] Cache initialization took {init_timings['cache_init']:.3f}s")

        # === Trajectory Management ===
        t1 = time.perf_counter()
        self.traj_lengths: torch.Tensor = torch.empty(
            0, device=self.device, dtype=torch.long
        )
        self.num_trajectories: int = 0
        self.traj_num_windows: torch.Tensor = torch.empty(
            0, device=self.device, dtype=torch.long
        )
        self.traj_window_offsets: torch.Tensor = torch.empty(
            0, device=self.device, dtype=torch.long
        )
        self.total_windows: int = 0
        self.env2traj: torch.Tensor = torch.empty(
            0, device=self.device, dtype=torch.long
        )
        self.env2step: torch.Tensor = torch.empty(
            0, device=self.device, dtype=torch.long
        )

        # === Reference Pose Tracking ===
        self.ref_qpos: Optional[torch.Tensor] = None
        self.ref_root_pos: Optional[torch.Tensor] = None
        self.ref_root_rot: Optional[torch.Tensor] = None

        # === Joint Mapping ===
        self._qpos_mapping: list[int] = []
        self._qpos_inv_mapping: list[int] = []
        self._root_pos_idx: list[int] = []
        self._root_rot_idx: list[int] = []
        init_timings["tensor_setup"] = time.perf_counter() - t1

        # === Async Prefetching ===
        t1 = time.perf_counter()
        self._prefetch_executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=8)
        executor_time = time.perf_counter() - t1
        print(f"[INIT]   - ThreadPoolExecutor creation: {executor_time:.3f}s")

        t1 = time.perf_counter()
        self._prefetch_futures: list[Optional[Future[None]]] = [
            None for _ in range(self.num_envs)
        ]
        futures_list_time = time.perf_counter() - t1
        print(
            f"[INIT]   - Futures list creation (num_envs={self.num_envs}): {futures_list_time:.3f}s"
        )

        t1 = time.perf_counter()
        self._prefetch_lock: threading.Lock = threading.Lock()
        lock_time = time.perf_counter() - t1

        init_timings["prefetch_setup"] = executor_time + futures_list_time + lock_time
        print(
            f"[INIT] Prefetch setup (num_envs={self.num_envs}) took {init_timings['prefetch_setup']:.3f}s"
        )

        # === Debug Support ===
        self.debug_reference_check: bool = False
        self._debug_loader: Optional[LoaderType] = None
        self.debug_timing: bool = getattr(cfg, "debug_timing", False)
        self._timing_stats: dict[str, list[float]] = {}

        # === Initialize Components ===
        t1 = time.perf_counter()
        self._init_trajectory_attributes()
        init_timings["trajectory_init"] = time.perf_counter() - t1
        print(
            f"[INIT] Trajectory attributes initialization took {init_timings['trajectory_init']:.3f}s"
        )

        t1 = time.perf_counter()
        self._init_debug_loader(cfg)
        init_timings["debug_loader_init"] = time.perf_counter() - t1

        # === Print Initialization Summary ===
        total_init_time = time.perf_counter() - init_start_time
        init_timings["total_init"] = total_init_time
        self._print_init_summary(init_timings)

    # ================================================================
    # INITIALIZATION METHODS
    # ================================================================

    def _validate_dataset_path(self, cfg: Any) -> str:
        """Validate and return the dataset path from config."""
        dataset_path = getattr(cfg, "dataset_path", None)
        if dataset_path is None:
            raise ValueError(
                "dataset_path must be provided in the config for ImitationRLEnv."
            )
        return dataset_path

    def _initialize_dataset(self, cfg: Any) -> ZarrBackedTrajectoryDataset:
        """Initialize or create the Zarr dataset."""
        if not _check_zarr_exists(self.dataset_path):
            self._create_zarr_dataset(cfg)

        return ZarrBackedTrajectoryDataset(
            self.dataset_path,
            window_size=self.window_size,
            device=self.device,
            batch_size=self.batch_size,
        )

    def _create_zarr_dataset(self, cfg: Any) -> None:
        """Create Zarr dataset from loader if it doesn't exist."""
        loader_type = getattr(cfg, "loader_type", None)
        loader_kwargs = getattr(cfg, "loader_kwargs", None)

        if loader_type is None or loader_kwargs is None:
            raise RuntimeError(
                "Zarr dataset not found and loader_type/loader_kwargs not provided in config. "
                "Please specify how to generate the dataset."
            )

        loader = _get_loader(loader_type, loader_kwargs)
        export_trajectories_to_zarr(
            loader,
            self.dataset_path,
            window_size=self.window_size,
            control_freq=_get_control_freq(self.cfg),
            desired_horizon_steps=_get_desired_horizon_steps(self.cfg),
            horizon_multiplier=2.0,
        )

    def _init_trajectory_attributes(self) -> None:
        """Initialize trajectory-related attributes including lengths, window counts, and assignments."""
        print(
            f"[INIT] Initializing trajectory attributes for {self.num_envs} environments..."
        )

        t1 = time.perf_counter()
        self.traj_lengths = torch.tensor(
            self.dataset.lengths, device=self.device, dtype=torch.long
        )
        self.num_trajectories = len(self.traj_lengths)
        print(
            f"[INIT]   - Trajectory lengths setup: {time.perf_counter() - t1:.3f}s (num_trajectories={self.num_trajectories})"
        )

        # Compute number of windows per trajectory (at least 1 per trajectory)
        t1 = time.perf_counter()
        self.traj_num_windows = torch.tensor(
            [
                max(1, int(length) - self.window_size + 1)
                for length in self.traj_lengths
            ],
            device=self.device,
            dtype=torch.long,
        )
        print(f"[INIT]   - Window count computation: {time.perf_counter() - t1:.3f}s")

        # Compute window offsets for global indexing
        t1 = time.perf_counter()
        self.traj_window_offsets = torch.cat(
            [
                torch.tensor([0], device=self.device, dtype=torch.long),
                torch.cumsum(self.traj_num_windows, dim=0)[:-1],
            ]
        )
        self.total_windows = int(self.traj_num_windows.sum().item())
        print(
            f"[INIT]   - Window offset computation: {time.perf_counter() - t1:.3f}s (total_windows={self.total_windows})"
        )

        # Assign each env a trajectory and a step within that trajectory
        t1 = time.perf_counter()
        self.env2traj = torch.empty(self.num_envs, dtype=torch.long, device=self.device)
        self.env2step = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        tensor_creation_time = time.perf_counter() - t1
        print(
            f"[INIT]   - Tensor creation (num_envs={self.num_envs}): {tensor_creation_time:.3f}s"
        )

        t1 = time.perf_counter()
        env_indices = torch.arange(self.num_envs, device=self.device)
        index_creation_time = time.perf_counter() - t1
        print(f"[INIT]   - Environment indices creation: {index_creation_time:.3f}s")

        t1 = time.perf_counter()
        self.assign_trajectories(env_indices)
        assignment_time = time.perf_counter() - t1
        print(f"[INIT]   - Trajectory assignment: {assignment_time:.3f}s")

    def _init_debug_loader(self, cfg: Any) -> None:
        """Optionally initialize a debug loader for reference-vs-loader checks."""
        if self.debug_reference_check:
            loader_type = getattr(cfg, "loader_type", None)
            loader_kwargs = getattr(cfg, "loader_kwargs", None)
            if loader_type and loader_kwargs:
                self._debug_loader = _get_loader(loader_type, loader_kwargs)

    # ================================================================
    # TRAJECTORY MANAGEMENT
    # ================================================================

    def assign_trajectories(self, env_ids: Sequence[int] | torch.Tensor) -> None:
        """
        Assign a random trajectory to each environment in env_ids.
        You can override this for round-robin, curriculum, or fixed assignment.

        Args:
            env_ids: Indices of environments to assign.

        Example:
            self.assign_trajectories(torch.arange(self.num_envs, device=self.device))
        """
        if hasattr(self, "debug_timing") and getattr(self, "debug_timing", False):
            print(
                f"[INIT]     - Assigning trajectories to {len(env_ids)} environments..."
            )

        t1 = time.perf_counter()
        self.env2traj[env_ids] = torch.randint(
            0, self.num_trajectories, (len(env_ids),), device=self.device
        )
        assignment_duration = time.perf_counter() - t1

        if hasattr(self, "debug_timing") and getattr(self, "debug_timing", False):
            print(f"[INIT]     - torch.randint call took: {assignment_duration:.3f}s")

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor) -> None:
        """
        Reset the specified environments, assign new trajectories, and prefetch the first window.

        Args:
            env_ids: Indices of environments to reset.
        """
        if self.debug_timing and len(env_ids) > 100:
            reset_start = time.perf_counter()
            print(f"[RESET] Resetting {len(env_ids)} environments...")

        # Call parent reset
        if self.debug_timing and len(env_ids) > 100:
            t1 = time.perf_counter()
        super()._reset_idx(env_ids)  # type: ignore
        if self.debug_timing and len(env_ids) > 100:
            parent_reset_time = time.perf_counter() - t1
            print(f"[RESET]   - Parent reset took: {parent_reset_time:.3f}s")

        # Assign new trajectory to each reset env
        if self.debug_timing and len(env_ids) > 100:
            t1 = time.perf_counter()
        self.assign_trajectories(env_ids)
        self.env2step[env_ids] = 0
        if self.debug_timing and len(env_ids) > 100:
            assign_time = time.perf_counter() - t1
            print(f"[RESET]   - Trajectory assignment took: {assign_time:.3f}s")

        # Cancel any outstanding prefetches for these envs
        if self.debug_timing and len(env_ids) > 100:
            t1 = time.perf_counter()
        self._cancel_prefetches(env_ids)
        if self.debug_timing and len(env_ids) > 100:
            cancel_time = time.perf_counter() - t1
            print(f"[RESET]   - Cancel prefetches took: {cancel_time:.3f}s")

        # Optionally prefetch the first window for these envs
        if self.debug_timing and len(env_ids) > 100:
            t1 = time.perf_counter()
        for env in env_ids:
            self._maybe_prefetch(env.item(), self.env2traj[env].item(), 0)  # type: ignore
        if self.debug_timing and len(env_ids) > 100:
            prefetch_time = time.perf_counter() - t1
            total_reset_time = time.perf_counter() - reset_start
            print(f"[RESET]   - Initial prefetches took: {prefetch_time:.3f}s")
            print(f"[RESET] Total reset time: {total_reset_time:.3f}s")

    def _get_window_index(self, traj_idx: int, step_idx: int) -> int:
        """
        Map (traj_idx, step_idx) to global window index in the Zarr dataset.

        Args:
            traj_idx: Trajectory index.
            step_idx: Step index within trajectory.

        Returns:
            Global window index.
        """
        return int(self.traj_window_offsets[traj_idx] + step_idx)

    # ================================================================
    # ASYNC PREFETCHING
    # ================================================================

    def _cancel_prefetches(self, env_ids: Sequence[int] | torch.Tensor) -> None:
        """
        Cancel any outstanding prefetches for the given environments.

        Args:
            env_ids: Indices of environments.
        """
        if self.debug_timing and len(env_ids) > 1000:
            cancel_start = time.perf_counter()
            print(
                f"[PREFETCH] Cancelling prefetches for {len(env_ids)} environments..."
            )

        with self._prefetch_lock:
            for env in env_ids:
                fut = self._prefetch_futures[env.item()]  # type: ignore
                if fut is not None and not fut.done():
                    fut.cancel()
                self._prefetch_futures[env.item()] = None  # type: ignore

        if self.debug_timing and len(env_ids) > 1000:
            cancel_time = time.perf_counter() - cancel_start
            print(f"[PREFETCH] Cancel prefetches completed in: {cancel_time:.3f}s")

    def _maybe_prefetch(self, env_idx: int, traj_idx: int, step_idx: int) -> None:
        """
        If not already prefetched, prefetch the window for (traj_idx, step_idx) in a background thread.

        Args:
            env_idx: Environment index.
            traj_idx: Trajectory index.
            step_idx: Step index within trajectory.
        """

        def prefetch() -> None:
            # This will populate the cache for the next window
            self.cache.dataset[self._get_window_index(traj_idx, step_idx)]

        with self._prefetch_lock:
            fut = self._prefetch_futures[env_idx]
            if fut is not None and not fut.done():
                return  # Already prefetching
            self._prefetch_futures[env_idx] = self._prefetch_executor.submit(prefetch)

    # ================================================================
    # MAIN ENVIRONMENT STEP
    # ================================================================

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """
        Step the environment with the given action, update trajectory window, and optionally replay reference pose.

        Args:
            action: Action tensor for all environments.

        Returns:
            Standard vectorized env step return.
        """
        step_start_time = time.perf_counter()
        timings: dict[str, float] = {}

        # Advance time for each env
        t1 = time.perf_counter()
        self.env2step += 1
        timings["env_step_increment"] = time.perf_counter() - t1

        # Clamp to valid range for each env (0 to num_windows-1 for that trajectory)
        t1 = time.perf_counter()
        max_step = self.traj_num_windows[self.env2traj] - 1
        self.env2step = torch.minimum(self.env2step, max_step)
        timings["step_clamping"] = time.perf_counter() - t1

        # Async prefetch: if next step will move out of window, prefetch next window
        t1 = time.perf_counter()
        self._handle_async_prefetch()
        timings["async_prefetch"] = time.perf_counter() - t1

        # Compute and store the current reference qpos for all envs
        t1 = time.perf_counter()
        self.ref_qpos = self.compute_reference(key="qpos")
        timings["compute_reference"] = time.perf_counter() - t1

        # Also extract and store reference root position and orientation for all envs
        t1 = time.perf_counter()
        self._extract_and_store_root_pose()
        timings["extract_root_pose"] = time.perf_counter() - t1

        # Replay reference if enabled
        t1 = time.perf_counter()
        if self.replay_reference:
            self._apply_reference_replay()
        timings["reference_replay"] = time.perf_counter() - t1

        # Call parent step
        t1 = time.perf_counter()
        result = super().step(action)
        timings["parent_step"] = time.perf_counter() - t1

        # Log timing information
        timings["total_step_time"] = time.perf_counter() - step_start_time

        if self.debug_timing:
            self._log_step_timing(timings)

        return result

    def _handle_async_prefetch(self) -> None:
        """Handle async prefetching for the next step."""
        next_step = self.env2step + 1
        need_prefetch = next_step < self.traj_num_windows[self.env2traj]
        envs_to_prefetch = torch.where(need_prefetch)[0]

        for i in envs_to_prefetch:
            self._maybe_prefetch(i.item(), self.env2traj[i].item(), next_step[i].item())  # type: ignore

    def _apply_reference_replay(self) -> None:
        """Apply reference pose to the simulator if replay_reference is enabled."""
        if self.debug_timing:
            replay_start = time.perf_counter()

        isaaclab_joint_names = self.scene["robot"].joint_names
        reference_joint_names = self.get_reference_joint_names()

        if not self._qpos_mapping:
            if self.debug_timing:
                t1 = time.perf_counter()
            self.compute_qpos_mapping(reference_joint_names)
            if self.debug_timing:
                mapping_time = time.perf_counter() - t1
                self._timing_stats.setdefault("replay_compute_mapping", []).append(
                    mapping_time
                )

        # Apply the reference pose to the simulator (root + joints)
        if self.debug_timing:
            t1 = time.perf_counter()
        self._apply_reference_to_scene(isaaclab_joint_names, reference_joint_names)
        if self.debug_timing:
            apply_time = time.perf_counter() - t1
            self._timing_stats.setdefault("apply_reference_to_scene", []).append(
                apply_time
            )

            total_replay_time = time.perf_counter() - replay_start
            self._timing_stats.setdefault("reference_replay_total", []).append(
                total_replay_time
            )

    # ================================================================
    # REFERENCE COMPUTATION
    # ================================================================

    def compute_reference(self, key: str = "qpos") -> torch.Tensor:
        """
        Fully vectorized: get the reference state for each env at the current time index using batch_get.

        Args:
            key: Which observation key to extract (e.g., 'qpos', 'qvel', etc.).

        Returns:
            Tensor of shape (num_envs, ...)

        Example:
            ref_qpos = env.compute_reference(key='qpos')
        """
        if self.debug_timing:
            ref_start = time.perf_counter()

        # necessary to call this before computing the reference, since the asset_cfg might be reset
        if self.debug_timing:
            t1 = time.perf_counter()
        self.compute_qpos_mapping()
        if self.debug_timing:
            mapping_time = time.perf_counter() - t1
            self._timing_stats.setdefault("compute_qpos_mapping", []).append(
                mapping_time
            )

        if key == "root":
            key = "qpos"
        window_indices = self.env2step
        # Always use the first step in the window for standard imitation
        step_indices = torch.zeros_like(window_indices)

        # Use the underlying dataset's batch_get directly for (window_idx, step_in_window)
        if self.debug_timing:
            t1 = time.perf_counter()
        batch = self.cache.dataset.batch_get(
            window_indices, step_indices, key=key, data_type="observations"
        )
        if self.debug_timing:
            batch_get_time = time.perf_counter() - t1
            self._timing_stats.setdefault("dataset_batch_get", []).append(
                batch_get_time
            )

        # Defensive: ensure output is always 2D (num_envs, qpos_dim)
        if batch.ndim == 1:
            batch = batch.unsqueeze(0)
        if key == "root":
            return batch[..., :7]

        if self.debug_timing:
            total_ref_time = time.perf_counter() - ref_start
            self._timing_stats.setdefault("compute_reference_total", []).append(
                total_ref_time
            )

        return batch

    def _extract_and_store_root_pose(self) -> None:
        """Extract and store the reference root position and orientation for all environments."""
        if self.ref_qpos is None:
            return

        if self._root_pos_idx:
            self.ref_root_pos = self.ref_qpos[:, self._root_pos_idx]
        else:
            self.ref_root_pos = None

        if self._root_rot_idx:
            self.ref_root_rot = self.ref_qpos[:, self._root_rot_idx]
        else:
            self.ref_root_rot = None

    # ================================================================
    # JOINT MAPPING AND CONVERSION
    # ================================================================

    def compute_qpos_mapping(
        self, reference_joint_names: Optional[list[str]] = None
    ) -> tuple[list[int], list[int]]:
        """
        Compute mapping between IsaacLab qpos order and a reference qpos order (configurable).
        Also separates root position and orientation indices for convenience.

        Args:
            reference_joint_names: List of joint names in reference qpos order.
                                 If None, uses cfg.reference_joint_names or self.get_loco_joint_names().

        Returns:
            Tuple of (mapping, inv_mapping) where:
            - mapping: Indices such that qpos_reference = qpos_isaaclab[mapping]
            - inv_mapping: Indices such that qpos_isaaclab = qpos_reference[inv_mapping]

        Example:
            mapping, inv_mapping = env.compute_qpos_mapping()
        """
        isaaclab_joint_names = self.scene["robot"].joint_names

        # Allow joint name list to be set via config
        if reference_joint_names is None:
            reference_joint_names = getattr(self.cfg, "reference_joint_names", None)
        if reference_joint_names is None:
            reference_joint_names = self.get_loco_joint_names()

        # Build mappings
        mapping = self._build_mapping(isaaclab_joint_names, reference_joint_names)
        inv_mapping = self._build_inv_mapping(
            isaaclab_joint_names, reference_joint_names
        )

        self._qpos_mapping = mapping
        self._qpos_inv_mapping = inv_mapping

        # Identify root position and orientation indices (assume standard naming)
        self._root_pos_idx = [
            i
            for i, name in enumerate(reference_joint_names)
            if name in ["root_x", "root_y", "root_z"]
        ]
        self._root_rot_idx = [
            i
            for i, name in enumerate(reference_joint_names)
            if name in ["root_qw", "root_qx", "root_qy", "root_qz"]
        ]

        # Optional debug info
        if getattr(self, "debug_mapping", False):
            self._log_joint_mapping_debug(
                isaaclab_joint_names, reference_joint_names, mapping, inv_mapping
            )

        return mapping, inv_mapping

    def _build_mapping(
        self, isaaclab_joint_names: list[str], reference_joint_names: list[str]
    ) -> list[int]:
        """Build mapping from reference joint order to IsaacLab joint order."""
        return [
            isaaclab_joint_names.index(name)
            for name in reference_joint_names
            if name in isaaclab_joint_names and name in reference_joint_names
        ]

    def _build_inv_mapping(
        self, isaaclab_joint_names: list[str], reference_joint_names: list[str]
    ) -> list[int]:
        """Build inverse mapping from IsaacLab joint order to reference joint order."""
        return [
            reference_joint_names.index(name)
            for name in isaaclab_joint_names
            if name in reference_joint_names and name in isaaclab_joint_names
        ]

    def get_loco_joint_names(self) -> list[str]:
        """
        Return the list of joint names in loco-mujoco qpos order for this robot.
        Override or set this method/config per robot.

        Returns:
            Joint names in loco-mujoco order. This is for the Unitree G1 robot.

        Example:
            joint_names = env.get_loco_joint_names()
        """
        # Example for UnitreeG1 (should generalize for other robots)
        return [
            "root_x",
            "root_y",
            "root_z",
            "root_qw",
            "root_qx",
            "root_qy",
            "root_qz",
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

    def get_reference_joint_names(self) -> list[str]:
        """
        Return the list of joint names in the reference qpos order for this robot.
        Override or set this method/config per robot. By default, uses cfg.reference_joint_names
        or falls back to get_loco_joint_names().

        Returns:
            Reference joint names.
        """
        joint_names = getattr(self.cfg, "reference_joint_names", None)
        if joint_names is not None:
            return joint_names
        return self.get_loco_joint_names()

    def get_qpos_in_loco_order(
        self, qpos: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Return IsaacLab qpos in loco-mujoco order using the mapping.

        Args:
            qpos: qpos array to convert. If None, uses self.qpos.

        Returns:
            qpos in loco-mujoco order.

        Example:
            qpos_loco = env.get_qpos_in_loco_order()
        """
        if qpos is None:
            qpos = getattr(self, "qpos", None)
            if qpos is None:
                raise AttributeError(
                    "self.qpos is not set. Please set self.qpos before calling get_qpos_in_loco_order, "
                    "or pass qpos explicitly."
                )

        if not self._qpos_mapping:
            self.compute_qpos_mapping()

        return qpos[..., self._qpos_mapping]

    def get_qpos_in_isaaclab_order(
        self, qpos_loco: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Convert qpos from loco-mujoco order to IsaacLab order using the inverse mapping.

        Args:
            qpos_loco: qpos array in loco-mujoco order. If None, uses self.qpos_loco.

        Returns:
            qpos in IsaacLab order.

        Example:
            qpos_isaaclab = env.get_qpos_in_isaaclab_order()
        """
        if qpos_loco is None:
            qpos_loco = getattr(self, "qpos_loco", None)
            if qpos_loco is None:
                raise AttributeError(
                    "self.qpos_loco is not set. Please set self.qpos_loco before calling "
                    "get_qpos_in_isaaclab_order, or pass qpos_loco explicitly."
                )

        if not self._qpos_inv_mapping:
            self.compute_qpos_mapping()

        return qpos_loco[..., self._qpos_inv_mapping]

    # ================================================================
    # SCENE INTERACTION
    # ================================================================

    def _apply_reference_to_scene(
        self, isaaclab_joint_names: list[str], reference_joint_names: list[str]
    ) -> None:
        """
        Helper that pushes the reference root pose and joint configuration to the simulator.

        Args:
            isaaclab_joint_names: Joint name order used by the IsaacLab robot.
            reference_joint_names: Joint name order used by the reference dataset.
        """
        if self.ref_qpos is None:
            raise RuntimeError(
                "ref_qpos has not been computed before applying reference pose. "
                "Call compute_reference() first."
            )

        # Construct and apply the root pose
        root_pose = self._construct_root_pose(self.ref_qpos, reference_joint_names)
        default_root_state = self.scene["robot"].data.default_root_state.clone()
        default_root_state[..., :7] = root_pose
        default_root_state[..., :3] += self.scene.env_origins
        self.scene["robot"].write_root_state_to_sim(default_root_state)

        # Construct and apply the joint configuration
        qpos_joints = self._construct_joint_qpos(
            self.ref_qpos, isaaclab_joint_names, reference_joint_names
        )
        self.scene["robot"].write_joint_state_to_sim(
            qpos_joints,
            self.scene["robot"].data.default_joint_vel.clone(),
        )
        self.scene.write_data_to_sim()

    def _construct_root_pose(
        self, ref_qpos: torch.Tensor, reference_joint_names: list[str]
    ) -> torch.Tensor:
        """
        Builds the 7-DoF root pose tensor in the canonical order (x, y, z, qw, qx, qy, qz).
        Any missing components are filled with sensible defaults.

        Args:
            ref_qpos: Reference qpos tensor (num_envs, ...).
            reference_joint_names: Joint names in reference order.

        Returns:
            Root pose (num_envs, 7)
        """
        canonical_root_order = [
            "root_x",
            "root_y",
            "root_z",
            "root_qw",
            "root_qx",
            "root_qy",
            "root_qz",
        ]
        root_pose_list: list[torch.Tensor] = []

        for root_name in canonical_root_order:
            if root_name in reference_joint_names:
                idx = reference_joint_names.index(root_name)
                root_pose_list.append(ref_qpos[:, idx : idx + 1])
            else:
                # Fallback defaults
                default_val = self._get_default_root_value(root_name)
                root_pose_list.append(default_val)

        return torch.cat(root_pose_list, dim=1)  # (num_envs, 7)

    def _get_default_root_value(self, root_name: str) -> torch.Tensor:
        """
        Return a sensible default value for a missing root component.

        Args:
            root_name: Name of the root component.

        Returns:
            Default value (num_envs, 1)
        """
        if root_name == "root_x":
            return (
                self.scene.env_origins[:, 0:1]
                if hasattr(self.scene, "env_origins")
                else torch.zeros((self.num_envs, 1), device=self.device)
            )
        elif root_name == "root_y":
            return (
                self.scene.env_origins[:, 1:2]
                if hasattr(self.scene, "env_origins")
                else torch.zeros((self.num_envs, 1), device=self.device)
            )
        elif root_name == "root_z":
            return (
                self.scene.env_origins[:, 2:3]
                if hasattr(self.scene, "env_origins")
                else torch.zeros((self.num_envs, 1), device=self.device)
            )
        elif root_name == "root_qw":
            return torch.ones((self.num_envs, 1), device=self.device)
        else:
            return torch.zeros((self.num_envs, 1), device=self.device)

    def _construct_joint_qpos(
        self,
        ref_qpos: torch.Tensor,
        isaaclab_joint_names: list[str],
        reference_joint_names: list[str],
    ) -> torch.Tensor:
        """
        Builds a (num_envs, num_joints) tensor containing the reference joint positions
        in IsaacLab joint order, excluding the root pose components.

        Args:
            ref_qpos: Reference qpos tensor (num_envs, ...).
            isaaclab_joint_names: IsaacLab joint order.
            reference_joint_names: Reference joint order.

        Returns:
            Joint positions in IsaacLab order (num_envs, num_joints)
        """
        # Identify joint indices in reference order that correspond to actual joints
        root_names = [
            reference_joint_names[i] for i in (self._root_pos_idx + self._root_rot_idx)
        ]
        joint_indices_ref = [
            i for i, n in enumerate(reference_joint_names) if n not in root_names
        ]
        joint_names_ref = [reference_joint_names[i] for i in joint_indices_ref]

        # Prepare joint position array (num_envs, num_joints)
        qpos_joints = torch.zeros(
            (ref_qpos.shape[0], len(isaaclab_joint_names)),
            device=ref_qpos.device,
        )

        # Transfer available joint positions from reference → IsaacLab order
        for idx_ref, name in zip(joint_indices_ref, joint_names_ref):
            if name in isaaclab_joint_names:
                idx_isaac = isaaclab_joint_names.index(name)
                qpos_joints[:, idx_isaac] = ref_qpos[:, idx_ref]

        return qpos_joints

    # ================================================================
    # DEBUG UTILITIES
    # ================================================================

    def debug_reference_vs_loader(
        self,
        loader: LoaderType,
        num_envs_to_check: int = 3,
        num_steps: int = 10,
        key: str = "qpos",
    ) -> None:
        """
        Debug utility to compare reference data fetched from the Zarr dataset (via env) and the original loader.

        Args:
            loader: The original trajectory loader (e.g., LocoMuJoCoLoader).
            num_envs_to_check: Number of environments to check.
            num_steps: Number of steps to check for each env.
            key: Which observation key to check (e.g., 'qpos').

        Example:
            env.debug_reference_vs_loader(loader, num_envs_to_check=2, num_steps=5)
        """
        print("=== Debug: Reference vs Loader ===")

        for env_id in range(min(num_envs_to_check, self.num_envs)):
            traj_idx = self.env2traj[env_id].item()
            print(f"\n[Env {env_id}] Trajectory index: {traj_idx}")

            # Get the original trajectory from the loader
            orig_traj = loader[traj_idx]  # type: ignore
            orig_qpos = orig_traj.observations[key]
            orig_dt = orig_traj.dt
            print(f"  Loader qpos shape: {orig_qpos.shape}, dt: {orig_dt}")

            # For each step, compare the reference from env and the loader
            for step in range(min(num_steps, self.traj_num_windows[traj_idx].item())):  # type: ignore
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

    def _log_joint_mapping_debug(
        self,
        isaaclab_joint_names: list[str],
        reference_joint_names: list[str],
        mapping: list[int],
        inv_mapping: list[int],
    ) -> None:
        """Pretty-print joint mapping statistics when debugging is enabled."""
        missing_in_isaac = [
            name for name in reference_joint_names if name not in isaaclab_joint_names
        ]
        missing_in_ref = [
            name for name in isaaclab_joint_names if name not in reference_joint_names
        ]

        print("[DEBUG] Joint mapping verification")
        if missing_in_isaac:
            print(f"  Joints in reference but not in IsaacLab: {missing_in_isaac}")
        if missing_in_ref:
            print(f"  Joints in IsaacLab but not in reference: {missing_in_ref}")
        print(
            f"  Reference joint order (len={len(reference_joint_names)}): {reference_joint_names}"
        )
        print(
            f"  IsaacLab joint order   (len={len(isaaclab_joint_names)}): {isaaclab_joint_names}"
        )
        print(f"  Reference → IsaacLab indices: {mapping}")
        print(f"  IsaacLab  → Reference indices: {inv_mapping}")

    def _log_step_timing(self, timings: dict[str, float]) -> None:
        """
        Log timing statistics for the step function.
        Prints timing info every N steps and accumulates statistics.

        Args:
            timings: Dictionary of timing measurements for this step.
        """
        # Accumulate timing statistics
        for key, duration in timings.items():
            if key not in self._timing_stats:
                self._timing_stats[key] = []
            self._timing_stats[key].append(duration)

        # Print timing info every 100 steps
        if len(self._timing_stats.get("total_step_time", [])) % 100 == 0:
            self.print_timing_summary()

    def print_timing_summary(self, clear_stats: bool = False) -> None:
        """
        Print a summary of timing statistics accumulated so far.

        Args:
            clear_stats: Whether to clear accumulated statistics after printing.
        """
        if not self._timing_stats:
            print("[TIMING] No timing statistics available")
            return

        print("\n" + "=" * 80)
        print(
            f"[TIMING] Step Function Performance Summary (last {len(self._timing_stats.get('total_step_time', []))} steps)"
        )
        print("=" * 80)

        # Sort by average time (descending)
        avg_times = []
        for operation, times in self._timing_stats.items():
            if times:
                avg_time = sum(times) / len(times)
                max_time = max(times)
                min_time = min(times)
                avg_times.append((operation, avg_time, max_time, min_time, len(times)))

        avg_times.sort(key=lambda x: x[1], reverse=True)

        # Print timing table
        print(
            f"{'Operation':<30} {'Avg (ms)':<10} {'Max (ms)':<10} {'Min (ms)':<10} {'Samples':<10} {'% of Total':<12}"
        )
        print("-" * 80)

        total_avg = self._timing_stats.get("total_step_time", [0])
        total_avg_time = sum(total_avg) / len(total_avg) if total_avg else 1.0

        for operation, avg_time, max_time, min_time, samples in avg_times:
            avg_ms = avg_time * 1000
            max_ms = max_time * 1000
            min_ms = min_time * 1000
            percentage = (
                (avg_time / total_avg_time) * 100
                if operation != "total_step_time"
                else 100.0
            )

            print(
                f"{operation:<30} {avg_ms:<10.3f} {max_ms:<10.3f} {min_ms:<10.3f} {samples:<10} {percentage:<12.1f}%"
            )

        print("=" * 80)

        # Identify potential bottlenecks
        bottlenecks = [
            op for op, avg_time, _, _, _ in avg_times[:3] if op != "total_step_time"
        ]
        if bottlenecks:
            print(f"[TIMING] Top bottlenecks: {', '.join(bottlenecks)}")

        total_steps = len(self._timing_stats.get("total_step_time", []))
        if total_steps > 0:
            fps = 1.0 / total_avg_time if total_avg_time > 0 else 0
            print(
                f"[TIMING] Average step time: {total_avg_time*1000:.3f}ms (~{fps:.1f} FPS)"
            )

        print("=" * 80 + "\n")

        if clear_stats:
            self._timing_stats.clear()

    def enable_timing_debug(self, enable: bool = True) -> None:
        """
        Enable or disable timing debug mode.

        Args:
            enable: Whether to enable timing debug.

        Example:
            env.enable_timing_debug(True)  # Enable timing
            # ... run some steps ...
            env.print_timing_summary()    # Print accumulated stats
            env.enable_timing_debug(False) # Disable timing
        """
        self.debug_timing = enable
        if enable:
            print(
                "[TIMING] Timing debug mode enabled. Statistics will be printed every 100 steps."
            )
        else:
            print("[TIMING] Timing debug mode disabled.")

    def reset_timing_stats(self) -> None:
        """
        Clear all accumulated timing statistics.

        Example:
            env.reset_timing_stats()
        """
        self._timing_stats.clear()
        print("[TIMING] Timing statistics cleared.")

    def _print_init_summary(self, init_timings: dict[str, float]) -> None:
        """
        Print a summary of initialization timing.

        Args:
            init_timings: Dictionary of timing measurements from initialization.
        """
        print("\n" + "=" * 80)
        print("[INIT] Initialization Timing Summary")
        print("=" * 80)

        # Sort by time (descending)
        sorted_timings = sorted(init_timings.items(), key=lambda x: x[1], reverse=True)

        print(f"{'Component':<30} {'Time (s)':<10} {'% of Total':<12}")
        print("-" * 80)

        total_time = init_timings.get("total_init", 1.0)
        for component, duration in sorted_timings:
            if component != "total_init":
                percentage = (duration / total_time) * 100
                print(f"{component:<30} {duration:<10.3f} {percentage:<12.1f}%")

        print("-" * 80)
        print(f"{'TOTAL INITIALIZATION':<30} {total_time:<10.3f} {'100.0%':<12}")
        print("=" * 80)

        # Identify potential bottlenecks
        bottlenecks = [
            comp for comp, time_val in sorted_timings[:3] if comp != "total_init"
        ]
        if bottlenecks:
            print(f"[INIT] Primary bottlenecks: {', '.join(bottlenecks)}")

        if total_time > 10.0:
            print(
                f"[INIT] WARNING: Initialization took {total_time:.1f}s - consider optimizing the slowest components"
            )
        elif total_time > 5.0:
            print(f"[INIT] NOTE: Initialization took {total_time:.1f}s")
        else:
            print(f"[INIT] Initialization completed successfully in {total_time:.1f}s")

        print("=" * 80 + "\n")
