import os
import torch
import threading
from concurrent.futures import ThreadPoolExecutor, Future

from .manager_based_rl_env import ManagerBasedRLEnv

# Import dataset utilities from ImitationLearningTools
from iltools_datasets import ZarrBackedTrajectoryDataset, export_trajectories_to_zarr
from iltools_datasets.utils import ZarrTrajectoryWindowCache
import logging

logging.getLogger("jax").disabled = True


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
            export_trajectories_to_zarr(loader, dataset_path, window_size=window_size)
        # Now load the Zarr dataset
        self.dataset = ZarrBackedTrajectoryDataset(
            dataset_path,
            window_size=window_size,
            device=device,
            batch_size=batch_size,
        )
        self.num_trajectories = len(self.dataset.lengths)
        self.traj_lengths = torch.tensor(
            self.dataset.lengths, device=device, dtype=torch.long
        )
        self.env2traj = torch.randint(
            0, self.num_trajectories, (self.num_envs,), device=device, dtype=torch.long
        )
        self.env2step = torch.zeros(self.num_envs, device=device, dtype=torch.long)
        # Efficient per-env window cache
        self.window_size = window_size
        self.cache = ZarrTrajectoryWindowCache(
            self.dataset, window_size=window_size, device=device
        )
        # Async prefetching: thread pool and per-env prefetch state
        self._prefetch_executor = ThreadPoolExecutor(max_workers=8)
        self._prefetch_futures = [None for _ in range(self.num_envs)]
        self._prefetch_lock = threading.Lock()
        # --- Add replay_reference option ---
        self.replay_reference = getattr(cfg, "replay_reference", False)

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        # Assign new random trajectory to each reset env
        self.env2traj[env_ids] = torch.randint(
            0, self.num_trajectories, (len(env_ids),), device=self.device
        )
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

    def step(self, actions):
        # Advance time for each env
        self.env2step += 1
        # Clamp to valid range (0 to min(traj_length-1, window_size-1))
        max_step = self.traj_lengths[self.env2traj] - 1
        max_window = torch.full_like(max_step, self.window_size - 1)
        self.env2step = torch.minimum(
            self.env2step, torch.minimum(max_step, max_window)
        )
        # Async prefetch: if next step will move out of window, prefetch next window
        next_step = self.env2step + 1
        window_start = (self.env2step // self.window_size) * self.window_size
        next_window_start = (next_step // self.window_size) * self.window_size
        need_prefetch = next_window_start != window_start
        envs_to_prefetch = torch.where(need_prefetch)[0]
        for i in envs_to_prefetch:
            self._maybe_prefetch(
                i.item(), self.env2traj[i].item(), next_window_start[i].item()
            )
        # Compute and store the current reference qpos for all envs
        self.ref_qpos = self.compute_reference(key="qpos")
        # --- Replay reference if enabled ---
        if self.replay_reference:
            isaaclab_joint_names = self.scene["robot"].joint_names
            reference_joint_names = self.get_reference_joint_names()
            # Build mapping: for each reference joint, find its index in isaaclab_joint_names
            mapping = [
                isaaclab_joint_names.index(name)
                for name in reference_joint_names
                if name in isaaclab_joint_names
            ]
            # Create a zero vector for all joints
            qpos_full = torch.zeros(
                (self.ref_qpos.shape[0], len(isaaclab_joint_names)),
                device=self.ref_qpos.device,
            )
            # Fill in the reference values at the mapped indices
            qpos_full[:, mapping] = self.get_qpos_in_isaaclab_order(self.ref_qpos)
            # Set the robot's joint positions to the full vector
            self.scene["robot"].set_joint_position_target(qpos_full)
            self.scene["robot"].write_data_to_sim()
        # Call parent step
        return super().step(actions)

    def _maybe_prefetch(self, env_idx, traj_idx, window_start):
        """
        If not already prefetched, prefetch the window for (traj_idx, window_start) in a background thread.
        """

        def prefetch():
            # This will populate the cache for the next window
            self.cache.dataset.get_window(traj_idx, window_start, self.window_size)

        with self._prefetch_lock:
            fut = self._prefetch_futures[env_idx]
            if fut is not None and not fut.done():
                return  # Already prefetching
            self._prefetch_futures[env_idx] = self._prefetch_executor.submit(prefetch)

    def compute_reference(self, key="qpos"):
        """
        Fully vectorized: get the reference state for each env at the current time index using batch_get.
        Args:
            key: Which observation key to extract (e.g., 'qpos').
        Returns:
            refs: torch.Tensor of shape (num_envs, ...)
        """
        # Use batch_get for maximum efficiency (no Python loop)
        return self.cache.batch_get(
            self.env2traj, self.env2step, key=key, data_type="observations"
        )

    def compute_qpos_mapping(self, reference_joint_names=None):
        """
        Compute mapping between IsaacLab qpos order and a reference qpos order (configurable).
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
        return mapping, inv_mapping

    def get_loco_joint_names(self):
        """
        Return the list of joint names in loco-mujoco qpos order for this robot.
        Override or set this method/config per robot.
        """
        # Example for UnitreeG1 (should generalize for other robots)
        return [
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

    # ---
    # More advanced batching (future):
    # If your Zarr dataset supports batch access for arbitrary (traj_idx, step_idx) pairs,
    # you could implement a batch_get method in the dataset/cache, e.g.:
    #   refs = self.cache.batch_get(env_indices, traj_indices, step_indices, key=key, data_type="observations")
    # This would allow a single disk read for all required data, further reducing I/O.
    # For multi-step RL (e.g., n-step returns), you could also batch fetch [step_idx:step_idx+n] for each env.

    # --- Hooks for custom observation/reward logic ---
    # def _get_observations(self):
    #     # Use self.compute_reference() as needed
    #     pass
    #
    # def _get_rewards(self):
    #     # Use self.compute_reference() as needed
    #     pass
