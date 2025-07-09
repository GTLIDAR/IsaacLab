from typing import Any, Optional, Union, Sequence

import torch
from tensordict import TensorDict

# Import dataset utilities from ImitationLearningTools
from iltools_datasets.manager import TrajectoryDatasetManager

from .common import VecEnvStepReturn
from .manager_based_rl_env import ManagerBasedRLEnv


class ImitationRLEnv(ManagerBasedRLEnv):
    """
    Simplified RL environment for imitation learning with clean dataset interface.

    Config attributes (cfg):
        dataset_path: str, path to Zarr dataset directory
        assignment_strategy: str, trajectory assignment strategy ("random", "sequential", "curriculum")
        window_size: int, optional, window size for sequence sampling
        loader_type: str, required if Zarr does not exist
        loader_kwargs: dict, required if Zarr does not exist
        reference_joint_names: list[str], joint names in reference data order

    Example config:
        dataset_path = '/path/to/zarr'
        assignment_strategy = 'random'  # or 'sequential', 'curriculum'
        window_size = 64
        loader_type = 'loco_mujoco'
        loader_kwargs = {'env_name': 'UnitreeG1', 'task': 'walk'}
        reference_joint_names = ['left_hip_pitch_joint', ...]
    """

    def __init__(
        self, cfg: Any, render_mode: Optional[str] = None, **kwargs: Any
    ) -> None:
        """Initialize the simplified ImitationRLEnv."""
        print(
            f"[ImitationRLEnv] Starting initialization with num_envs={cfg.scene.num_envs}"
        )

        # Initialize parent class
        super().__init__(cfg, render_mode, **kwargs)

        # Initialize the trajectory dataset manager
        self.trajectory_manager = TrajectoryDatasetManager(
            cfg, self.num_envs, self.device
        )

        # Current reference data cache
        self.current_reference: Optional[TensorDict] = None

        # Store reference joint mapping
        self.reference_joint_names = getattr(cfg, "reference_joint_names", [])
        self._joint_mapping_cache: Optional[torch.Tensor] = None

        print("[ImitationRLEnv] Initialization complete")

    def _reset_idx(self, env_ids: Sequence[int]):
        """Reset the specified environments."""

        # Call parent reset
        result = super()._reset_idx(env_ids)

        # Reset trajectory tracking
        self.trajectory_manager.reset_trajectories(
            torch.tensor(env_ids, device=self.device)
        )

        # Get initial reference data
        self.current_reference = self.trajectory_manager.get_reference_data()
        return result

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """Step the environment and update reference data."""
        # Get next reference data point
        self.current_reference = self.trajectory_manager.get_reference_data()
        assert isinstance(self.current_reference, TensorDict)
        assert self.current_reference.batch_size == self.num_envs
        assert "root_pos" in self.current_reference
        assert "root_quat" in self.current_reference
        assert "joint_pos" in self.current_reference
        assert "joint_vel" in self.current_reference
        assert self.current_reference.get("joint_pos").shape == (
            self.num_envs,
            len(self.scene["robot"].joint_names),
        )
        assert self.current_reference.get("joint_vel").shape == (
            self.num_envs,
            len(self.scene["robot"].joint_names),
        )

        print(self.current_reference)
        # Call parent step
        return super().step(action)

    def get_reference_data(
        self, key: Optional[str] = None
    ) -> Union[TensorDict, torch.Tensor]:
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

        return self.current_reference[key]
