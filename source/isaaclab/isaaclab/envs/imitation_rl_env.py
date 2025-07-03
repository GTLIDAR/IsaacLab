import os
import json
from typing import Any, Optional, Union, List

import torch
from tensordict import TensorDict

# Import dataset utilities from ImitationLearningTools
from iltools_datasets import TrajectoryDatasetManager

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

    def _reset_idx(self, env_ids: List[int]):
        """Reset the specified environments."""
        # Convert to tensor if needed
        if isinstance(env_ids, torch.Tensor):
            env_ids_list = env_ids.cpu().tolist()
        else:
            env_ids_list = env_ids

        # Call parent reset
        result = super()._reset_idx(env_ids_list)

        # Reset trajectory tracking
        if isinstance(env_ids, list):
            env_ids_tensor = torch.tensor(env_ids, device=self.device)
        else:
            env_ids_tensor = env_ids

        self.trajectory_manager.reset_trajectories(env_ids_tensor)

        # Get initial reference data
        self.current_reference = self.trajectory_manager.get_reference_data()
        return result

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """Step the environment and update reference data."""
        # Get next reference data point
        self.current_reference = self.trajectory_manager.get_reference_data()

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

    def get_joint_mapping(self) -> torch.Tensor:
        """
        Get mapping from reference joint order to robot joint order.

        Returns:
            Tensor of indices mapping reference joints to robot joints.
            Shape: (num_reference_joints,)
        """
        if self._joint_mapping_cache is not None:
            return self._joint_mapping_cache

        # Get robot joint names from scene
        robot_joint_names = self.scene["robot"].data.joint_names

        # Create mapping
        mapping_indices = []
        for ref_joint in self.reference_joint_names:
            if ref_joint.startswith("root_"):
                # Skip root joints as they're handled separately
                continue

            try:
                robot_idx = robot_joint_names.index(ref_joint)
                mapping_indices.append(robot_idx)
            except ValueError:
                print(
                    f"Warning: Reference joint '{ref_joint}' not found in robot joints"
                )

        self._joint_mapping_cache = torch.tensor(mapping_indices, device=self.device)
        return self._joint_mapping_cache

    def get_mapped_joint_positions(self) -> torch.Tensor:
        """
        Get reference joint positions mapped to robot joint order.

        Returns:
            Tensor of joint positions. Shape: (num_envs, num_mapped_joints)
        """
        reference_data = self.get_reference_data()
        assert isinstance(reference_data, TensorDict)
        joint_mapping = self.get_joint_mapping()
        ref_joint_pos = reference_data.get(
            "joint_pos", torch.zeros_like(joint_mapping, device=self.device)
        )

        # Map to robot joint order
        return ref_joint_pos[:, joint_mapping]

    def get_mapped_joint_velocities(self) -> torch.Tensor:
        """
        Get reference joint velocities mapped to robot joint order.

        Returns:
            Tensor of joint velocities. Shape: (num_envs, num_mapped_joints)
        """
        reference_data = self.get_reference_data()
        assert isinstance(reference_data, TensorDict)
        joint_mapping = self.get_joint_mapping()
        ref_joint_vel = reference_data.get(
            "joint_vel", torch.zeros_like(joint_mapping, device=self.device)
        )

        # Map to robot joint order
        return ref_joint_vel[:, joint_mapping]
