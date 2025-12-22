import time
from typing import Any, Optional, Sequence, Union

import torch

from tensordict import TensorDict

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation

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

        # Initialize the trajectory dataset manager
        self.trajectory_manager = TrajectoryDatasetManager(
            cfg, cfg.scene.num_envs, cfg.sim.device
        )

        # needed to initialize the reference data
        self.trajectory_manager.reset_trajectories()

        # Current reference data cache
        self.current_reference: TensorDict = (
            self.trajectory_manager.get_reference_data()
        )

        # Store reference joint mapping
        self.reference_joint_names = getattr(cfg, "reference_joint_names", [])
        self._joint_mapping_cache: Optional[torch.Tensor] = None
        self.replay_reference = getattr(cfg, "replay_reference", False)

        # Store initial poses for replay
        self._init_root_pos = torch.zeros(
            (cfg.scene.num_envs, 3), device=cfg.sim.device
        )
        self._init_root_quat = torch.zeros(
            (cfg.scene.num_envs, 4), device=cfg.sim.device
        )

        # Initialize parent class
        super().__init__(cfg, render_mode, **kwargs)

        self.robot: Articulation = self.scene["robot"]

        print("[ImitationRLEnv] Initialization complete")

    def _reset_idx(self, env_ids: Sequence[int]):
        """Reset the specified environments."""

        if not isinstance(env_ids, torch.Tensor):
            env_ids_tensor = torch.tensor(env_ids, device=self.device)
        else:
            env_ids_tensor = env_ids

        # Reset trajectory tracking
        self.trajectory_manager.reset_trajectories(env_ids_tensor)

        # Get initial reference data
        self.current_reference = self.trajectory_manager.get_reference_data()

        # Trigger the reset events
        result = super()._reset_idx(env_ids_tensor)  # type: ignore

        # Store initial poses for replay
        self._init_root_pos[env_ids_tensor] = self.robot.data.root_state_w[
            env_ids_tensor, 0:3
        ]
        self._init_root_quat[env_ids_tensor] = self.robot.data.root_state_w[
            env_ids_tensor, 3:7
        ]

        self._replay_reference(env_ids_tensor)

        return result

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """Step the environment and update reference data."""

        # Get next reference data point
        self.current_reference = self.trajectory_manager.get_reference_data()

        self._replay_reference()

        # Call parent step
        return super().step(action)

    def get_reference_data(
        self, key: Optional[str] = None, joint_indices: Optional[Sequence[int]] = None
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

        if joint_indices is not None:
            return self.current_reference[key][..., joint_indices]
        else:
            return self.current_reference[key]

    def _replay_reference(self, env_ids: Optional[torch.Tensor] = None):
        """Replay the reference data. If env_ids is provided, only replay the reference data for the given environments.
        If env_ids is not provided, replay the reference data for all environments."""
        if not self.replay_reference:
            return

        if env_ids is None:
            init_pos = self._init_root_pos
            init_quat = self._init_root_quat
            ref = self.current_reference
            defaults_pos = self.robot.data.default_joint_pos
            defaults_vel = self.robot.data.default_joint_vel
            write_env_ids = None
        else:
            env_ids_tensor = env_ids
            init_pos = self._init_root_pos[env_ids_tensor]
            init_quat = self._init_root_quat[env_ids_tensor]
            ref = self.current_reference[env_ids_tensor]
            defaults_pos = self.robot.data.default_joint_pos[env_ids_tensor]
            defaults_vel = self.robot.data.default_joint_vel[env_ids_tensor]
            write_env_ids = env_ids_tensor

        # Rotate reference root_pos by initial orientation, then translate by initial position
        root_pos = math_utils.quat_apply(init_quat, ref["root_pos"])
        root_pos[..., :2] += init_pos[..., :2]
        root_pos[..., 2] = init_pos[..., 2]
        root_quat = math_utils.quat_mul(init_quat, ref["root_quat"])
        root_lin_vel = math_utils.quat_apply(init_quat, ref["root_lin_vel"])
        root_ang_vel = math_utils.quat_apply(init_quat, ref["root_ang_vel"])
        root_pose = torch.cat([root_pos, root_quat], dim=-1)
        root_vel = torch.cat([root_lin_vel, root_ang_vel], dim=-1)
        joint_pos = ref["joint_pos"].clone()
        joint_vel = ref["joint_vel"].clone()
        pos_mask = torch.isnan(joint_pos)
        vel_mask = torch.isnan(joint_vel)
        joint_pos[pos_mask] = defaults_pos[pos_mask]
        joint_vel[vel_mask] = defaults_vel[vel_mask]
        self.robot.write_root_pose_to_sim(root_pose, env_ids=write_env_ids)  # type: ignore
        self.robot.write_root_velocity_to_sim(root_vel, env_ids=write_env_ids)  # type: ignore
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=write_env_ids)  # type: ignore
