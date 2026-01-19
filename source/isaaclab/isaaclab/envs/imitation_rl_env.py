from pathlib import Path
from typing import Any, Optional, Sequence, Union

import torch
from tensordict import TensorDict

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation

from isaaclab.envs.common import VecEnvStepReturn
from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv

# Import the new manager and utilities
try:
    from iltools.datasets.loco_mujoco.loader import LocoMuJoCoLoader
    from iltools.datasets.manager import ParallelTrajectoryManager, ResetSchedule
    from iltools.datasets.utils import make_rb_from
except ImportError as e:
    raise ImportError(
        f"Failed to import required modules from iltools_datasets: {e}. "
        "Make sure ImitationLearningTools is installed."
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

    def __init__(
        self, cfg: Any, render_mode: Optional[str] = None, **kwargs: Any
    ) -> None:
        """Initialize the simplified ImitationRLEnv."""
        print(
            f"[ImitationRLEnv] Starting initialization with num_envs={cfg.scene.num_envs}"
        )

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
                print(
                    f"[ImitationRLEnv] Zarr not found at {zarr_path}, creating with {loader_type} loader..."
                )
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
        self.current_reference: TensorDict = self.trajectory_manager.sample(
            advance=False
        )

        # Store reference joint mapping
        self.reference_joint_names = reference_joint_names
        self._joint_mapping_cache: Optional[torch.Tensor] = None
        self.replay_reference = getattr(cfg, "replay_reference", False)
        self.replay_only = getattr(cfg, "replay_only", False)
        if self.replay_only and not self.replay_reference:
            self.replay_reference = True
            print(
                "[ImitationRLEnv] replay_only enabled; forcing replay_reference=True."
            )

        # Store initial poses for replay
        self._init_root_pos = torch.zeros((num_envs, 3), device=device)
        self._init_root_quat = torch.zeros((num_envs, 4), device=device)

        # Initialize parent class
        super().__init__(cfg, render_mode, **kwargs)

        self.robot: Articulation = self.scene["robot"]
        joint_names = self.robot.joint_names
        print("[ImitationRLEnv] G1 Joint names: ", joint_names)

        print("[ImitationRLEnv] Initialization complete")

    def _reset_idx(self, env_ids: Sequence[int]):
        """Reset the specified environments."""

        if not isinstance(env_ids, torch.Tensor):
            env_ids_tensor = torch.tensor(
                env_ids, device=self.device, dtype=torch.int64
            )
        else:
            env_ids_tensor = env_ids.to(dtype=torch.int64)

        # Reset trajectory tracking (reassigns trajectories and resets steps)
        self.trajectory_manager.reset_envs(env_ids_tensor)

        # Get initial reference data for all envs (manager handles indexing)
        self.current_reference = self.trajectory_manager.sample(advance=True)

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

        # Get next reference data point (advance=True to move to next step)
        self.current_reference: TensorDict = self.trajectory_manager.sample(
            advance=True
        )

        if self.replay_only:
            self._replay_reference()

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

        data = self.current_reference[key]
        if joint_indices is not None:
            if isinstance(data, torch.Tensor):
                return data[..., joint_indices]
            else:
                # Handle TensorDict case - data should be a Tensor
                return data[..., joint_indices]  # type: ignore[return-value]
        else:
            return data  # type: ignore[return-value]

    def _replay_reference(self, env_ids: Optional[torch.Tensor] = None):
        """Replay the reference data. If env_ids is provided, only replay the reference data for the given environments.
        If env_ids is not provided, replay the reference data for all environments."""

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
        # Extract joint data from reference TensorDict
        # ref is a TensorDict, so accessing keys returns tensors
        joint_pos_raw = ref["joint_pos"]  # type: ignore[assignment]
        joint_vel_raw = ref["joint_vel"]  # type: ignore[assignment]
        joint_pos = joint_pos_raw.clone()
        joint_vel = joint_vel_raw.clone()

        # Replace NaN positions with default values
        pos_mask = torch.isnan(joint_pos)
        vel_mask = torch.isnan(joint_vel)
        joint_pos[pos_mask] = defaults_pos[pos_mask]
        joint_vel[vel_mask] = defaults_vel[vel_mask]
        self.robot.write_root_pose_to_sim(root_pose, env_ids=write_env_ids)  # type: ignore
        self.robot.write_root_velocity_to_sim(root_vel, env_ids=write_env_ids)  # type: ignore
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=write_env_ids)  # type: ignore
