from __future__ import annotations

from dataclasses import field
from typing import Any, Literal

import gymnasium as gym
import torch
from torchrl.data.tensor_specs import Composite, Unbounded
from torchrl.envs.libs.gym import GymWrapper, _gym_to_torchrl_spec_transform

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils import configclass


class IsaacLabWrapper(GymWrapper):
    """A wrapper for IsaacLab environments.

    Args:
        env (isaaclab.envs.ManagerBasedRLEnv or equivalent): the environment instance to wrap.
        categorical_action_encoding (bool, optional): if ``True``, categorical
            specs will be converted to the TorchRL equivalent (:class:`torchrl.data.Categorical`),
            otherwise a one-hot encoding will be used (:class:`torchrl.data.OneHot`).
            Defaults to ``False``.
        allow_done_after_reset (bool, optional): if ``True``, it is tolerated
            for envs to be ``done`` just after :meth:`reset` is called.
            Defaults to ``False``.

    For other arguments, see the :class:`torchrl.envs.GymWrapper` documentation.

    Refer to `the Isaac Lab doc for installation instructions <https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html>`_.

    Example:
        >>> # This code block ensures that the Isaac app is started in headless mode
        >>> from scripts_isaaclab.app import AppLauncher
        >>> import argparse

        >>> parser = argparse.ArgumentParser(description="Train an RL agent with TorchRL.")
        >>> AppLauncher.add_app_launcher_args(parser)
        >>> args_cli, hydra_args = parser.parse_known_args(["--headless"])
        >>> app_launcher = AppLauncher(args_cli)

        >>> # Imports and env
        >>> import gymnasium as gym
        >>> import isaaclab_tasks  # noqa: F401
        >>> from isaaclab_tasks.manager_based.classic.ant.ant_env_cfg import AntEnvCfg
        >>> from torchrl.envs.libs.isaac_lab import IsaacLabWrapper

        >>> env = gym.make("Isaac-Ant-v0", cfg=AntEnvCfg())
        >>> env = IsaacLabWrapper(env)

    """

    def __init__(
        self,
        env: ManagerBasedRLEnv,  # noqa: F821
        *,
        categorical_action_encoding: bool = False,
        allow_done_after_reset: bool = True,
        convert_actions_to_numpy: bool = False,
        device: torch.device | None = None,
        **kwargs,
    ):
        if device is None:
            device = torch.device("cuda:0")
        super().__init__(
            env,
            device=device,
            categorical_action_encoding=categorical_action_encoding,
            allow_done_after_reset=allow_done_after_reset,
            convert_actions_to_numpy=convert_actions_to_numpy,
            **kwargs,
        )

    def seed(self, seed: int | None):
        self._set_seed(seed)

    def _build_env(
        self,
        env,
        from_pixels: bool = False,
        pixels_only: bool = False,
    ) -> gym.core.Env:  # noqa: F821
        env = super()._build_env(
            env,
            from_pixels=from_pixels,
            pixels_only=pixels_only,
        )
        env.autoreset_mode = "SameStep"
        return env

    def _make_specs(self, env: gym.Env, batch_size=None) -> None:  # noqa: F821
        # Build specs from IsaacLab's unbatched spaces to preserve observation keys.
        if batch_size is None:
            batch_size = self.batch_size
        env_unwrapped = getattr(env, "unwrapped", env)

        action_space = getattr(env_unwrapped, "single_action_space", None)
        action_needs_batch = action_space is not None
        action_space = action_space if action_space is not None else env.action_space
        action_spec = _gym_to_torchrl_spec_transform(
            action_space,
            device=self.device,
            categorical_action_encoding=self._categorical_action_encoding,
        )
        if action_needs_batch:
            action_spec = action_spec.expand(*batch_size, *action_spec.shape)  # type: ignore

        obs_space = getattr(env_unwrapped, "single_observation_space", None)
        obs_needs_batch = obs_space is not None
        obs_space = obs_space if obs_space is not None else env.observation_space
        observation_spec = _gym_to_torchrl_spec_transform(
            obs_space,
            device=self.device,
            categorical_action_encoding=self._categorical_action_encoding,
        )
        if obs_needs_batch:
            observation_spec = observation_spec.expand(*batch_size, *observation_spec.shape)  # type: ignore
        if not isinstance(observation_spec, Composite):
            if self.from_pixels:
                observation_spec = Composite(pixels=observation_spec, shape=batch_size)  # type: ignore
            else:
                observation_spec = Composite(observation=observation_spec, shape=batch_size)  # type: ignore

        reward_space = self._reward_space(env)
        if reward_space is not None:
            reward_spec = _gym_to_torchrl_spec_transform(
                reward_space,
                device=self.device,
                categorical_action_encoding=self._categorical_action_encoding,
            )
        else:
            reward_spec = Unbounded(shape=[1], device=self.device).expand(*batch_size, 1)  # type: ignore
        if reward_space is not None:
            reward_spec = reward_spec.expand(*batch_size, *reward_spec.shape)  # type: ignore

        self.done_spec = self._make_done_spec()  # type: ignore
        self.action_spec = action_spec  # type: ignore
        self.reward_spec = reward_spec  # type: ignore
        self.observation_spec = observation_spec  # type: ignore

    def _output_transform(self, step_outputs_tuple):  # type: ignore
        observations, reward, terminated, truncated, info = step_outputs_tuple
        done = terminated | truncated
        reward = reward.unsqueeze(-1)
        return (
            observations,
            reward,
            terminated.clone(),
            truncated.clone(),
            done.clone(),
            info,
        )


@configclass
class EnvConfig:
    """Environment configuration for RLOpt  ."""

    env_name: Any = "HalfCheetah-v4"
    """Name of the environment."""

    num_envs: int = 1
    """Number of environments to simulate."""

    device: str = "cpu"
    """Device to run the environment on."""

    library: str = "gymnasium"
    """Library to use for the environment."""


@configclass
class CollectorConfig:
    """Data collector configuration for RLOpt  ."""

    num_collectors: int = 1
    """Number of data collectors."""

    frames_per_batch: int = 12
    """Number of frames per batch."""

    total_frames: int = 100_000_000
    """Total number of frames to collect."""

    set_truncated: bool = False
    """Whether to set truncated to True when the episode is done."""

    init_random_frames: int = 1000
    """Number of random frames to collect."""

    scratch_dir: str | None = None
    """Directory to save scratch data."""

    shared: bool = False
    """Whether the buffer will be shared using multiprocessing or not.."""

    prefetch: int | None = None
    """Number of prefetch batches."""


@configclass
class ReplayBufferConfig:
    """Replay buffer configuration for RLOpt  ."""

    size: int = 1_000_000
    """Size of the replay buffer."""

    prb: bool = False
    """Whether to use a prioritized replay buffer."""

    scratch_dir: str | None = None
    """Directory to save scratch data."""

    prefetch: int = 3
    """Number of prefetch batches."""


@configclass
class LoggerConfig:
    """Logger configuration for RLOpt  ."""

    backend: str = "wandb"
    """Logger backend to use."""

    project_name: str = "RLOpt"
    """Project name for logging."""

    entity: str | None = None
    """W&B entity (username or team name) for logging."""

    group_name: str | None = None
    """Group name for logging."""

    exp_name: Any = "RLOpt"
    """Experiment name for logging."""

    test_interval: int = 1_000_000
    """Interval between test evaluations."""

    num_test_episodes: int = 5
    """Number of test episodes to run."""

    video: bool = False
    """Whether to record videos."""

    log_dir: str = "logs"
    """Base directory for logging. Structure: {log_dir}/{algorithm}/{env_name}/{timestamp}/
    Default creates: ./logs/SAC/Pendulum-v1/2025-10-27_19-49-59/
    """

    save_path: str = "models"
    """Path to save model checkpoints (relative to run directory)."""

    python_level: str | None = None
    """Overrides :attr:`RLOptConfig.log_level` for standard Python logging when provided."""

    log_to_console: bool = True
    """Whether to emit Python logs to the console."""

    console_use_rich: bool = True
    """Attempt to use ``rich``'s console handler when available for better readability."""

    console_format: str = "%(message)s"
    """Logging format string for console output (ignored when ``rich`` handler is used)."""

    log_to_file: bool = True
    """Whether to persist Python logs to a file."""

    file_name: str = "rlopt.log"
    """Filename (relative to ``log_dir``) for file logging."""

    file_format: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    """Logging format string for the file handler."""

    file_rotation_bytes: int = 10_000_000
    """Rotate the log file once it reaches this many bytes (<=0 disables rotation)."""

    file_backup_count: int = 5
    """Number of rotated log files to keep when rotation is enabled."""


@configclass
class OptimizerConfig:
    """Optimizer configuration for RLOpt."""

    optimizer: str = "adam"
    """Name of the optimizer to use (e.g. ``"adamw"``, ``"adam"``, ``"sgd"``)."""

    lr: float = 3e-4
    """Base learning rate."""

    weight_decay: float = 0.0
    """Weight decay applied to all parameter groups."""

    optimizer_kwargs: dict[str, Any] = field(default_factory=lambda: {"betas": (0.9, 0.999), "eps": 1e-8})
    """Extra keyword arguments forwarded to the optimizer (defaults tailored for Adam-family optimizers)."""

    scheduler: str | None = "steplr"
    """Optional learning-rate scheduler name (e.g. ``"steplr"``, ``"cosineannealinglr"``)."""

    scheduler_kwargs: dict[str, Any] = field(default_factory=lambda: {"step_size": 1_000, "gamma": 0.9})
    """Keyword arguments passed to the scheduler constructor."""

    scheduler_step: Literal["update", "epoch"] = "update"
    """Whether to step the scheduler after each optimizer update or once per epoch."""

    device: str = "cpu"
    """Device for optimizer state when applicable."""

    target_update_polyak: float = 0.995
    """Polyak averaging coefficient for target network updates."""

    max_grad_norm: float | None = 0.5
    """Maximum gradient norm for clipping; set to ``None`` to disable clipping."""


@configclass
class LossConfig:
    """Loss function configuration for RLOpt  ."""

    gamma: float = 0.99
    """Discount factor."""

    mini_batch_size: int = 256
    """Mini-batch size for training."""

    epochs: int = 4
    """Number of training epochs."""

    loss_critic_type: str = "l2"
    """Type of critic loss."""


@configclass
class CompileConfig:
    """Compilation configuration for RLOpt  ."""

    compile: bool = False
    """Whether to compile the model."""

    compile_mode: str = "default"
    """Compilation mode."""

    cudagraphs: bool = False
    """Whether to use CUDA graphs."""

    warmup: int = 1
    """Number of warmup iterations when compiling policies.
    Used by collectors that accept a warmup parameter.
    """


@configclass
class PolicyConfig:
    """Policy network configuration for RLOpt  ."""

    num_cells: list[int] = field(default_factory=lambda: [256, 256])  # Match TorchRL stable architecture
    """Number of cells in each layer."""

    default_policy_scale: float = 1.0  # Match TorchRL default
    """Default policy scale."""


@configclass
class ValueNetConfig:
    """Value network configuration for RLOpt  ."""

    num_cells: list[int] = field(default_factory=lambda: [256, 256])  # Match TorchRL stable architecture
    """Number of cells in each layer."""


@configclass
class ActionValueNetConfig:
    """Action-value (Q) network configuration for RLOpt."""

    num_cells: list[int] = field(default_factory=lambda: [256, 256])  # Match TorchRL stable architecture
    """Number of cells in each layer."""


@configclass
class FeatureExtractorConfig:
    """Feature extractor configuration for RLOpt  ."""

    num_cells: list[int] = field(default_factory=lambda: [256, 256])  # Match TorchRL stable architecture
    """Number of cells in each layer."""

    output_dim: int = 256  # Match TorchRL stable architecture
    """Output dimension of the feature extractor."""


@configclass
class NetworkConfig:
    """Network configuration for RLOpt  ."""

    num_cells: list[int] = field(default_factory=lambda: [256, 256])  # Match TorchRL stable architecture
    """Number of cells in each layer."""

    input_dim: int | None = None
    """Input dimension of the network. Defaults to lazy initialization if None."""

    output_dim: int = 256
    """Output dimension of the feature extractor."""

    input_keys: list[str] = field(default_factory=lambda: ["observation"])
    """Input keys for the network."""

    output_keys: list[str] = field(default_factory=list)
    """Output keys for the network."""

    activation_fn: str = "relu"
    """Activation function."""

    kwargs: dict[str, Any] = field(default_factory=dict)
    """Additional keyword arguments for the network."""


@configclass
class FeatureExtractorNetworkConfig(NetworkConfig):
    """Feature network config with type-discriminated config.

    One of mlp, lstm, or cnn may be set depending on `type`.
    """

    type: Literal["mlp", "lstm", "cnn"] = "mlp"
    # Feature extractor configuration for RLOpt.

    mlp: NetworkConfig | None = None

    lstm: NetworkConfig | None = None

    cnn: NetworkConfig | None = None

    output_dim: int = 256  # Match TorchRL stable architecture


@configclass
class TrainerConfig:
    """Trainer configuration for RLOpt  ."""

    optim_steps_per_batch: int = 10
    """Number of optimization steps per batch."""

    clip_grad_norm: bool = True
    """Whether to clip gradient norm."""

    clip_norm: float = 0.5
    """Gradient clipping norm."""

    progress_bar: bool = True
    """Whether to show progress bar."""

    save_trainer_interval: int = 10_000
    """Interval for saving trainer."""

    log_interval: int = 1000
    """Interval for logging."""

    save_trainer_file: str | None = None
    """File to save trainer to."""

    frame_skip: int = 1
    """Frame skip for training."""


@configclass
class RLOptConfig:
    """Main configuration class for RLOpt  ."""

    env: EnvConfig = field(default_factory=EnvConfig)
    """Environment configuration."""

    collector: CollectorConfig = field(default_factory=CollectorConfig)
    """Data collector configuration."""

    replay_buffer: ReplayBufferConfig = field(default_factory=ReplayBufferConfig)
    """Replay buffer configuration."""

    logger: LoggerConfig = field(default_factory=LoggerConfig)
    """Logger configuration."""

    optim: OptimizerConfig = field(default_factory=OptimizerConfig)
    """Optimizer configuration."""

    loss: LossConfig = field(default_factory=LossConfig)
    """Loss function configuration."""

    compile: CompileConfig = field(default_factory=CompileConfig)
    """Compilation configuration."""

    policy: NetworkConfig = field(default_factory=NetworkConfig)
    """Policy network configuration."""

    value_function: NetworkConfig | None = None
    """Value network configuration."""

    q_function: NetworkConfig | None = None
    """Action-value network configuration (used by off-policy agents such as SAC)."""

    feature_extractor: FeatureExtractorNetworkConfig | None = None
    """Feature extractor configuration."""

    trainer: TrainerConfig | None = None
    """Trainer configuration."""

    device: str = "cuda:0"
    """Device for training."""

    seed: int = 42
    """Random seed."""

    log_level: str = "warning"
    """Verbosity for internal debug logging (e.g. ``"debug"``, ``"info"``)."""

    save_interval: int = 10
    """Interval for saving the model."""
