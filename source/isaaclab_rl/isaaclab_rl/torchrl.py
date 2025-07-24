"""This is copied from torchrl's IsaacLabWrapper https://github.com/pytorch/rl/blob/main/torchrl/envs/libs/isaac_lab.py."""

from __future__ import annotations

from typing import Any
from dataclasses import MISSING
from collections import deque
import gymnasium as gym
import numpy as np
import torch
from torchrl.envs.libs.gym import GymWrapper, GymLikeEnv
from torchrl.envs import Transform
from torchrl.envs.libs.gym import terminal_obs_reader
from tensordict import TensorDictBase, TensorDict

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils import configclass


class IsaacLabWrapper(GymWrapper):
    """A wrapper for IsaacLab environments.

    Args:
        env (scripts_isaaclab.envs.ManagerBasedRLEnv or equivalent): the environment instance to wrap.
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
        self.log_infos = deque(maxlen=100)

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

    @property
    def _is_batched(self) -> bool:
        return True

    def _output_transform(self, step_outputs_tuple):  # type: ignore
        # IsaacLab will modify the `terminated` and `truncated` tensors
        #  in-place. We clone them here to make sure data doesn't inadvertently get modified.
        # The variable naming follows torchrl's convention here.
        observations, reward, terminated, truncated, info = step_outputs_tuple
        for k, v in observations.items():
            if torch.isnan(v).any():
                # print the first row with nan
                print(
                    f"NaN values found in observation {k} during step. First row: {v[0]}"
                )
                raise ValueError(
                    f"NaN values found in observation {k} during step. "
                    "This is likely due to an error in the environment or the model."
                )
        if torch.isnan(reward).any():
            raise ValueError(
                "NaN values found in reward during step. "
                "This is likely due to an error in the environment or the model."
            )

        done = terminated | truncated
        reward = reward.clone().unsqueeze(-1)  # to get to (num_envs, 1)

        # warn if reward is less than -100
        if reward.min() < -100:
            print(
                f"Warning: Reward is less than -100: {reward.min()}. This is likely due to an error in the environment or the model."
            )

        self.log_infos.append(info["log"])

        observations = CloneObsBuf(observations)

        if "final_obs_buf" in info:
            info = {"final_obs_buf": CloneObsBuf(info["final_obs_buf"])}
            return (
                observations,
                reward,
                terminated.clone(),
                truncated.clone(),
                done.clone(),
                info,
            )
        else:
            return (
                observations,
                reward,
                terminated.clone(),
                truncated.clone(),
                done.clone(),
                {},
            )

    def _reset_output_transform(self, reset_data):
        """Transform the output of the reset method."""
        observations, info = reset_data
        return (observations, {})


def CloneObsBuf(
    obs_buf: dict[str, torch.Tensor | dict],
) -> dict[str, torch.Tensor | dict]:
    """Clone the observation buffer.

    Args:
        obs_buf: Dictionary that can contain tensors or nested dictionaries of tensors.

    Returns:
        Cloned dictionary with the same structure as obs_buf.
    """
    cloned = {}
    for k, v in obs_buf.items():
        if isinstance(v, dict):
            # Recursively clone nested dictionaries
            cloned[k] = CloneObsBuf(v)
        elif isinstance(v, torch.Tensor):
            # Clone tensors
            cloned[k] = v.clone()
        else:
            # For other types, just copy the reference
            cloned[k] = v
    return cloned


class IsaacLabTerminalObsReader(terminal_obs_reader):
    """A terminal observation reader for IsaacLab environments.

    This reader extracts the terminal observation from the environment's info dictionary.
    It is used to read the terminal observation when the environment is reset."""

    def __call__(self, info_dict, tensordict):
        """Read the terminal observation from the info dictionary and update the tensordict.

        Args:
            info_dict (dict): The info dictionary from the environment.
            tensordict (TensorDictBase): The tensordict to update with the terminal observation.
        Returns:
            TensorDictBase: The updated tensordict with the terminal observation.
        """
        # convert info_dict to a tensordict
        info_dict = TensorDict(info_dict)
        # get the terminal observation
        terminal_obs = info_dict.pop("final_obs_buf", None)

        # get the terminal info dict
        terminal_info = info_dict.pop(self.backend_info_key[self.backend], None)

        if terminal_info is None:
            terminal_info = {}

        super().__call__(info_dict, tensordict)
        if not self._final_validated:
            self.info_spec[self.name] = self._obs_spec.update(self.info_spec)
            self._final_validated = True
        final_info = terminal_info.copy()
        if terminal_obs is not None:
            final_info["observation"] = terminal_obs

        for key in self.info_spec[self.name].keys():
            tensordict.set(
                (self.name, key),
                terminal_obs[key]
                if terminal_obs is not None
                else self.info_spec[self.name, key].zero(),
            )
        return tensordict


@configclass
class RLOptPPOConfig:
    """Main configuration class for RLOpt PPO."""

    @configclass
    class EnvConfig:
        """Environment configuration for RLOpt PPO."""

        env_name: Any = MISSING
        """Name of the environment."""

        device: str = "cuda:0"
        """Device to run the environment on."""

        num_envs: Any = MISSING
        """Number of environments to simulate."""

    @configclass
    class CollectorConfig:
        """Data collector configuration for RLOpt PPO."""

        num_collectors: int = 1
        """Number of data collectors."""

        frames_per_batch: int = 4096 * 24
        """Number of frames per batch."""

        total_frames: int = 100_000_000
        """Total number of frames to collect."""

        set_truncated: bool = False
        """Whether to set truncated to True when the episode is done."""

    @configclass
    class LoggerConfig:
        """Logger configuration for RLOpt PPO."""

        backend: str = "wandb"
        """Logger backend to use."""

        project_name: str = "IsaacLab"
        """Project name for logging."""

        group_name: str | None = None
        """Group name for logging."""

        exp_name: Any = MISSING
        """Experiment name for logging."""

        test_interval: int = 1_000_000
        """Interval between test evaluations."""

        num_test_episodes: int = 5
        """Number of test episodes to run."""

        video: bool = False
        """Whether to record videos."""

    @configclass
    class OptimConfig:
        """Optimizer configuration for RLOpt PPO."""

        lr: float = 3e-4
        """Learning rate."""

        weight_decay: float = 0.0
        """Weight decay for optimizer."""

        anneal_lr: bool = True
        """Whether to anneal learning rate."""

        device: str = "cuda:0"
        """Device for optimizer."""

    @configclass
    class LossConfig:
        """Loss function configuration for RLOpt PPO."""

        gamma: float = 0.99
        """Discount factor."""

        mini_batch_size: Any = MISSING
        """Mini-batch size for training."""

        epochs: int = 4
        """Number of training epochs."""

        gae_lambda: float = 0.95
        """GAE lambda parameter."""

        clip_epsilon: float = 0.2
        """Clipping epsilon for PPO."""

        clip_value: bool = False
        """Whether to clip value function."""

        anneal_clip_epsilon: bool = False
        """Whether to anneal clip epsilon."""

        critic_coeff: float = 1.0
        """Critic coefficient."""

        entropy_coeff: float = 0.01
        """Entropy coefficient."""

        loss_critic_type: str = "l2"
        """Type of critic loss."""

    @configclass
    class CompileConfig:
        """Compilation configuration for RLOpt PPO."""

        compile: bool = False
        """Whether to compile the model."""

        compile_mode: str = "default"
        """Compilation mode."""

        cudagraphs: bool = False
        """Whether to use CUDA graphs."""

    @configclass
    class PolicyConfig:
        """Policy network configuration for RLOpt PPO."""

        num_cells: list[int] = [512, 256, 128]
        """Number of cells in each layer."""

    @configclass
    class ValueNetConfig:
        """Value network configuration for RLOpt PPO."""

        num_cells: list[int] = [512, 256, 128]
        """Number of cells in each layer."""

    @configclass
    class FeatureExtractorConfig:
        """Feature extractor configuration for RLOpt PPO."""

        num_cells: list[int] = [512, 256, 128]
        """Number of cells in each layer."""

        output_dim: int = 128
        """Output dimension of the feature extractor."""

    @configclass
    class TrainerConfig:
        """Trainer configuration for RLOpt PPO."""

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

    env: EnvConfig = EnvConfig()
    """Environment configuration."""

    collector: CollectorConfig = CollectorConfig()
    """Data collector configuration."""

    logger: LoggerConfig = LoggerConfig()
    """Logger configuration."""

    optim: OptimConfig = OptimConfig()
    """Optimizer configuration."""

    loss: LossConfig = LossConfig()
    """Loss function configuration."""

    compile: CompileConfig = CompileConfig()
    """Compilation configuration."""

    policy: PolicyConfig = PolicyConfig()
    """Policy network configuration."""

    value_net: ValueNetConfig = ValueNetConfig()
    """Value network configuration."""

    feature_extractor: FeatureExtractorConfig = FeatureExtractorConfig()
    """Feature extractor configuration."""

    trainer: TrainerConfig = TrainerConfig()
    """Trainer configuration."""

    use_feature_extractor: bool = True
    """Whether to use a feature extractor."""

    device: str = "cuda:0"
    """Device for training."""

    seed: int = 0
    """Random seed."""

    save_interval: int = 500
    """Interval for saving the model."""

    policy_in_keys: list[str] = ["hidden"]
    """Keys to use for the policy."""

    value_net_in_keys: list[str] = ["hidden"]
    """Keys to use for the value network."""

    total_input_keys: list[str] = ["policy"]
    """Keys to use for the total input."""
