"""This is copied from torchrl's IsaacLabWrapper https://github.com/pytorch/rl/blob/main/torchrl/envs/libs/isaac_lab.py."""

from __future__ import annotations

from typing import Any
from dataclasses import MISSING

import torch
from torchrl.envs.libs.gym import GymWrapper
from torchrl.envs import Transform
import tensordict
from tensordict import TensorDict

import isaaclab
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

    def seed(self, seed: int | None):
        self._set_seed(seed)

    def _output_transform(self, step_outputs_tuple):  # type: ignore
        # IsaacLab will modify the `terminated` and `truncated` tensors
        #  in-place. We clone them here to make sure data doesn't inadvertently get modified.
        # The variable naming follows torchrl's convention here.
        observations, reward, terminated, truncated, info = step_outputs_tuple
        done = terminated | truncated
        reward = reward.unsqueeze(-1)  # to get to (num_envs, 1)
        terminal_obs = info.pop("terminal_obs", {})
        if terminal_obs != {}:
            terminal_obs = self.read_obs(terminal_obs)
            info["terminal_obs"] = terminal_obs

        return (
            observations,
            reward,
            terminated.clone(),
            truncated.clone(),
            done.clone(),
            info,
        )


# we need to patch the terminal observation to the next observation
class PatchTerminalObs(Transform):
    def __init__(self):
        super().__init__()

    def forward(self, td: TensorDict) -> TensorDict:
        done = td.get("done")
        info = td.get("info")
        if info is not None and "terminal_obs" in info:
            # if the terminal observation is an empty dict, then we don't need to patch
            if info["terminal_obs"] == {}:
                return td
            # read the terminal observation
            term_obs = info["terminal_obs"]
            # get the next observation
            next_obs = td.get("next_observation")
            # patch the terminal observation to the next observation
            mask = done.to(torch.bool)  # type: ignore
            next_obs[mask] = term_obs[mask]  # type: ignore
            td.set("next_observation", next_obs)
        return td


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

        frames_per_batch: int = 49150
        """Number of frames per batch."""

        total_frames: int = 500_000_000
        """Total number of frames to collect."""

    @configclass
    class LoggerConfig:
        """Logger configuration for RLOpt PPO."""

        backend: str = "wandb"
        """Logger backend to use."""

        project_name: str = "torchrl_isaaclab"
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

        epochs: int = 5
        """Number of training epochs."""

        gae_lambda: float = 0.95
        """GAE lambda parameter."""

        clip_epsilon: float = 0.2
        """Clipping epsilon for PPO."""

        clip_value: bool = False
        """Whether to clip value function."""

        anneal_clip_epsilon: bool = False
        """Whether to anneal clip epsilon."""

        critic_coef: float = 1.0
        """Critic coefficient."""

        entropy_coef: float = 0.01
        """Entropy coefficient."""

        loss_critic_type: str = "l2"
        """Type of critic loss."""

    @configclass
    class CompileConfig:
        """Compilation configuration for RLOpt PPO."""

        compile: bool = True
        """Whether to compile the model."""

        compile_mode: str = "default"
        """Compilation mode."""

        cudagraphs: bool = True
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

    trainer: TrainerConfig = TrainerConfig()
    """Trainer configuration."""

    device: str = "cuda:0"
    """Device for training."""

    seed: int = 0
    """Random seed."""
