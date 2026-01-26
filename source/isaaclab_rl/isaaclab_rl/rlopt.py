from __future__ import annotations

from dataclasses import field
from typing import Any, Literal

import gymnasium as gym
import torch
from rlopt.agent import IPMDRLOptConfig, PPORLOptConfig, SACRLOptConfig  # noqa: F401
from tensordict import TensorDict
from torchrl.data.tensor_specs import Composite, Unbounded
from torchrl.envs.libs.gym import GymWrapper, _gym_to_torchrl_spec_transform, terminal_obs_reader

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

    @property
    def _is_batched(self) -> bool:
        return True

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
        # IsaacLab will modify the `terminated` and `truncated` tensors
        #  in-place. We clone them here to make sure data doesn't inadvertently get modified.
        # The variable naming follows torchrl's convention here.
        observations, reward, terminated, truncated, info = step_outputs_tuple
        for k, v in observations.items():
            if torch.isnan(v).any():
                # print the first row with nan
                print(f"NaN values found in observation {k} during step. First row: {v[0]}")
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
        reward = reward.clone().unsqueeze(-1).to(dtype=torch.float32)  # to get to (num_envs, 1)

        observations = CloneObsBuf(observations)

        if "final_obs_buf" in info:
            info = {"final_obs_buf": CloneObsBuf(info["final_obs_buf"])}
            return (
                observations,
                reward,
                terminated.clone().to(dtype=torch.bool),
                truncated.clone().to(dtype=torch.bool),
                done.clone().to(dtype=torch.bool),
                info,
            )
        else:
            return (
                observations,
                reward,
                terminated.clone().to(dtype=torch.bool),
                truncated.clone().to(dtype=torch.bool),
                done.clone().to(dtype=torch.bool),
                {},
            )

    def _reset_output_transform(self, reset_data):
        """Transform the output of the reset method."""
        observations, info = reset_data
        return (CloneObsBuf(observations), {})


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
            assert v.dtype == torch.float32
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
            self.info_spec[self.name] = self._obs_spec.update(self.info_spec)  # type: ignore
            self._final_validated = True
        final_info = terminal_info.copy()  # type: ignore
        if terminal_obs is not None:
            final_info["observation"] = terminal_obs

        for key in self.info_spec[self.name].keys():  # type: ignore
            tensordict.set(
                (self.name, key),
                (terminal_obs[key] if terminal_obs is not None else self.info_spec[self.name, key].zero()),  # type: ignore
            )
        return tensordict
