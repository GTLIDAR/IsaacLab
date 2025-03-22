"""Wrapper to configure a :class:`ManagerBasedRLEnv` or :class:`DirectRLEnv` instance to torchrl compatible vectorized environment."""

# needed to import for allowing type-hinting: torch.Tensor | dict[str, torch.Tensor]
from __future__ import annotations

import gymnasium as gym
import torch
import numpy as np
from typing import Any, Union, Dict, Tuple, Optional, List, Callable


from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
from isaaclab.envs.common import VecEnvStepReturn

from torchrl.envs.libs.gym import default_info_dict_reader
from tensordict import NonTensorData, TensorDict, TensorDictBase


class VecIsaacLabWrapper(gym.vector.VectorEnv):
    """Wraps around Isaac Lab environment for TorchRL.

    Isaac Sim internally implements a vectorized environment. However, the interface
    is not compatible with the TorchRL library completely. This wrapper converts the Isaac Lab
    environment to a TorchRL compatible environment.


    We also add monitoring functionality that computes the un-discounted episode
    return and length. This information is added to the info dicts under key `episode`.

    In contrast to the Isaac Lab environment, Gymnasium expects the following:

    1. numpy datatype for MDP signals
    2. a list of info dicts for each sub-environment (instead of a dict)
    3. when environment has terminated, the observations from the environment should correspond
       to the one after reset. The "real" final observation is passed using the info dicts
       under the key ``terminal_observation``.

    .. warning::

        By the nature of physics stepping in Isaac Sim, it is not possible to forward the
        simulation buffers without performing a physics step. Thus, reset is performed
        inside the :meth:`step()` function after the actual physics step is taken.
        Thus, the returned observations for terminated environments is the one after the reset.

    .. caution::

        This class must be the last wrapper in the wrapper chain. This is because the wrapper does not follow
        the :class:`gym.Wrapper` interface. Any subsequent wrappers will need to be modified to work with this
        wrapper.

    """

    def __init__(self, env: Union[ManagerBasedRLEnv, DirectRLEnv]):
        """Initialize the wrapper.

        Args:
            env: The environment to wrap around.

        Raises:
            ValueError: When the environment is not an instance of :class:`ManagerBasedRLEnv` or :class:`DirectRLEnv`.
        """
        # check that input is valid
        if not isinstance(env.unwrapped, ManagerBasedRLEnv) and not isinstance(
            env.unwrapped, DirectRLEnv
        ):
            raise ValueError(
                "The environment must be inherited from ManagerBasedRLEnv or DirectRLEnv. Environment type:"
                f" {type(env)}"
            )
        # initialize the wrapper
        self.env: Union[ManagerBasedRLEnv, DirectRLEnv] = env
        self.num_envs = self.env.get_wrapper_attr("num_envs")
        # collect common information
        self.sim_device = self.env.get_wrapper_attr("device")
        self.render_mode = self.env.get_wrapper_attr("render_mode")

        # obtain gym spaces
        action_dim = sum(self.env.get_wrapper_attr("action_manager").action_term_dim)
        self.single_action_space = gym.spaces.Box(low=-1, high=1, shape=(action_dim,))

        # initialize VectorEnv parent class
        super().__init__(
            num_envs=self.num_envs,  # type: ignore
            observation_space=self.env.get_wrapper_attr("single_observation_space"),
            action_space=self.single_action_space,
        )

    def __str__(self):
        """Returns the wrapper name and the :attr:`env` representation string."""
        return f"<{type(self).__name__}{self.env}>"

    def __repr__(self):
        """Returns the string representation of the wrapper."""
        return str(self)

    @property
    def unwrapped(self) -> gym.Env:
        """Returns the base environment of the wrapper.

        This will be the bare :class:`gymnasium.Env` environment, underneath all layers of wrappers.
        """
        return self.env.unwrapped

    def seed(self, seed: int = -1) -> list[int]:  # noqa: D102
        return [self.env.unwrapped.seed(seed)] * self.num_envs  # type: ignore

    def reset_wait(
        self,
        seed: Optional[Union[int, List[int]]] = None,
        options: Optional[dict] = None,
    ):
        """Resets the environment and waits for the simulation to be ready.

        Args:
            seed: The seed to use for the reset. If a list of seeds is provided, each environment will be reset
                with the corresponding seed.
            options: Additional options to pass to the reset function.

        Returns:
            The initial observations of the environment.
        """
        obs_dict, extras = self.env.reset()
        self.extras = extras

        return obs_dict, {"final_info": obs_dict}

    def step_async(self, actions: torch.tensor):
        """Asynchronously steps the environment."""
        # actions = actions.to(device=self.sim_device, dtype=torch.float32)
        # check if action has nan
        if actions.isnan().any():
            raise ValueError("Actions contain NaN values.")
        # convert to tensor
        self._async_actions = actions

    def step_wait(self, **kwargs) -> VecEnvStepReturn:  # noqa: D102
        # record step information
        obs_dict, rew, terminated, truncated, extras = self.env.step(
            self._async_actions
        )
        self.extras = extras
        # check if obs_dict has nan in any of the values
        if any(torch.isnan(v).any() for v in obs_dict.values()):
            print("obs has nan. checking its nan index and the done terms")
            for k, v in obs_dict.items():
                v: torch.Tensor
                index = torch.nonzero(torch.isnan(v).sum(-1), as_tuple=False)
                print(
                    f"{k} index: {index}, done terms: {terminated[index]}, {truncated[index]}"
                )
            raise ValueError("Observations contain NaN values.")
        # check if rew has nan
        if torch.isnan(rew).any():
            raise ValueError("Rewards contain NaN values.")
        return (
            obs_dict,
            rew.unsqueeze(-1).clamp(min=-100, max=100),
            terminated,
            truncated,
            {"final_info": obs_dict},
        )
