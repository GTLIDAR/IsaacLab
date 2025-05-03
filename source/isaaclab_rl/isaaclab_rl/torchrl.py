"""Wrapper to configure a :class:`ManagerBasedRLEnv` or :class:`DirectRLEnv` instance to torchrl compatible vectorized environment."""

# needed to import for allowing type-hinting: torch.Tensor | dict[str, torch.Tensor]
from __future__ import annotations

import gymnasium as gym
import torch
from typing import Union, Optional, Tuple, Any, Dict


import isaacsim.core.utils.torch as torch_utils
from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
from isaaclab.envs.manager_based_rl_env_cfg import ManagerBasedRLEnvCfg
from isaaclab.envs.common import VecEnvStepReturn

from tensordict import TensorDictBase, TensorDict
from torchrl.data import Bounded, Composite, Unbounded
from torchrl.envs import EnvBase
from torchrl.envs.utils import make_composite_from_td
