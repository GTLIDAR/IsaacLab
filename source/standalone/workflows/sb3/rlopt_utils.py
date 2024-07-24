from typing import Any, Dict, Union

import numpy as np
import torch as th
from stable_baselines3.common.type_aliases import TensorDict


def obs_as_tensor(
    obs: Union[th.Tensor, np.ndarray, Dict[str, np.ndarray], Any], device: th.device
) -> Union[th.Tensor, TensorDict]:
    """
    Moves the observation to the given device.

    :param obs:
    :param device: PyTorch device
    :return: PyTorch tensor of the observation on a desired device.
    """
    if isinstance(obs, np.ndarray) or isinstance(obs, th.Tensor):
        return th.as_tensor(obs, device=device)
    elif isinstance(obs, dict):
        return {key: th.as_tensor(_obs, device=device) for (key, _obs) in obs.items()}
    else:
        raise Exception(f"Unrecognized type of observation {type(obs)}")
