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


# From stable baselines
def explained_variance(
    y_pred: np.ndarray | th.Tensor, y_true: np.ndarray | th.Tensor
) -> Union[float, np.floating[Any], np.ndarray, th.Tensor]:
    """
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]

    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero

    :param y_pred: the prediction
    :param y_true: the expected value
    :return: explained variance of ypred and y
    """
    assert y_true.ndim == 1 and y_pred.ndim == 1
    if isinstance(y_pred, np.ndarray):
        var_y = np.var(y_true)
        return np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
    elif isinstance(y_pred, th.Tensor) and isinstance(y_true, th.Tensor):
        var_y = th.var(y_true).item()
        return np.nan if var_y == 0 else 1 - th.var(y_true - y_pred).item() / var_y
    else:
        raise ValueError(
            "y_pred and y_true must be of the same type (np.ndarray or th.Tensor)"
        )
