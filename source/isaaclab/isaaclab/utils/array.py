# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing utilities for working with different array backends."""

# needed to import for allowing type-hinting: torch.device | str | None
from __future__ import annotations

import numpy as np
import torch
from typing import Union

import warp as wp

TensorData = Union[np.ndarray, torch.Tensor, wp.array]
"""Type definition for a tensor data.

Union of numpy, torch, and warp arrays.
"""

TENSOR_TYPES = {
    "numpy": np.ndarray,
    "torch": torch.Tensor,
    "warp": wp.array,
}
"""A dictionary containing the types for each backend.

The keys are the name of the backend ("numpy", "torch", "warp") and the values are the corresponding type
(``np.ndarray``, ``torch.Tensor``, ``wp.array``).
"""

TENSOR_TYPE_CONVERSIONS = {
    "numpy": {wp.array: lambda x: x.numpy(), torch.Tensor: lambda x: x.detach().cpu().numpy()},
    "torch": {wp.array: lambda x: wp.torch.to_torch(x), np.ndarray: lambda x: torch.from_numpy(x)},
    "warp": {np.array: lambda x: wp.array(x), torch.Tensor: lambda x: wp.torch.from_torch(x)},
}
"""A nested dictionary containing the conversion functions for each backend.

The keys of the outer dictionary are the name of target backend ("numpy", "torch", "warp"). The keys of the
inner dictionary are the source backend (``np.ndarray``, ``torch.Tensor``, ``wp.array``).
"""


def convert_to_torch(
    array: TensorData,
    dtype: torch.dtype = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Converts a given array into a torch tensor.

    The function tries to convert the array to a torch tensor. If the array is a numpy/warp arrays, or python
    list/tuples, it is converted to a torch tensor. If the array is already a torch tensor, it is returned
    directly.

    If ``device`` is None, then the function deduces the current device of the data. For numpy arrays,
    this defaults to "cpu", for torch tensors it is "cpu" or "cuda", and for warp arrays it is "cuda".

    Note:
        Since PyTorch does not support unsigned integer types, unsigned integer arrays are converted to
        signed integer arrays. This is done by casting the array to the corresponding signed integer type.

    Args:
        array: The input array. It can be a numpy array, warp array, python list/tuple, or torch tensor.
        dtype: Target data-type for the tensor.
        device: The target device for the tensor. Defaults to None.

    Returns:
        The converted array as torch tensor.
    """
    # Convert array to tensor
    # if the datatype is not currently supported by torch we need to improvise
    # supported types are: https://pytorch.org/docs/stable/tensors.html
    if isinstance(array, torch.Tensor):
        tensor = array
    elif isinstance(array, np.ndarray):
        if array.dtype == np.uint32:
            array = array.astype(np.int32)
        # need to deal with object arrays (np.void) separately
        tensor = torch.from_numpy(array)
    elif isinstance(array, wp.array):
        if array.dtype == wp.uint32:
            array = array.view(wp.int32)
        tensor = wp.to_torch(array)
    else:
        tensor = torch.Tensor(array)
    # Convert tensor to the right device
    if device is not None and str(tensor.device) != str(device):
        tensor = tensor.to(device)
    # Convert dtype of tensor if requested
    if dtype is not None and tensor.dtype != dtype:
        tensor = tensor.type(dtype)

    return tensor


# -----------------------------------------------------------------------------
# NaN checking utilities
# -----------------------------------------------------------------------------


def check_out_of_bounds(obs_buf, limit: float = 1e3):
    """Recursively detect values with magnitude larger than ``limit``.

    Returns a boolean mask of shape (num_envs,)."""
    if isinstance(obs_buf, torch.Tensor):
        if obs_buf.ndim == 0:
            return (torch.abs(obs_buf) > limit).view(1)
        elif obs_buf.ndim == 1:
            return torch.abs(obs_buf) > limit
        else:
            reduce_dims = tuple(range(1, obs_buf.ndim))
            return torch.any(torch.abs(obs_buf) > limit, dim=reduce_dims)
    elif isinstance(obs_buf, dict):
        mask = None
        for v in obs_buf.values():
            cur = check_out_of_bounds(v, limit)
            mask = cur if mask is None else torch.logical_or(mask, cur)
        if mask is None:
            raise ValueError("Empty observation dictionary in check_out_of_bounds.")
        return mask
    else:
        raise TypeError(f"Unsupported type in check_out_of_bounds: {type(obs_buf)}")


def check_nan(obs_buf):
    """Recursively check an observation buffer for NaNs.

    The observation buffer can be one of the following:

    1. ``torch.Tensor`` with leading dimension equal to number of environments.
    2. ``dict`` mapping strings to tensors as described above.
    3. ``dict`` that maps to other dictionaries of tensors (arbitrary depth).

    In all cases, the function returns a 1-D boolean tensor of shape ``(num_envs,)``
    where each element is ``True`` if *any* value belonging to that environment
    contains ``NaN``.

    Args:
        obs_buf: Observation buffer that may contain tensors, dictionaries of
            tensors, or nested dictionaries of tensors.

    Returns:
        torch.Tensor: 1-D boolean tensor indicating environments that contain
        NaNs.
    """
    # Note: torch is already imported at module level.
    if isinstance(obs_buf, torch.Tensor):
        # Collapse all dimensions except the first (env) dimension and check for NaNs.
        if obs_buf.ndim == 0:
            # Scalar tensor – treat as single-env.
            return torch.isnan(obs_buf).view(1)
        elif obs_buf.ndim == 1:
            # Vector per env (num_envs,)
            return torch.isnan(obs_buf)
        else:
            # Any higher dim – reduce over all but env dimension.
            reduce_dims = tuple(range(1, obs_buf.ndim))
            return torch.any(torch.isnan(obs_buf), dim=reduce_dims)

    elif isinstance(obs_buf, dict):
        nan_mask = None
        # Iterate over nested values and recursively accumulate NaN masks.
        for value in obs_buf.values():
            current_mask = check_nan(value)
            nan_mask = (
                current_mask
                if nan_mask is None
                else torch.logical_or(nan_mask, current_mask)
            )
        if nan_mask is None:
            raise ValueError(
                "Provided dictionary is empty – cannot determine num_envs for NaN check."
            )
        return nan_mask

    else:
        raise TypeError(
            "obs_buf must be a torch.Tensor or a (nested) dictionary of tensors, "
            f"got type {type(obs_buf)} instead."
        )
