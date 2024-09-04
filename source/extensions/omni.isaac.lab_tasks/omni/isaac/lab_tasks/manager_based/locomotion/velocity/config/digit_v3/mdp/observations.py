# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create observation terms.

The functions can be passed to the :class:`omni.isaac.lab.managers.ObservationTermCfg` object to enable
the observation introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import RayCaster

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def clock(
    env: ManagerBasedRLEnv)-> torch.Tensor:
    """Clock time using sin and cos from the phase of the simulation."""
    phase = env.get_phase()
    return torch.cat([torch.sin(2 * torch.pi * phase).unsqueeze(1), 
                   torch.cos(2 * torch.pi * phase).unsqueeze(1)], dim=1).to(env.device)

    




