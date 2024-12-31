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
import omni.physics.tensors.impl.api as physx

import omni.isaac.lab.sim as sim_utils
import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import RayCaster
from omni.isaac.lab.sensors import ContactSensor


if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def clock(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Clock time using sin and cos from the phase of the simulation."""
    phase = env.get_phase()
    return torch.cat(
        [
            torch.sin(2 * torch.pi * phase).unsqueeze(1),
            torch.cos(2 * torch.pi * phase).unsqueeze(1),
        ],
        dim=1,
    ).to(env.device)


def starting_leg(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Starting leg of the robot."""
    return env.get_starting_leg().unsqueeze(-1).float()


def get_environment_parameters(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Get environment parameters including the gravity and the friction coefficient."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    # Get physics_scene_gravity and physics_scene_friction
    # set the gravity into the physics simulation
    physics_sim_view: physx.SimulationView = (
        sim_utils.SimulationContext.instance().physics_sim_view
    )
    gravity_floats = physics_sim_view.get_gravity()
    gravity = (
        torch.tensor(gravity_floats, device=env.device, dtype=torch.float32)
        .unsqueeze(0)
        .repeat(env.num_envs, 1)
    )

    # retrieve material buffer
    materials = (
        asset.root_physx_view.get_material_properties()
        .to(env.device)
        .view(env.num_envs, -1)
    )
    # Get additional base mass
    # get the current masses of the bodies
    masses = asset.root_physx_view.get_masses().to(env.device).view(env.num_envs, -1)

    # Get external torque and push force
    external_force = asset._external_force_b.to(env.device).view(env.num_envs, -1)  # type: ignore
    external_torque = asset._external_torque_b.to(env.device).view(env.num_envs, -1)  # type: ignore

    # return torch.cat([gravity, friction], dim=0).to(env.device)
    return torch.cat(
        [gravity, materials, masses, external_force, external_torque], dim=1
    )
