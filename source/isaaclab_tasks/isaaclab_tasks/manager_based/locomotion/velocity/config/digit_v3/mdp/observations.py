# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create observation terms.

The functions can be passed to the :class:`isaaclab.managers.ObservationTermCfg` object to enable
the observation introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING
import omni.physics.tensors.impl.api as physx

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster
from isaaclab.sensors import ContactSensor


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


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


def _aggregate_ray_stats(scanner: RayCaster) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute terrain height statistics from raycaster.
    """
    ray_hits_w = scanner.data.ray_hits_w # shape: (num_envs, num_rays, 3)
    
    # Compute relative height
    sensor_h = scanner.data.pos_w[:, 2].unsqueeze(1)  # shape: (num_envs, 1)
    ray_hits_z = sensor_h - ray_hits_w[..., 2]  # shape: (num_envs, num_rays)

    if hasattr(scanner.data, "ray_distances"):
        dist_mask = scanner.data.ray_distances < scanner.cfg.max_distance
    else:
        dist_mask = torch.ones_like(ray_hits_z, dtype=torch.bool)
    finite_mask = torch.isfinite(ray_hits_z)
    valid_mask = dist_mask & finite_mask

    # Compute mean
    denom = valid_mask.sum(dim=1, keepdim=True).clamp(min=1)
    masked_values = torch.where(valid_mask, ray_hits_z, torch.zeros_like(ray_hits_z))
    mean_z = masked_values.sum(dim=1, keepdim=True) / denom
    
    # Fill invalid entries with per-env mean to avoid skewing statistics
    ray_hits_z_filled = torch.where(valid_mask, ray_hits_z, mean_z.expand_as(ray_hits_z))
    ray_hits_z_filled = torch.nan_to_num(ray_hits_z_filled, nan=0.0, posinf=0.0, neginf=0.0)

    mean_val = torch.nan_to_num(mean_z, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Compute variance
    var_val = torch.var(ray_hits_z_filled, dim=1, keepdim=True)
    var_val = torch.nan_to_num(var_val, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Compute min and max values
    min_val, _ = torch.min(ray_hits_z_filled, dim=1, keepdim=True)
    max_val, _ = torch.max(ray_hits_z_filled, dim=1, keepdim=True)
    min_val = torch.nan_to_num(min_val, nan=0.0, posinf=0.0, neginf=0.0)
    max_val = torch.nan_to_num(max_val, nan=0.0, posinf=0.0, neginf=0.0)
    
    rng_val = torch.nan_to_num(max_val - min_val, nan=0.0, posinf=0.0, neginf=0.0)
    
    return mean_val, var_val, min_val, max_val, rng_val


def foot_terrain_flatness_features(
    env: ManagerBasedRLEnv,
    foot_scanner_core: tuple[str, str],
    foot_scanner_safe: tuple[str, str] | None = None,
) -> torch.Tensor:
    """
    Aggregate dual-zone foot terrain features for flat surface detection.
    """
    feats = []
    for i, name in enumerate(foot_scanner_core):
        core_scanner: RayCaster = env.scene.sensors[name]
        c_mean, c_var, _c_min, _c_max, c_rng = _aggregate_ray_stats(core_scanner)
        
        if foot_scanner_safe is not None:
            safe_scanner: RayCaster = env.scene.sensors[foot_scanner_safe[i]]
            s_mean, s_var, _s_min, _s_max, s_rng = _aggregate_ray_stats(safe_scanner)
            mean_diff = c_mean - s_mean
            
            # Concatenate features in fixed order: [core(3), safe(3), diff(1)]
            foot_feat = torch.cat([c_mean, c_var, c_rng, s_mean, s_var, s_rng, mean_diff], dim=1)
        else:
            # Core zone only: [core(3)]
            foot_feat = torch.cat([c_mean, c_var, c_rng], dim=1)
        
        feats.append(foot_feat)
    
    feats_cat = torch.cat(feats, dim=1)
    
    return torch.nan_to_num(feats_cat, nan=0.0, posinf=0.0, neginf=0.0)

