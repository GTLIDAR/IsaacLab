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


def get_foot_trajectory_observations(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Compute and return the target foot trajectory positions as observations.

    Args:
        env (ManagerBasedRLEnv): The simulation environment.
        asset_cfg (SceneEntityCfg): Configuration for the robot asset.
        sensor_cfg (SceneEntityCfg): Configuration for the contact sensor.

    Returns:
        torch.Tensor: A tensor of shape [n_envs, y], where y = num_feet * 3,
                      containing the target foot trajectory positions (x, y, z)
                      for each foot.
    """
    # Extract the robot asset
    asset: RigidObject = env.scene[asset_cfg.name]

    # Get the foot positions in world frame [n_envs, num_feet, 3]
    foot_xyz = asset.data.body_pos_w[:, asset_cfg.body_ids, :3]

    # Retrieve contact sensor data
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]  # type: ignore
    contacts = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]  # type: ignore
        .norm(dim=-1)
        .max(dim=1)[0]
        > 1.0
    )  # [n_envs, num_feet]
    swing_mask = ~contacts  # True where foot is in swing phase

    # Get the CoM position and velocity [n_envs, 3]
    com_pos = asset.data.root_pos_w[:, :3]  # [n_envs, 3]
    com_vel = asset.data.root_vel_w[:, :3]  # [n_envs, 3]

    # Extract yaw velocity (angular velocity around z-axis)
    ang_vel = asset.data.root_vel_w[:, 3:6]  # [n_envs, 3]
    yaw_vel = ang_vel[:, 2]  # [n_envs]

    # Expand CoM position, velocity, and yaw velocity to match foot dimensions
    com_pos_expanded = com_pos.unsqueeze(1).expand(
        -1, foot_xyz.shape[1], -1
    )  # [n_envs, num_feet, 3]
    com_vel_expanded = com_vel.unsqueeze(1).expand(
        -1, foot_xyz.shape[1], -1
    )  # [n_envs, num_feet, 3]
    yaw_vel_expanded = yaw_vel.unsqueeze(1).expand(
        -1, foot_xyz.shape[1]
    )  # [n_envs, num_feet]

    # Raibert heuristic parameters
    k_p = 0.05  # Proportional gain for linear velocity
    k_d = 0.1  # Gain for yaw velocity term (adjust as needed)

    # Initialize foot target positions
    foot_xyz_target = foot_xyz.clone()

    # Compute the rotational adjustment terms
    rotational_adjustment_x = -k_d * yaw_vel_expanded * com_pos_expanded[:, :, 1]
    rotational_adjustment_y = k_d * yaw_vel_expanded * com_pos_expanded[:, :, 0]

    # Compute desired foot positions in x-y plane using modified Raibert heuristic
    foot_xyz_target[:, :, 0] = (
        com_pos_expanded[:, :, 0]
        + k_p * com_vel_expanded[:, :, 0]
        + rotational_adjustment_x
    )
    foot_xyz_target[:, :, 1] = (
        com_pos_expanded[:, :, 1]
        + k_p * com_vel_expanded[:, :, 1]
        + rotational_adjustment_y
    )

    # Parameters for von Mises distribution
    h_max = 0.2  # Maximum foot height during swing
    kappa = 0.8  # Concentration parameter for the von Mises distribution

    # Current phase from environment [n_envs]
    phase = env.get_phase()  # [n_envs]

    # Total gait cycle duration from environment (for both legs)
    T_total = env.phase_dt  # [scalar]

    # Assuming equal swing and stance durations, swing duration is half of T_total
    T_swing = T_total / 2.0  # [scalar]

    # Expand phase to match foot dimensions [n_envs, num_feet]
    phase_expanded = phase.unsqueeze(1).expand(
        -1, foot_xyz.shape[1]
    )  # [n_envs, num_feet]

    # Define phase offsets for each foot [num_feet]
    # For a biped robot with two legs, legs are 180 degrees out of phase
    foot_phase_offsets = torch.tensor([0.0, 0.5], device=phase.device)  # [num_feet]
    foot_phase_offsets = foot_phase_offsets.unsqueeze(0).expand(
        phase_expanded.shape
    )  # [n_envs, num_feet]

    # Adjust the phase for each foot
    foot_phase = (phase_expanded + foot_phase_offsets) % 1.0  # [n_envs, num_feet]

    # Determine if each foot is in swing phase based on the adjusted phase
    # Assuming swing phase occurs when phase is in [0, 0.5)
    swing_phase_mask = (foot_phase >= 0.0) & (foot_phase < 0.5)  # [n_envs, num_feet]

    # Compute time within the swing phase for each foot
    t_swing = foot_phase * T_total  # [n_envs, num_feet]
    # For swing phase from 0.0 to 0.5 in phase, time is from 0 to T_swing

    # Compute 'z_t' for all feet
    z_t = foot_xyz[:, :, 2].clone()  # Initialize with current foot heights

    # Compute 'z_t' using von Mises distribution for swing feet
    swing_feet = swing_phase_mask
    z_t_swing = h_max * (
        1
        - torch.exp(kappa * torch.cos(2 * torch.pi * t_swing / T_swing))
        / torch.exp(torch.tensor(kappa, device=t_swing.device))
    )
    z_t[swing_feet] = z_t_swing[swing_feet]  # Update z_t for swing feet

    # Update the target foot height using torch.where
    foot_xyz_target[:, :, 2] = torch.where(swing_feet, z_t, foot_xyz[:, :, 2])

    # Reshape the target foot positions to [n_envs, y], where y = num_feet * 3
    n_envs = foot_xyz_target.shape[0]
    num_feet = foot_xyz_target.shape[1]
    foot_target_flat = foot_xyz_target.reshape(n_envs, num_feet * 3)  # [n_envs, y]

    # Cancatenate with the current foot positions
    foot_xyz_flat = foot_xyz.reshape(n_envs, num_feet * 3)  # [n_envs, y]
    foot_target_flat = torch.cat([foot_target_flat, foot_xyz_flat], dim=1)

    return foot_target_flat
