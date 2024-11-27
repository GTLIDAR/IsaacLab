from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Tuple
import math

from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import ContactSensor

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


@torch.jit.script
def create_stance_mask(phase: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Creates a stance mask based on the gait phase.
    """
    sin_pos = torch.sin(2 * torch.pi * phase).unsqueeze(-1).repeat(1, 2)
    stance_mask = torch.where(sin_pos >= 0, 1, 0)
    stance_mask[:, 1] = 1 - stance_mask[:, 1]
    stance_mask[torch.abs(sin_pos) < 0.1] = 1

    mask_2 = 1 - stance_mask
    mask_2[torch.abs(sin_pos) < 0.1] = 1
    return stance_mask, mask_2


def reward_feet_contact_number(
    env, sensor_cfg: SceneEntityCfg, pos_rw: float, neg_rw: float
) -> torch.Tensor:
    """
    Calculates a reward based on the number of feet contacts aligning with the gait phase.
    Rewards or penalizes depending on whether the foot contact matches the expected gait phase.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]  # type: ignore
        .norm(dim=-1)
        .max(dim=1)[0]
        > 1.0
    )
    # print("contact", contacts.shape, contacts)
    phase = env.get_phase()
    stance_mask, mask_2 = create_stance_mask(phase)

    reward = torch.where(contacts == stance_mask, pos_rw, neg_rw)
    return torch.mean(reward, dim=1)


def foot_clearance_reward(
    env: ManagerBasedRLEnv,
    target_height: float,
    std: float,
    tanh_mult: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Reward the swinging feet for clearing a specified height off the ground
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    com_z = asset.data.root_pos_w[:, 2]
    standing_position_com_z = asset.data.default_root_state[:, 2]
    standing_height = com_z - standing_position_com_z
    standing_position_toe_roll_z = (
        0.0626 + 0.1
    )  # recorded from the default position, 0.1 compensation for walking
    offset = (standing_height + standing_position_toe_roll_z).unsqueeze(-1)

    foot_z_target_error = torch.square(
        (
            asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
            - (target_height + offset).repeat(1, 2)
        ).clip(max=0.0)
    )

    # weighted by the velocity of the feet in the xy plane
    foot_velocity_tanh = torch.tanh(
        tanh_mult
        * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)
    )
    reward = foot_velocity_tanh * foot_z_target_error
    return torch.exp(-torch.sum(reward, dim=1) / std)


@torch.jit.script
def height_target(t: torch.Tensor):
    a5, a4, a3, a2, a1, a0 = [9.6, 12.0, -18.8, 5.0, 0.1, 0.0]
    return a5 * t**5 + a4 * t**4 + a3 * t**3 + a2 * t**2 + a1 * t + a0


def track_foot_height(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    std: float,
) -> torch.Tensor:
    """"""

    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]  # type: ignore
    contacts = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]  # type: ignore
        .norm(dim=-1)
        .max(dim=1)[0]
        > 1.0
    )
    com_z = asset.data.root_pos_w[:, 2]
    standing_position_com_z = asset.data.default_root_state[:, 2]
    standing_height = com_z - standing_position_com_z
    standing_position_toe_roll_z = (
        0.0626 + 0.1
    )  # recorded from the default position, 0.1 compensation for walking
    offset = standing_height + standing_position_toe_roll_z

    phase = env.get_phase()

    stance_mask, mask_2 = create_stance_mask(phase)

    swing_mask = 1 - stance_mask

    filt_foot = torch.where(swing_mask == 1, foot_z, torch.zeros_like(foot_z))

    phase_mod = torch.fmod(phase, 0.5)
    feet_z_target = height_target(phase_mod) + offset
    feet_z_value = torch.sum(filt_foot, dim=1)

    error = torch.square(feet_z_value - feet_z_target)
    reward = torch.exp(-error / std**2)

    return reward


@torch.compile
def bezier_curve(control_points, t):
    """
    Computes Bézier curve for given control points and parameter t.
    """
    n = len(control_points) - 1  # Degree of the Bézier curve
    dim = control_points.shape[1]  # Dimension of control points
    curve_points = torch.zeros(
        (t.shape[0], dim), dtype=control_points.dtype, device=t.device
    )

    # Calculate the Bézier curve points
    for k in range(n + 1):
        binomial_coeff = math.comb(n, k)
        bernstein_poly = binomial_coeff * (t**k) * ((1 - t) ** (n - k))
        curve_points += bernstein_poly.unsqueeze(1) * control_points[k]

    return curve_points


@torch.compile
def desired_height(phase, starting_foot):
    """
    Computes the desired heights for both legs for each environment.

    Args:
        phase (torch.Tensor): Tensor of shape (n_envs,) representing current phase values.
        starting_foot (torch.Tensor): Tensor of shape (n_envs,) with values 0 or 1 indicating which foot starts swinging first.

    Returns:
        torch.Tensor: Tensor of shape (n_envs, 2) containing desired heights for both legs.
    """
    n_envs = phase.shape[0]
    desired_heights = torch.zeros((n_envs, 2), dtype=phase.dtype, device=phase.device)

    # Step length (L) and max height (H) for the swing phase
    L = 0.5  # Step length
    H = 0.15  # Maximum height in the swing phase

    # Define control points for the swing phase Bézier curve
    control_points_swing = torch.tensor(
        [
            [0.0, 0.0],  # Start of swing phase
            [0.3 * L, 0.1 * H],  # Lift-off point
            [0.6 * L, H],  # Peak of swing
            [L, 0.0],  # Landing point
        ],
        dtype=torch.float32,
        device=phase.device,
    )

    # Loop over legs (0: left leg, 1: right leg)
    for leg in [0, 1]:
        # Determine which environments have this leg as the starting foot
        is_starting_leg = starting_foot == leg
        is_other_leg = ~is_starting_leg

        # Swing phase masks for the starting leg
        swing_mask_starting_leg = is_starting_leg & (phase >= 0.02) & (phase < 0.5)
        t_swing_starting = (phase[swing_mask_starting_leg] - 0.02) / 0.48

        # Swing phase masks for the other leg
        swing_mask_other_leg = is_other_leg & (phase >= 0.52) & (phase < 1.0)
        t_swing_other = (phase[swing_mask_other_leg] - 0.52) / 0.48

        # Combine swing masks
        swing_mask_leg = swing_mask_starting_leg | swing_mask_other_leg

        # Initialize t_swing for all environments
        t_swing = torch.zeros(n_envs, dtype=phase.dtype, device=phase.device)
        t_swing[swing_mask_starting_leg] = t_swing_starting
        t_swing[swing_mask_other_leg] = t_swing_other

        # Compute desired heights for swing phase
        if swing_mask_leg.any():
            t_swing_leg = t_swing[swing_mask_leg]
            swing_heights = bezier_curve(control_points_swing, t_swing_leg)
            desired_heights[swing_mask_leg, leg] = swing_heights[
                :, 1
            ]  # Only the y-coordinate (height)

        # Stance phase (excluding double stance): foot is on the ground (height = 0)
        # No action needed since desired_heights is initialized to zero

    # For double stance phases, both legs are already set to height = 0
    return desired_heights


def feet_distance_l1(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, min_dist: float, max_dist: float
) -> torch.Tensor:
    """
    Calculates the reward based on the distance between the feet. Penalize feet get close to each other or too far away.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :3]
    foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)

    d_min = torch.clamp(foot_dist - min_dist, -0.5, 0.0)
    # d_max = torch.clamp(foot_dist - max_dist, 0, 0.5)

    return torch.exp(-torch.abs(d_min) * 100)


def joint_torques_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Penalize joint torques on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.linalg.norm((asset.data.applied_torque), dim=1)


def torque_applied_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, std: float
) -> torch.Tensor:
    """
    Calculates the L2 norm of the torques applied to the joints of the robot.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    torques = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.exp(-torch.norm(torques, dim=1) / std)


def center_of_mass_deviation_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, std: float
) -> torch.Tensor:
    """
    Calculates the L2 norm of the deviation of the center of mass from the center of two foot.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    com_xy = asset.data.root_pos_w[:, :2]
    foot_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :3]
    foot_center = torch.mean(
        foot_pos, dim=1
    ).squeeze()  # from (batch, 2, 3) to (batch, 3)
    foot_center_xy = foot_center[:, :2]
    return torch.exp(-torch.norm(com_xy - foot_center_xy, dim=1) / std)


def shoulder_center_deviation_foot_center_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, std: float
) -> torch.Tensor:
    """
    Calculates the L2 norm of the deviation of the center of the shoulder from the center of two foot.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    shoulder_foot_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :3]
    shoulder_pos = shoulder_foot_pos[:, :2, :]
    shoulder_pos_center_xy = torch.mean(shoulder_pos, dim=1).squeeze()[:, :2]
    foot_pos = shoulder_foot_pos[:, 2:, :]
    foot_center_xy = torch.mean(foot_pos, dim=1).squeeze()[:, :2]
    return torch.exp(-torch.norm(shoulder_pos_center_xy - foot_center_xy, dim=1) / std)
