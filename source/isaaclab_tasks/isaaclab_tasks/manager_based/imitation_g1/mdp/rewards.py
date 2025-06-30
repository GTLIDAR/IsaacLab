# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def joint_pos_target_l2(
    env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # wrap the joint positions to (-pi, pi)
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    # compute the reward
    return torch.sum(torch.square(joint_pos - target), dim=1)


def qpos_imitation_l2(
    env: "ManagerBasedRLEnv", reference_joint_names=None
) -> torch.Tensor:
    """
    Penalize L2 distance between the reference and actual qpos for joints that are in both the reference and the robot.
    Args:
        env: The environment instance.
        reference_joint_names: Optional list of joint names in reference order. If None, uses env.cfg.reference_joint_names or env.get_loco_joint_names().
    Returns:
        Tensor of shape (num_envs,) with the L2 error for each environment.
    """
    # Get actual qpos from the robot (IsaacLab order)
    qpos_actual = env.scene["robot"].data.joint_pos
    # Get reference qpos from the dataset (reference order)
    qpos_reference = env.compute_reference(key="qpos")
    # Compute mapping between IsaacLab and reference order
    isaaclab_joint_names = env.scene["robot"].joint_names
    if reference_joint_names is None:
        reference_joint_names = getattr(env.cfg, "reference_joint_names", None)
    if reference_joint_names is None:
        reference_joint_names = env.get_loco_joint_names()
    # Find common joints and their indices in both orders
    common_names = [
        name for name in reference_joint_names if name in isaaclab_joint_names
    ]
    idx_actual = [isaaclab_joint_names.index(name) for name in common_names]
    idx_reference = [reference_joint_names.index(name) for name in common_names]
    # Select only the common joints
    qpos_actual_common = qpos_actual[:, idx_actual]
    qpos_reference_common = qpos_reference[:, idx_reference]
    # Compute L2 error
    l2_error = torch.sum((qpos_actual_common - qpos_reference_common) ** 2, dim=1)
    return -l2_error  # Negative for reward (higher is better)
