# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from typing import Any, Literal

from isaaclab.utils import configclass
from isaaclab_rl.torchrl import RLOptPPOConfig


# Convenience configurations for different scenarios
@configclass
class AntRLOptPPOConfig(RLOptPPOConfig):
    """RLOpt PPO configuration for ANYMAL-D."""

    def __post_init__(self):
        """Post-initialization setup."""
        # Set mini_batch_size to frames_per_batch if not specified
        self.loss.mini_batch_size = int(
            self.collector.frames_per_batch / self.loss.epochs
        )

        # Adjust configurations for flat terrain (typically easier)
        self.policy.num_cells = [512, 256, 128]
        self.value_net.num_cells = [512, 256, 128]
        self.collector.total_frames = 300000000  # Fewer frames for flat terrain

        self.policy_in_keys = ["policy"]
        self.value_net_in_keys = ["policy"]
