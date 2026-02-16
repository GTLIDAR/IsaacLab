# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rlopt import FastTD3RLOptConfig


# Convenience configurations for different scenarios
@configclass
class G1RLOptFastTD3Config(FastTD3RLOptConfig):
    """RLOpt FastTD3 configuration for G1.

    Note: input_dim values are left as None for lazy initialization.
    The networks will automatically infer dimensions from the environment specs.
    """

    def __post_init__(self):
        """Post-initialization setup."""
        super().__post_init__()

        # Collector settings
        self.collector.frames_per_batch = 1  # num_steps_per_env (multiplied by num_envs in train.py)
        self.collector.init_random_frames = 10

        # FastTD3 settings
        self.fasttd3.gamma = 0.99
        self.fasttd3.policy_noise = 0.001
        self.fasttd3.noise_clip = 0.5
        self.fasttd3.use_cdq = True
        self.fasttd3.disable_bootstrap = False
        self.fasttd3.v_min = -10.0
        self.fasttd3.v_max = 10.0
        self.fasttd3.batch_size = 8
        self.fasttd3.action_bounds = 1.0
        self.fasttd3.std_max = 0.4
        self.fasttd3.num_atoms = 251
        self.fasttd3.tau = 0.1
        self.fasttd3.num_updates = 4
        self.fasttd3.num_steps = 8

        # optimizer
        self.optim.optimizer = "adamw"
        self.optim.weight_decay = 0.1
        self.optim.lr = 3e-4
        self.optim.max_grad_norm = None

        # buffer
        self.replay_buffer.size = 1024 * 10
        self.replay_buffer.prb = False


@configclass
class G1RLOptFastTD3FlatConfig(G1RLOptFastTD3Config):
    """RLOpt SAC configuration for G1 on flat terrain."""

    def __post_init__(self):
        """Post-initialization setup for flat terrain."""
        super().__post_init__()

        # assert self.q_function is not None, "Q function configuration must be provided."

        # Network architecture for flat terrain
        self.fasttd3.num_steps = 8
        self.fasttd3.num_updates = 4

        # Training duration
        self.collector.total_frames = 100_000_000
