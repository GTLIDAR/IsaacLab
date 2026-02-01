from isaaclab.utils import configclass

from isaaclab_rl.rlopt import SACRLOptConfig


# Convenience configurations for different scenarios
@configclass
class G1RLOptSACConfig(SACRLOptConfig):
    """RLOpt SAC configuration for G1.
    
    Note: input_dim values are left as None for lazy initialization.
    The networks will automatically infer dimensions from the environment specs.
    """

    def __post_init__(self):
        """Post-initialization setup."""
        super().__post_init__()

        assert self.q_function is not None, "Q function configuration must be provided."

        self.policy.input_keys = ["policy"]
        self.q_function.input_keys = ["policy"]

        # Collector settings
        self.collector.frames_per_batch = 24  # num_steps_per_env (multiplied by num_envs in train.py)
        self.collector.init_random_frames = 0  # Overridden in train.py

        # Loss settings - SAC uses single epoch per batch
        self.loss.epochs = 1
        self.loss.mini_batch_size = 256  # Typical SAC batch size
        self.loss.gamma = 0.99

        # SAC-specific settings
        self.sac.alpha_init = 1.0
        self.sac.target_entropy = "auto"  # -dim(action) will be computed
        self.sac.num_qvalue_nets = 2  # Twin Q-networks
        
        # Target network update (soft update)
        self.optim.target_update_polyak = 0.995  # tau = 1 - polyak = 0.005
        self.optim.lr = 3e-4

        # Note: input_dim is left as None for lazy initialization
        # Networks will infer dimensions from environment specs


@configclass
class G1RLOptSACFlatConfig(G1RLOptSACConfig):
    """RLOpt SAC configuration for G1 on flat terrain."""

    def __post_init__(self):
        """Post-initialization setup for flat terrain."""
        super().__post_init__()

        assert self.q_function is not None, "Q function configuration must be provided."

        # Network architecture for flat terrain
        self.policy.num_cells = [256, 256, 256]
        self.q_function.num_cells = [256, 256, 256]
        
        # Training duration
        self.collector.total_frames = 100_000_000
