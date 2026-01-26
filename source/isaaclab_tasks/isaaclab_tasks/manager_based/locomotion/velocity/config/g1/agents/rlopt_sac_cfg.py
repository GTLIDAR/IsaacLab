from isaaclab.utils import configclass

from isaaclab_rl.rlopt import SACRLOptConfig


# Convenience configurations for different scenarios
@configclass
class G1RLOptSACConfig(SACRLOptConfig):
    """RLOpt SAC configuration for G1."""

    def __post_init__(self):
        """Post-initialization setup."""
        super().__post_init__()

        assert self.q_function is not None, "Q function configuration must be provided."

        # Set mini_batch_size to frames_per_batch if not specified
        self.loss.mini_batch_size = int(self.collector.frames_per_batch / self.loss.epochs)

        self.policy.input_keys = ["policy"]
        self.q_function.input_keys = ["policy"]

        self.loss.epochs = 1
        self.collector.init_random_frames = self.collector.frames_per_batch * 2


@configclass
class G1RLOptSACFlatConfig(G1RLOptSACConfig):
    """RLOpt SAC configuration for G1 on flat terrain."""

    def __post_init__(self):
        """Post-initialization setup for flat terrain."""
        super().__post_init__()

        assert self.q_function is not None, "Q function configuration must be provided."

        # Adjust configurations for flat terrain (typically easier)
        self.policy.num_cells = [256, 128]
        self.q_function.num_cells = [256, 128]
        self.collector.total_frames = 300000000  # Fewer frames for flat terrain
