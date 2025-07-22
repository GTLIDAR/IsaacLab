from isaaclab.utils import configclass
from isaaclab_rl.torchrl import RLOptPPOConfig


@configclass
class G1ImitationPPOConfig(RLOptPPOConfig):
    """RLOpt PPO configuration for G1 Flat."""

    def __post_init__(self):
        """Post-initialization setup for flat terrain."""
        # Set mini_batch_size to frames_per_batch if not specified
        self.loss.mini_batch_size = int(
            self.collector.frames_per_batch / self.loss.epochs
        )

        # Adjust configurations for flat terrain (typically easier)
        self.policy.num_cells = [256, 256, 128]
        self.value_net.num_cells = [256, 256, 128]
        self.collector.total_frames = (
            self.collector.frames_per_batch * 1_000
        )  # Fewer frames for flat terrain
