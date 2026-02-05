from isaaclab.utils import configclass

from isaaclab_rl.rlopt import PPORLOptConfig


# Convenience configurations for different scenarios
@configclass
class G1ImitationRLOptPPOConfig(PPORLOptConfig):
    """RLOpt PPO configuration for G1.

    This config is designed to match RSL-RL's PPO behavior as closely as possible.
    Key matching parameters:
    - clip_log_std=False (RSL-RL doesn't clip log_std)
    - normalize_advantage=True (RSL-RL normalizes advantages globally)
    - scheduler="adaptive" with KL-based LR adjustment
    - num_mini_batches=4 equivalent via mini_batch_size calculation
    """

    def __post_init__(self):
        """Post-initialization setup."""
        super().__post_init__()

        assert isinstance(self, PPORLOptConfig)

        assert self.value_function is not None, "Value function configuration must be provided."

        self.policy.input_keys = ["policy"]
        self.value_function.input_keys = ["policy"]

        self.collector.init_random_frames = 0

        # Match RSL-RL's num_steps_per_env=24 with num_envs
        # frames_per_batch = num_steps_per_env (collector handles num_envs internally)
        self.collector.frames_per_batch = 24
        self.replay_buffer.size = 4096 * 24

        # Match RSL-RL: num_learning_epochs=5, num_mini_batches=4
        self.loss.epochs = 5
        # mini_batch_size = (num_envs * frames_per_batch) / num_mini_batches
        # For 4096 envs: 4096 * 24 / 4 = 24576
        self.loss.mini_batch_size = 4096 * 24 // 4
        self.loss.loss_critic_type = "l2"

        # PPO-specific settings to match RSL-RL
        self.ppo.clip_epsilon = 0.2
        self.ppo.gae_lambda = 0.95
        self.ppo.entropy_coeff = 0.008
        self.ppo.critic_coeff = 1.0
        self.ppo.clip_value = True
        self.ppo.normalize_advantage = True  # RSL-RL normalizes advantages by default
        # RSL-RL does NOT clip log_std
        self.ppo.clip_log_std = False
        self.ppo.log_std_init = 0.0  # exp(0) = 1.0, matches RSL-RL's init_noise_std=1.0

        # Optimizer settings to match RSL-RL
        self.optim.lr = 1.0e-3
        self.optim.max_grad_norm = 1.0
        self.optim.scheduler = "adaptive"  # KL-based adaptive LR like RSL-RL
        self.optim.desired_kl = 0.01

        # Loss settings
        self.loss.gamma = 0.99

        self.collector.total_frames = 300000000
