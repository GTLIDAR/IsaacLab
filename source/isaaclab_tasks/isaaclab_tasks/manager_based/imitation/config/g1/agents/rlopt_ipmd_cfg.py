from isaaclab.utils import configclass

from isaaclab_rl.rlopt import IPMDRLOptConfig
from isaaclab_tasks.manager_based.imitation.config.g1.imitation_g1_env_cfg import (
    G1_POLICY_OBS_KEYS,
    G1_REWARD_OBS_KEYS,
)


# Convenience configurations for different scenarios
@configclass
class G1ImitationRLOptIPMDConfig(IPMDRLOptConfig):
    """RLOpt IPMD (PPO-based) configuration for G1 imitation.

    Observation key convention (``concatenate_terms=False``):

    The IsaacLab wrapper flattens nested observation groups so that each
    term name (e.g. ``"joint_pos"``) is a top-level TensorDict key.
    ``input_keys`` therefore lists the flat term names directly.
    The reward estimator can use a *subset* of those terms (e.g. excluding
    ``last_actions``).
    """

    def __post_init__(self):
        """Post-initialization setup."""
        super().__post_init__()

        assert isinstance(self, IPMDRLOptConfig)
        assert self.value_function is not None, "Value function configuration must be provided."

        # -- Observation key wiring (concatenate_terms=False) --
        # Wrapper flattens "policy" group → term names become top-level keys.
        self.policy.input_keys = list(G1_POLICY_OBS_KEYS)
        self.value_function.input_keys = list(G1_POLICY_OBS_KEYS)

        # Reward estimator uses a subset (no last_actions).
        self.ipmd.reward_input_keys = list(G1_REWARD_OBS_KEYS)

        self.collector.init_random_frames = 0

        # Match RSL-RL's num_steps_per_env=24 with num_envs
        # frames_per_batch = num_steps_per_env (collector handles num_envs internally)
        self.collector.frames_per_batch = 24
        self.replay_buffer.size = 4096 * 24

        # Match RSL-RL: num_learning_epochs=5, num_mini_batches=4
        # mini_batch_size = (num_envs * frames_per_batch) / num_mini_batches
        # For 4096 envs: 4096 * 24 / 4 = 24576
        self.loss.epochs = 5
        self.loss.mini_batch_size = 4096 * 24 // 4
        self.loss.loss_critic_type = "l2"

        # PPO-specific settings to match RSL-RL
        self.ppo.clip_epsilon = 0.2
        self.ppo.gae_lambda = 0.95
        self.ppo.entropy_coeff = 0.008
        self.ppo.critic_coeff = 1.0
        self.ppo.clip_value = True
        self.ppo.normalize_advantage = True
        self.ppo.clip_log_std = False
        self.ppo.log_std_init = 0.0

        # Optimizer settings
        self.optim.lr = 1.0e-3
        self.optim.max_grad_norm = 1.0
        self.optim.scheduler = "adaptive"
        self.optim.desired_kl = 0.01

        # Loss settings
        self.loss.gamma = 0.99

        self.collector.total_frames = 300000000

        # IPMD-specific settings
        self.ipmd.reward_input_type = "s"
        self.ipmd.use_estimated_rewards_for_ppo = True
        self.ipmd.bc_loss_coeff = 0.0
        self.ipmd.expert_batch_size = 4000
