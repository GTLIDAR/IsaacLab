import gymnasium as gym

from . import agents, imitation_g1_env_cfg

__all__ = ["imitation_g1_env_cfg", "agents"]

gym.register(
    id="Isaac-Imitation-G1-v0",
    entry_point="isaaclab.envs:ImitationRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.imitation_g1_env_cfg:ImitationG1EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1ImitationPPORunnerCfg",
        "rlopt_ppo_cfg_entry_point": f"{agents.__name__}.rlopt_ppo_cfg:G1ImitationRLOptPPOConfig",
    },
)
