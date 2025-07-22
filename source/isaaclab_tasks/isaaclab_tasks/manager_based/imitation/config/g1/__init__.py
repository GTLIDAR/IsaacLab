import gymnasium as gym

from . import agents, imitation_g1_env_cfg

__all__ = ["imitation_g1_env_cfg", "agents"]

gym.register(
    id="Isaac-Imitation-G1-v0",
    entry_point="isaaclab.envs:ImitationRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.imitation_g1_env_cfg:ImitationG1EnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1ImitationPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "rlopt_cfg_entry_point": f"{agents.__name__}.rlopt_ppo_cfg:G1ImitationPPOConfig",
    },
)
