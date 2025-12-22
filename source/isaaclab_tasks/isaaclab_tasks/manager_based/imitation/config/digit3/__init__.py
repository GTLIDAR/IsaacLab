import gymnasium as gym

from . import agents, imitation_digit3_env_cfg

__all__ = ["imitation_digit3_env_cfg", "agents"]

gym.register(
    id="Isaac-Imitation-Digit3-v0",
    entry_point="isaaclab.envs:ImitationRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.imitation_digit3_env_cfg:ImitationDigit3EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Digit3ImitationPPORunnerCfg",
    },
)
