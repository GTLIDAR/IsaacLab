# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with Stable Baselines3.

Since Stable-Baselines3 does not support buffers living on GPU directly,
we recommend using smaller number of environments. Otherwise,
there will be significant overhead in GPU->CPU transfer.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Train an RL agent with Stable-Baselines3."
)
parser.add_argument(
    "--video", action="store_true", default=False, help="Record videos during training."
)
parser.add_argument(
    "--video_length",
    type=int,
    default=200,
    help="Length of the recorded video (in steps).",
)
parser.add_argument(
    "--video_interval",
    type=int,
    default=2000,
    help="Interval between video recordings (in steps).",
)
parser.add_argument(
    "--num_envs", type=int, default=None, help="Number of environments to simulate."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--seed", type=int, default=None, help="Seed used for the environment"
)
parser.add_argument(
    "--max_iterations", type=int, default=None, help="RL Policy training iterations."
)
parser.add_argument(
    "--resume_training",
    action="store_true",
    default=False,
    help="Resume training from a checkpoint.",
)
parser.add_argument(
    "--note",
    type=str,
    default=None,
    help="Add a note to the log directory name.",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="Path to the checkpoint to resume training from.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import os
import random
from datetime import datetime

from isaaclab_rl.sb3 import (
    process_sb3_cfg,
)
from isaaclab_rl.torchrl import TorchRLEnvWrapper


from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

from isaaclab_tasks.utils.hydra import hydra_task_config


from omegaconf import OmegaConf

from rlopt.envs.gymlike import make_isaaclab_gym_env
from rlopt.agent.ppo import PPO


@hydra_task_config(args_cli.task, "rlopt_cfg_entry_point")
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict
):
    """Train with stable-baselines agent."""
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = (
        args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    )
    agent_cfg["seed"] = (
        args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg["seed"]

    log_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    note = "_" + args_cli.note if args_cli.note else ""
    log_time_note = log_time + note
    # directory for logging into
    log_dir = os.path.join("logs", "rlopt", args_cli.task, log_time_note)
    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # post-process agent configuration
    agent_cfg = process_sb3_cfg(agent_cfg)  # type: ignore
    # read configurations about the agent-training

    # create isaac environment
    env = gym.make(
        args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None
    )

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)  # type: ignore
    # wrap around environment for stable baselines
    # env = TorchRLEnvWrapper(env)  # type: ignore
    env = make_isaaclab_gym_env(
        env=env,
    )

    # set the seed
    env.seed(seed=agent_cfg["seed"])

    if "normalize_input" in agent_cfg:
        env = VecNormalize(
            env,
            training=True,
            norm_obs="normalize_input" in agent_cfg
            and agent_cfg.pop("normalize_input"),
            norm_reward="normalize_value" in agent_cfg
            and agent_cfg.pop("normalize_value"),
            clip_obs="clip_obs" in agent_cfg and agent_cfg.pop("clip_obs"),
            gamma=agent_cfg["gamma"],
            clip_reward=np.inf,
        )

    # create agent from stable baselines
    agent = PPO(
        env=env,
        config=OmegaConf.create(agent_cfg),
    )
    # train the agent
    agent.train()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()  # type: ignore

    # close sim app
    simulation_app.close()
