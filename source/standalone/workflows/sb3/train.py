# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
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

from omni.isaac.lab.app import AppLauncher

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
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
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
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import os
from datetime import datetime

import torch
from rlopt.agent.torch.ppo.ppo import PPO
from rlopt.agent.torch.l2t.l2t import L2T
from rlopt.agent.torch.l2t.recurrent_l2t import RecurrentL2T
from rlopt.common.torch.buffer import RolloutBuffer as RLOptRolloutBuffer
from rlopt.common.torch.buffer import DictRolloutBuffer as RLOptDictRolloutBuffer
from rlopt.common.torch.buffer import RLOptDictRecurrentReplayBuffer
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecNormalize

import wandb
from wandb.integration.sb3 import WandbCallback


from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab.utils.io import dump_pickle, dump_yaml


import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import load_cfg_from_registry, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.sb3 import (
    process_sb3_cfg,
    Sb3VecEnvGPUWrapper,
    L2tSb3VecEnvGPUWrapper,
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def main():
    """Train with stable-baselines agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    agent_cfg = load_cfg_from_registry(args_cli.task, "sb3_cfg_entry_point")

    # override configuration with command line arguments
    if args_cli.seed is not None:
        agent_cfg["seed"] = args_cli.seed  # type: ignore

    # max iterations for training
    if args_cli.max_iterations:
        agent_cfg["n_timesteps"] = (  # type: ignore
            args_cli.max_iterations * agent_cfg["n_steps"] * env_cfg.scene.num_envs  # type: ignore
        )

    # directory for logging into
    log_dir = os.path.join(
        "logs", "sb3", args_cli.task, datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # post-process agent configuration
    agent_cfg = process_sb3_cfg(agent_cfg)  # type: ignore
    # read configurations about the agent-training
    policy_arch = agent_cfg.pop("policy")
    n_timesteps = agent_cfg.pop("n_timesteps")

    # create isaac environment
    env = gym.make(
        args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None
    )
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)  # type: ignore
    # wrap around environment for stable baselines
    env = Sb3VecEnvGPUWrapper(env)  # type: ignore
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
        policy_arch,
        env,
        verbose=0,
        rollout_buffer_class=RLOptRolloutBuffer,
        **agent_cfg
    )
    # configure the logger
    new_logger = configure(log_dir, ["stdout", "tensorboard"])
    agent.set_logger(new_logger)

    # callbacks for agent
    checkpoint_callback = CheckpointCallback(
        save_freq=1000, save_path=log_dir, name_prefix="model", verbose=2
    )

    # train the agent
    agent.learn(
        total_timesteps=n_timesteps,
        callback=checkpoint_callback,
        progress_bar=True,
    )

    # save the final model
    agent.save(os.path.join(log_dir, "model"))

    # close the simulator
    env.close()


def train_l2t():
    """Train with stable-baselines agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    agent_cfg = load_cfg_from_registry(args_cli.task, "sb3_cfg_entry_point")

    # override configuration with command line arguments
    if args_cli.seed is not None:
        agent_cfg["seed"] = args_cli.seed  # type: ignore

    # max iterations for training
    if args_cli.max_iterations:
        agent_cfg["n_timesteps"] = (  # type: ignore
            args_cli.max_iterations * agent_cfg["n_steps"] * env_cfg.scene.num_envs  # type: ignore
        )

    # directory for logging into
    log_dir = os.path.join(
        "logs", "sb3", args_cli.task, datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # post-process agent configuration
    agent_cfg = process_sb3_cfg(agent_cfg)  # type: ignore
    # read configurations about the agent-training
    policy_arch = agent_cfg.pop("policy")
    n_timesteps = agent_cfg.pop("n_timesteps")

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
    env = L2tSb3VecEnvGPUWrapper(env)  # type: ignore
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

    wandb.tensorboard.patch(root_logdir=log_dir)
    # initialize wandb and make callback
    run = wandb.init(
        project="l2t_digit",
        entity="rl-digit",
        config=agent_cfg,
        sync_tensorboard=True,
        monitor_gym=False,
        save_code=False,
    )
    wandb_callback = WandbCallback()

    # create agent from stable baselines
    agent = L2T(
        policy_arch,
        env,
        verbose=1,
        rollout_buffer_class=RLOptDictRolloutBuffer,
        **agent_cfg
    )
    # configure the logger
    new_logger = configure(log_dir, ["tensorboard"])
    agent.set_logger(new_logger)

    # callbacks for agent
    checkpoint_callback = CheckpointCallback(
        save_freq=1000, save_path=log_dir, name_prefix="model", verbose=0
    )

    # chain the callbacks
    callback_list = CallbackList([checkpoint_callback, wandb_callback])

    # train the agent
    agent.learn(
        total_timesteps=n_timesteps,
        callback=callback_list,
        progress_bar=True,
    )

    # save the final model
    agent.save(os.path.join(log_dir, "model"))

    # close the simulator
    env.close()

    # finish wandb
    run.finish()  # type: ignore


def train_recurrentl2t():
    """Train with stable-baselines agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    agent_cfg = load_cfg_from_registry(args_cli.task, "sb3_cfg_entry_point")

    # override configuration with command line arguments
    if args_cli.seed is not None:
        agent_cfg["seed"] = args_cli.seed  # type: ignore

    # max iterations for training
    if args_cli.max_iterations:
        agent_cfg["n_timesteps"] = (  # type: ignore
            args_cli.max_iterations * agent_cfg["n_steps"] * env_cfg.scene.num_envs  # type: ignore
        )

    # directory for logging into
    log_dir = os.path.join(
        "logs", "sb3", args_cli.task, datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # post-process agent configuration
    agent_cfg = process_sb3_cfg(agent_cfg)  # type: ignore
    # read configurations about the agent-training
    policy_arch = agent_cfg.pop("policy")
    n_timesteps = agent_cfg.pop("n_timesteps")

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
    env = L2tSb3VecEnvGPUWrapper(env)  # type: ignore
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

    wandb.tensorboard.patch(root_logdir=log_dir)
    # initialize wandb and make callback
    run = wandb.init(
        project="l2t_digit",
        entity="rl-digit",
        name=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        config=agent_cfg,
        sync_tensorboard=True,
        monitor_gym=False,
        save_code=False,
    )
    wandb_callback = WandbCallback()

    # create agent from stable baselines
    agent = RecurrentL2T(
        policy_arch,
        env,
        verbose=1,
        rollout_buffer_class=RLOptDictRecurrentReplayBuffer,
        **agent_cfg
    )
    # configure the logger
    new_logger = configure(log_dir, ["tensorboard"])
    agent.set_logger(new_logger)

    # callbacks for agent
    checkpoint_callback = CheckpointCallback(
        save_freq=1000, save_path=log_dir, name_prefix="model", verbose=0
    )

    # chain the callbacks
    callback_list = CallbackList([checkpoint_callback, wandb_callback])

    # train the agent
    agent.learn(
        total_timesteps=n_timesteps,
        callback=callback_list,
        progress_bar=True,
    )

    # save the final model
    agent.save(os.path.join(log_dir, "model"))

    # close the simulator
    env.close()

    # finish wandb
    run.finish()  # type: ignore


if __name__ == "__main__":
    # run the main function
    # main()
    # train_l2t()
    train_recurrentl2t()
    # close sim app
    simulation_app.close()
