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
    L2tSb3VecEnvGPUWrapper,
)
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecNormalize

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab.utils import class_to_dict
from isaaclab_tasks.utils import get_checkpoint_path

import wandb
from wandb.integration.sb3 import WandbCallback

from rlopt.agent import RecurrentL2T, RecurrentStudent
from rlopt.common import RLOptDictRecurrentReplayBuffer


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
    # max iterations for training
    if args_cli.max_iterations is not None:
        agent_cfg["n_timesteps"] = (
            args_cli.max_iterations * agent_cfg["n_steps"] * env_cfg.scene.num_envs
        )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg["seed"]

    if args_cli.resume_training:
        # directory for logging into
        log_root_path = os.path.join("logs", "sb3", args_cli.task)
        log_root_path = os.path.abspath(log_root_path)
        # check checkpoint is valid
        if args_cli.checkpoint is None:
            if args_cli.use_last_checkpoint:
                checkpoint = "model_.*.zip"
            else:
                checkpoint = "model.zip"
            checkpoint_path = get_checkpoint_path(log_root_path, ".*", checkpoint)
        else:
            checkpoint_path = args_cli.checkpoint

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

    wandb.tensorboard.patch(root_logdir=log_dir)  # type: ignore

    # initialize wandb and make callback
    run = wandb.init(
        project="L2T Digit flat" if "flat" in args_cli.task else "L2T Digit",
        entity="rl-digit",
        name=log_time_note,
        config=agent_cfg | class_to_dict(env_cfg),
        sync_tensorboard=True,
        monitor_gym=True if args_cli.video else False,
        save_code=False,
    )
    wandb_callback = WandbCallback()

    # create agent from stable baselines
    agent = RecurrentL2T(
        policy_arch,
        env,
        verbose=1,
        rollout_buffer_class=RLOptDictRecurrentReplayBuffer,
        **agent_cfg,
    )

    # load the model if required
    if args_cli.resume_training:
        agent.set_parameters(checkpoint_path)  # type: ignore

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


@hydra_task_config(args_cli.task, "rlopt_cfg_entry_point")
def student_only(
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
    # max iterations for training
    if args_cli.max_iterations is not None:
        agent_cfg["n_timesteps"] = (
            args_cli.max_iterations * agent_cfg["n_steps"] * env_cfg.scene.num_envs
        )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg["seed"]

    if args_cli.resume_training:
        # directory for logging into
        log_root_path = os.path.join("logs", "rlopt", args_cli.task)
        log_root_path = os.path.abspath(log_root_path)
        # check checkpoint is valid
        if args_cli.checkpoint is None:
            if args_cli.use_last_checkpoint:
                checkpoint = "model_.*.zip"
            else:
                checkpoint = "model.zip"
            checkpoint_path = get_checkpoint_path(log_root_path, ".*", checkpoint)
        else:
            checkpoint_path = args_cli.checkpoint  # type: ignore

    log_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    note = "_" + args_cli.note if args_cli.note else ""
    log_time_note = log_time + note
    # directory for logging into
    log_dir = os.path.join("logs", "sb3", args_cli.task, log_time_note)
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

    wandb.tensorboard.patch(root_logdir=log_dir)  # type: ignore

    # initialize wandb and make callback
    run = wandb.init(
        project="L2T Digit Perception flat"
        if "flat" in args_cli.task
        else "L2T Digit Perception",
        entity="rl-digit",
        name=log_time_note,
        config=agent_cfg | class_to_dict(env_cfg),
        sync_tensorboard=True,
        monitor_gym=True if args_cli.video else False,
        save_code=False,
    )
    wandb_callback = WandbCallback()

    teacher_agent = RecurrentL2T(
        policy_arch,
        env,
        verbose=1,
        rollout_buffer_class=RLOptDictRecurrentReplayBuffer,
        **agent_cfg,
    )
    teacher_agent.set_parameters(args_cli.checkpoint)

    teacher_policy = teacher_agent.policy

    # create agent from stable baselines
    agent = RecurrentStudent(
        policy_arch,
        env,
        verbose=1,
        rollout_buffer_class=RLOptDictRecurrentReplayBuffer,
        **agent_cfg,
        teacher_policy=teacher_policy,  # type: ignore
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
    main()  # type: ignore
    # student_only()  # type: ignore
    # close sim app
    simulation_app.close()
