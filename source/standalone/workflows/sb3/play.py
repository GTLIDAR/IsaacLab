# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from Stable-Baselines3."""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Play a checkpoint of an RL agent from Stable-Baselines3."
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
    "--checkpoint", type=str, default=None, help="Path to model checkpoint."
)
parser.add_argument(
    "--use_last_checkpoint",
    action="store_true",
    help="When no checkpoint provided, use the last saved model. Otherwise use the best saved model.",
)
parser.add_argument(
    "--l2t",
    action="store_true",
    help="Use the student model.",
)
parser.add_argument(
    "--video", action="store_true", default=False, help="Record videos during training."
)
parser.add_argument(
    "--video_length",
    type=int,
    default=2000,
    help="Length of the recorded video (in steps).",
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
import torch
from datetime import datetime

# from stable_baselines3 import PPO
from rlopt.agent.torch.ppo.ppo import PPO
from rlopt.agent.torch.l2t.l2t import L2T
from rlopt.agent.torch.l2t.recurrent_l2t import RecurrentL2T
from stable_baselines3.common.vec_env import VecNormalize

from omni.isaac.lab.utils.dict import print_dict

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils.parse_cfg import (
    get_checkpoint_path,
    load_cfg_from_registry,
    parse_env_cfg,
)
from omni.isaac.lab_tasks.utils.wrappers.sb3 import (
    Sb3VecEnvWrapper,
    process_sb3_cfg,
    L2tSb3VecEnvGPUWrapper,
)


from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab.utils.io import dump_pickle, dump_yaml


def main():
    """Play with stable-baselines agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    agent_cfg = load_cfg_from_registry(args_cli.task, "sb3_cfg_entry_point")
    # post-process agent configuration
    agent_cfg = process_sb3_cfg(agent_cfg)  # type: ignore

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    # wrap around environment for stable baselines
    env = Sb3VecEnvWrapper(env)  # type: ignore

    # normalize environment (if needed)
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
    # create agent from stable baselines
    print(f"Loading checkpoint from: {checkpoint_path}")
    agent = L2T.load(checkpoint_path, env, print_system_info=True)

    # reset environment
    obs = env.reset()
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            obs = {"student": obs["student"].cpu().numpy()}
            # agent stepping
            actions, _ = agent.predict(obs, deterministic=True)  # type: ignore
            # env stepping
            obs, _, _, _ = env.step(actions)

    # close the simulator
    env.close()


def main_l2t_student():
    """Play with stable-baselines agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    agent_cfg = load_cfg_from_registry(args_cli.task, "sb3_cfg_entry_point")
    # post-process agent configuration
    agent_cfg = process_sb3_cfg(agent_cfg)  # type: ignore

    # directory for logging into
    log_dir = os.path.join(
        "logs", "sb3", args_cli.task, datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )

    # create isaac environment
    env = gym.make(
        args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None
    )
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos"),
            "step_trigger": lambda step: step % 1 == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)  # type: ignore
    # wrap around environment for stable baselines
    env = L2tSb3VecEnvGPUWrapper(env)  # type: ignore

    # normalize environment (if needed)
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
    # create agent from stable baselines
    print(f"Loading checkpoint from: {checkpoint_path}")
    agent = L2T.load(checkpoint_path, env, print_system_info=True)

    # reset environment
    obs = env.reset()
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            obs = {"student": obs["student"].cpu().numpy()}
            # agent stepping
            actions, _ = agent.student_predict(obs, deterministic=True)  # type: ignore
            # env stepping
            obs, _, _, _ = env.step(actions)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

    # close the simulator
    env.close()


def main_recurrentl2t_student():
    from sb3_contrib.common.recurrent.type_aliases import RNNStates  # type: ignore
    from stable_baselines3.common.buffers import BaseBuffer

    """Play with stable-baselines agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    agent_cfg = load_cfg_from_registry(args_cli.task, "sb3_cfg_entry_point")
    # post-process agent configuration
    agent_cfg = process_sb3_cfg(agent_cfg)  # type: ignore

    # directory for logging into
    log_dir = os.path.join(
        "logs", "sb3", args_cli.task, datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )

    # create isaac environment
    env = gym.make(
        args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None
    )
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos"),
            "step_trigger": lambda step: step % 1 == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)  # type: ignore
    # wrap around environment for stable baselines
    env = L2tSb3VecEnvGPUWrapper(env)  # type: ignore

    # normalize environment (if needed)
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
    # create agent from stable baselines
    print(f"Loading checkpoint from: {checkpoint_path}")
    agent = RecurrentL2T.load(checkpoint_path, env, print_system_info=True)
    _last_episode_starts = None

    _last_lstm_states = None
    episode_starts = None
    # reset environment
    obs = env.reset()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            obs = {"student": obs["student"].cpu().numpy()}

            # agent stepping
            # actions, _ = agent.student_predict(obs, deterministic=True)  # type: ignore
            actions, lstm_states = agent.student_policy.predict(
                obs["student"],
                _last_lstm_states,
                episode_starts,
            )
            # env stepping
            obs, reward, done, info = env.step(actions)

            _last_lstm_states = lstm_states
            _last_episode_starts = done
            episode_starts = (
                _last_episode_starts.clone().to(agent.device).type(torch.float32)
            )
            print(f"actions: {actions}")
            if done.any():
                _last_episode_starts = None
                _last_lstm_states = None
                episode_starts = None

        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

    # close the simulator
    env.close()


if __name__ == "__main__":
    if args_cli.l2t:
        # run the main function
        # main_l2t_student()
        main_recurrentl2t_student()
    else:
        # run the main function
        main()
    # close sim app
    simulation_app.close()
