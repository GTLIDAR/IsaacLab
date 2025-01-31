# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to run the RL environment for the cartpole balancing task."""

"""Launch Isaac Sim Simulator first."""

import argparse
from time import sleep

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Tutorial on running the cartpole RL environment."
)
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of environments to spawn."
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

from omni.isaac.lab.envs import ManagerBasedRLEnv

from omni.isaac.lab_tasks.manager_based.locomotion.velocity.config.digit_v3.l2t_env_cfg import (
    DigitV3L2TFlatEnvCfg,
)


def main():
    """Main function."""
    # create environment configuration
    env_cfg = DigitV3L2TFlatEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    # setup RL environment
    env = ManagerBasedRLEnv(cfg=env_cfg)
    # robot = env.scene["robot"]
    # robot.articulation.set_joint_position(
    # simulate physics
    count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            # reset
            if count % 300 == 0:
                count = 0
                obs, _ = env.reset()
                print("-" * 80)
                print("[INFO]: Resetting environment...")
                # print("[Env 0]: joint: ", obs["teacher"])
            # sample random actions
            joint_efforts = torch.randn_like(env.action_manager.action)
            # step the environment
            obs, rew, terminated, truncated, info = env.step(joint_efforts)
            # print current orientation of pole
            print("[Env 0]: Pole joint: ", obs["teacher"][0][1].item())
            # update counter
            count += 1
            # sleep(100)

    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
