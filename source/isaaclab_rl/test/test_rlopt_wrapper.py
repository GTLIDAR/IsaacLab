# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch the simulator
app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app


"""Rest everything follows."""

import gymnasium as gym
import pytest
import torch
from tensordict import TensorDictBase

import carb  # type: ignore
import omni.usd  # type: ignore

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent

from isaaclab_rl.rlopt import IsaacLabWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


@pytest.fixture(scope="module")
def registered_tasks():
    # acquire all Isaac environments names
    registered_tasks = list()
    for task_spec in gym.registry.values():
        if "Isaac" in task_spec.id:
            cfg_entry_point = gym.spec(task_spec.id).kwargs.get("env_cfg_entry_point")
            if cfg_entry_point is not None:
                registered_tasks.append(task_spec.id)
    # sort environments by name
    registered_tasks.sort()
    registered_tasks = registered_tasks[:3]

    # this flag is necessary to prevent a bug where the simulation gets stuck randomly when running the
    # test on many environments.
    carb_settings_iface = carb.settings.get_settings()
    carb_settings_iface.set_bool("/physics/cooking/ujitsoCollisionCooking", False)

    # print all existing task names
    print(">>> All registered environments:", registered_tasks)
    return registered_tasks


def test_random_actions(registered_tasks):
    """Run random actions and check environments return valid signals."""
    # common parameters
    num_envs = 64
    device = "cuda"
    for task_name in registered_tasks:
        # Use pytest's subtests
        print(f">>> Running test for environment: {task_name}")
        # create a new stage
        omni.usd.get_context().new_stage()
        # reset the rtx sensors carb setting to False
        carb.settings.get_settings().set_bool("/isaaclab/render/rtx_sensors", False)
        try:
            # parse configuration
            env_cfg = parse_env_cfg(task_name, device=device, num_envs=num_envs)
            # create environment
            env = gym.make(task_name, cfg=env_cfg)
            # convert to single-agent instance if required by the RL algorithm
            if isinstance(env.unwrapped, DirectMARLEnv):
                env = multi_agent_to_single_agent(env)  # type: ignore
            # wrap environment
            env = IsaacLabWrapper(env, device=torch.device(device))
        except Exception as e:
            if "env" in locals() and hasattr(env, "_is_closed"):  # type: ignore
                env.close()
            else:
                if hasattr(e, "obj") and hasattr(e.obj, "_is_closed"):  # type: ignore
                    e.obj.close()  # type: ignore
            pytest.fail(f"Failed to set-up the environment for task {task_name}. Error: {e}")

        # avoid shutdown of process on simulation stop
        env.unwrapped.sim._app_control_on_stop_handle = None  # type: ignore

        # reset environment
        td = env.reset()
        # check signal
        assert _check_valid_tensor(td)  # type: ignore

        # simulate environment for 100 steps
        with torch.inference_mode():
            for _ in range(100):
                transition = env.rand_step()  # type: ignore
                assert "next" in transition.keys()
                assert "policy" in transition["next"].keys()  # type: ignore

        # close the environment
        print(f">>> Closing environment: {task_name}")
        env.close()


"""
Helper functions.
"""


@staticmethod  # type: ignore
def _check_valid_tensor(data: torch.Tensor | dict | TensorDictBase) -> bool:
    """Checks if given data does not have corrupted values.

    Args:
        data: Data buffer.

    Returns:
        True if the data is valid.
    """
    if isinstance(data, TensorDictBase):
        valid_tensor = True
        for value in data.values():
            valid_tensor &= _check_valid_tensor(value)  # type: ignore
        return valid_tensor
    if isinstance(data, torch.Tensor):
        if data.is_floating_point():
            return not torch.isnan(data).any().item()
        return True
    elif isinstance(data, dict):
        valid_tensor = True
        for value in data.values():
            valid_tensor &= _check_valid_tensor(value)
        return valid_tensor
    else:
        raise ValueError(f"Input data of invalid type: {type(data)}.")
