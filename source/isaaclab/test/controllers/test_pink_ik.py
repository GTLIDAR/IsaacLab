# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""
# Import pinocchio in the main script to force the use of the dependencies installed by IsaacLab and not the one installed by Isaac Sim
# pinocchio is required by the Pink IK controller
import sys

if sys.platform != "win32":
    import pinocchio  # noqa: F401

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import contextlib
import gymnasium as gym
import json
import numpy as np
import re
import torch
from pathlib import Path

import omni.usd
import pytest
from pink.configuration import Configuration
from pink.tasks import FrameTask

from isaaclab.utils.math import (
    axis_angle_from_quat,
    matrix_from_quat,
    quat_from_matrix,
    quat_inv,
)

import isaaclab_tasks  # noqa: F401
import isaaclab_tasks.manager_based.locomanipulation.pick_place  # noqa: F401
import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


def load_test_config(env_name):
    """Load test configuration based on environment type."""
    # Determine which config file to load based on environment name
    if "G1" in env_name:
        config_file = "pink_ik_g1_test_configs.json"
    elif "GR1" in env_name:
        config_file = "pink_ik_gr1_test_configs.json"
    else:
        raise ValueError(f"Unknown environment type in {env_name}. Expected G1 or GR1.")

    config_path = Path(__file__).parent / "test_ik_configs" / config_file
    with open(config_path) as f:
        return json.load(f)


def is_waist_enabled(env_cfg):
    """Check if waist joints are enabled in the environment configuration."""
    if not hasattr(env_cfg.actions, "upper_body_ik"):
        return False

    pink_controlled_joints = env_cfg.actions.upper_body_ik.pink_controlled_joint_names

    # Also check for pattern-based joint names (e.g., "waist_.*_joint")
    return any(re.match("waist", joint) for joint in pink_controlled_joints)


def create_test_env(env_name, num_envs):
    """Create a test environment with the Pink IK controller."""
    device = "cuda:0"

    omni.usd.get_context().new_stage()

    try:
        env_cfg = parse_env_cfg(env_name, device=device, num_envs=num_envs)
        # Modify scene config to not spawn the packing table to avoid collision with the robot
        del env_cfg.scene.packing_table
        del env_cfg.terminations.object_dropping
        del env_cfg.terminations.time_out
        return gym.make(env_name, cfg=env_cfg).unwrapped, env_cfg
    except Exception as e:
        print(f"Failed to create environment: {str(e)}")
        raise


@pytest.fixture(
    scope="module",
    params=[
        "Isaac-PickPlace-GR1T2-Abs-v0",
        "Isaac-PickPlace-GR1T2-WaistEnabled-Abs-v0",
        "Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-v0",
        "Isaac-PickPlace-Locomanipulation-G1-Abs-v0",
    ],
)
def env_and_cfg(request):
    """Create environment and configuration for tests."""
    env_name = request.param

    # Load the appropriate test configuration based on environment type
    test_cfg = load_test_config(env_name)

    env, env_cfg = create_test_env(env_name, num_envs=1)

    # Get only the FrameTasks from variable_input_tasks
    variable_input_tasks = [
        task for task in env_cfg.actions.upper_body_ik.controller.variable_input_tasks if isinstance(task, FrameTask)
    ]
    assert len(variable_input_tasks) == 2, "Expected exactly two FrameTasks (left and right hand)."
    frames = [task.frame for task in variable_input_tasks]
    # Try to infer which is left and which is right
    left_candidates = [f for f in frames if "left" in f.lower()]
    right_candidates = [f for f in frames if "right" in f.lower()]
    assert (
        len(left_candidates) == 1 and len(right_candidates) == 1
    ), f"Could not uniquely identify left/right frames from: {frames}"
    left_eef_urdf_link_name = left_candidates[0]
    right_eef_urdf_link_name = right_candidates[0]

    # Set up camera view
    env.sim.set_camera_view(eye=[2.5, 2.5, 2.5], target=[0.0, 0.0, 1.0])

    # Create test parameters from test_cfg
    test_params = {
        "position": test_cfg["tolerances"]["position"],
        "rotation": test_cfg["tolerances"]["rotation"],
        "pd_position": test_cfg["tolerances"]["pd_position"],
        "check_errors": test_cfg["tolerances"]["check_errors"],
        "left_eef_urdf_link_name": left_eef_urdf_link_name,
        "right_eef_urdf_link_name": right_eef_urdf_link_name,
    }

    try:
        yield env, env_cfg, test_cfg, test_params
    finally:
        env.close()


@pytest.fixture
def test_setup(env_and_cfg):
    """Set up test case - runs before each test."""
    env, env_cfg, test_cfg, test_params = env_and_cfg

    num_joints_in_robot_hands = env_cfg.actions.upper_body_ik.controller.num_hand_joints

    # Get Action Term and IK controller
    action_term = env.action_manager.get_term(name="upper_body_ik")
    pink_controllers = action_term._ik_controllers
    articulation = action_term._asset

    # Initialize Pink Configuration for forward kinematics
    test_kinematics_model = Configuration(
        pink_controllers[0].pink_configuration.model,
        pink_controllers[0].pink_configuration.data,
        pink_controllers[0].pink_configuration.q,
    )
    left_target_link_name = env_cfg.actions.upper_body_ik.target_eef_link_names["left_wrist"]
    right_target_link_name = env_cfg.actions.upper_body_ik.target_eef_link_names["right_wrist"]

    return {
        "env": env,
        "env_cfg": env_cfg,
        "test_cfg": test_cfg,
        "test_params": test_params,
        "num_joints_in_robot_hands": num_joints_in_robot_hands,
        "action_term": action_term,
        "pink_controllers": pink_controllers,
        "articulation": articulation,
        "test_kinematics_model": test_kinematics_model,
        "left_target_link_name": left_target_link_name,
        "right_target_link_name": right_target_link_name,
        "left_eef_urdf_link_name": test_params["left_eef_urdf_link_name"],
        "right_eef_urdf_link_name": test_params["right_eef_urdf_link_name"],
    }


@pytest.mark.parametrize(
    "test_name",
    [
        "horizontal_movement",
        "horizontal_small_movement",
        "stay_still",
        "forward_waist_bending_movement",
        "vertical_movement",
        "rotation_movements",
    ],
)
def test_movement_types(test_setup, test_name):
    """Test different movement types using parametrization."""
    test_cfg = test_setup["test_cfg"]
    env_cfg = test_setup["env_cfg"]

    if test_name not in test_cfg["tests"]:
        print(f"Skipping {test_name} test for {env_cfg.__class__.__name__} environment (test not defined)...")
        pytest.skip(f"Test {test_name} not defined for {env_cfg.__class__.__name__}")
        return

    test_config = test_cfg["tests"][test_name]

    # Check if test requires waist bending and if waist is enabled
    requires_waist_bending = test_config.get("requires_waist_bending", False)
    waist_enabled = is_waist_enabled(env_cfg)

    env_name = "Isaac-PickPlace-GR1T2-Abs-v0"
    device = "cuda:0"
    env_cfg = parse_env_cfg(env_name, device=device, num_envs=pink_ik_test_config["num_envs"])

    # create environment from loaded config
    env = gym.make(env_name, cfg=env_cfg).unwrapped


def run_movement_test(test_setup, test_config, test_cfg, aux_function=None):
    """Run a movement test with the given configuration."""
    env = test_setup["env"]
    num_joints_in_robot_hands = test_setup["num_joints_in_robot_hands"]

    left_hand_poses = np.array(test_config["left_hand_pose"], dtype=np.float32)
    right_hand_poses = np.array(test_config["right_hand_pose"], dtype=np.float32)

    curr_pose_idx = 0
    test_counter = 0
    num_runs = 0

    # Get poses from config
    left_hand_roll_link_pose = pink_ik_test_config["left_hand_roll_link_pose"].copy()
    right_hand_roll_link_pose = pink_ik_test_config["right_hand_roll_link_pose"].copy()

    # simulate environment -- run everything in inference mode
    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        while simulation_app.is_running() and not simulation_app.is_exiting():

            num_runs += 1
            setpoint_poses = left_hand_roll_link_pose + right_hand_roll_link_pose
            actions = setpoint_poses + [0.0] * pink_ik_test_config["num_joints_in_robot_hands"]
            actions = torch.tensor(actions, device=device)
            actions = torch.stack([actions for _ in range(env.num_envs)])

            obs, _, _, _, _ = env.step(actions)

            left_hand_roll_link_pose_obs = obs["policy"]["robot_links_state"][
                :, env.scene["robot"].data.body_names.index("left_hand_roll_link"), :7
            ]
            right_hand_roll_link_pose_obs = obs["policy"]["robot_links_state"][
                :, env.scene["robot"].data.body_names.index("right_hand_roll_link"), :7
            ]

            # The setpoints are wrt the env origin frame
            # The observations are also wrt the env origin frame
            left_hand_roll_link_feedback = left_hand_roll_link_pose_obs
            left_hand_roll_link_setpoint = (
                torch.tensor(left_hand_roll_link_pose, device=device).unsqueeze(0).repeat(env.num_envs, 1)
            )
            left_hand_roll_link_pos_error = left_hand_roll_link_setpoint[:, :3] - left_hand_roll_link_feedback[:, :3]
            left_hand_roll_link_rot_error = axis_angle_from_quat(
                quat_from_matrix(
                    matrix_from_quat(left_hand_roll_link_setpoint[:, 3:])
                    * matrix_from_quat(quat_inv(left_hand_roll_link_feedback[:, 3:]))
                )
            )

            right_hand_roll_link_feedback = right_hand_roll_link_pose_obs
            right_hand_roll_link_setpoint = (
                torch.tensor(right_hand_roll_link_pose, device=device).unsqueeze(0).repeat(env.num_envs, 1)
            )
            right_hand_roll_link_pos_error = right_hand_roll_link_setpoint[:, :3] - right_hand_roll_link_feedback[:, :3]
            right_hand_roll_link_rot_error = axis_angle_from_quat(
                quat_from_matrix(
                    matrix_from_quat(right_hand_roll_link_setpoint[:, 3:])
                    * matrix_from_quat(quat_inv(right_hand_roll_link_feedback[:, 3:]))
                )
            )

            if num_runs % pink_ik_test_config["num_steps_controller_convergence"] == 0:
                # Check if the left hand roll link is at the target position
                torch.testing.assert_close(
                    torch.mean(torch.abs(left_hand_roll_link_pos_error), dim=1),
                    torch.zeros(env.num_envs, device="cuda:0"),
                    rtol=0.0,
                    atol=pink_ik_test_config["pos_tolerance"],
                )

                # Check if the right hand roll link is at the target position
                torch.testing.assert_close(
                    torch.mean(torch.abs(right_hand_roll_link_pos_error), dim=1),
                    torch.zeros(env.num_envs, device="cuda:0"),
                    rtol=0.0,
                    atol=pink_ik_test_config["pos_tolerance"],
                )

                # Check if the left hand roll link is at the target orientation
                torch.testing.assert_close(
                    torch.mean(torch.abs(left_hand_roll_link_rot_error), dim=1),
                    torch.zeros(env.num_envs, device="cuda:0"),
                    rtol=0.0,
                    atol=pink_ik_test_config["rot_tolerance"],
                )

                # Check if the right hand roll link is at the target orientation
                torch.testing.assert_close(
                    torch.mean(torch.abs(right_hand_roll_link_rot_error), dim=1),
                    torch.zeros(env.num_envs, device="cuda:0"),
                    rtol=0.0,
                    atol=pink_ik_test_config["rot_tolerance"],
                )

                # Change the setpoints to move the hands up and down as per the counter
                test_counter += 1
                if move_hands_up and test_counter > pink_ik_test_config["num_times_to_move_hands_up"]:
                    move_hands_up = False
                elif not move_hands_up and test_counter > (
                    pink_ik_test_config["num_times_to_move_hands_down"]
                    + pink_ik_test_config["num_times_to_move_hands_up"]
                ):
                    # Test is done after moving the hands up and down
                    break
                if move_hands_up:
                    left_hand_roll_link_pose[1] += 0.05
                    left_hand_roll_link_pose[2] += 0.05
                    right_hand_roll_link_pose[1] += 0.05
                    right_hand_roll_link_pose[2] += 0.05
                else:
                    left_hand_roll_link_pose[1] -= 0.05
                    left_hand_roll_link_pose[2] -= 0.05
                    right_hand_roll_link_pose[1] -= 0.05
                    right_hand_roll_link_pose[2] -= 0.05

    env.close()
