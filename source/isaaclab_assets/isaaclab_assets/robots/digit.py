# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Digit implementation made by LIDAR Gatech

"""Configuration for Agility robots.
The following configurations are available:
* :obj:`DIGITV4_CFG`: Agility Cassie robot with simple PD controller for the legs
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

# from omni.isaac.lab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR

##
# Configuration
##

import os
import torch

torch.cuda.empty_cache()

full_path = os.path.dirname(os.path.realpath(__file__))


DIGITV3_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robot/Agility/digit/digit_v3_oct28.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=True,
            linear_damping= 1e6,
            angular_damping= 1e6,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.03077151),
        joint_vel={".*": 0.0},
        joint_pos={
            "left_hip_roll": 3.65171317e-01,
            "left_hip_yaw": -6.58221569e-03,
            "left_hip_pitch": 3.16910843e-01,
            "left_knee": 3.57944829e-01,
            "left_tarsus": -0.3311601,
            "left_heel_spring": -0.01160161,
            "left_toe_A": -1.32137105e-01,
            "left_toe_B": 1.24904386e-01,
            "left_toe_pitch": 0.13114588,
            "left_toe_roll": -0.01159121,
            "left_shoulder_roll": -1.50466737e-01,
            "left_shoulder_pitch": 1.09051174e00,
            "left_shoulder_yaw": 3.43707570e-04,
            "left_elbow": -1.39091311e-01,
            "right_hip_roll": -3.65734576e-01,
            "right_hip_yaw": 6.42881761e-03,
            "right_hip_pitch": -3.16910843e-01,
            "right_knee": -3.58016735e-01,
            "right_tarsus": 0.33119604,
            "right_heel_spring": 0.01160569,
            "right_toe_A": 1.32006717e-01,
            "right_toe_B": -1.25034774e-01,
            "right_toe_pitch": -0.13114439,
            "right_toe_roll": 0.01117851,
            "right_shoulder_roll": 1.50437975e-01,
            "right_shoulder_pitch": -1.09045901e00,
            "right_shoulder_yaw": -3.51377474e-04,
            "right_elbow": 1.39086517e-01,
        },
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[
                "left_hip_roll",
                "left_hip_yaw",
                "left_hip_pitch",
                "left_knee",
                "right_hip_roll",
                "right_hip_yaw",
                "right_hip_pitch",
                "right_knee",
            ],
            effort_limit_sim={
                "left_hip_roll": 126,
                "left_hip_yaw": 79,
                "left_hip_pitch": 216,
                "left_knee": 231,
                "right_hip_roll": 126,
                "right_hip_yaw": 79,
                "right_hip_pitch": 216,
                "right_knee": 231,
            },
            velocity_limit_sim=60.0,
            stiffness={
                "left_hip_roll": 100,
                "left_hip_yaw": 100,
                "left_hip_pitch": 200,
                "left_knee": 200,
                "right_hip_roll": 100,
                "right_hip_yaw": 100,
                "right_hip_pitch": 200,
                "right_knee": 200,
            },
            damping={
                "left_hip_roll": 5.0 + 66.849,
                "left_hip_yaw": 5.0 + 26.1129,
                "left_hip_pitch": 5.0 + 38.05,
                "left_knee": 5.0 + 38.05,
                "right_hip_roll": 5.0 + 66.849,
                "right_hip_yaw": 5.0 + 26.1129,
                "right_hip_pitch": 5.0 + 38.05,
                "right_knee": 5.0 + 38.05,
            },
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                "left_shoulder_roll",
                "left_shoulder_pitch",
                "left_shoulder_yaw",
                "left_elbow",
                "right_shoulder_roll",
                "right_shoulder_pitch",
                "right_shoulder_yaw",
                "right_elbow",
            ],
            effort_limit_sim={
                "left_shoulder_roll": 126,
                "left_shoulder_pitch": 126,
                "left_shoulder_yaw": 79,
                "left_elbow": 126,
                "right_shoulder_roll": 126,
                "right_shoulder_pitch": 126,
                "right_shoulder_yaw": 79,
                "right_elbow": 126,
            },
            velocity_limit_sim=60.0,
            stiffness={
                "left_shoulder_roll": 150,
                "left_shoulder_pitch": 150,
                "left_shoulder_yaw": 100,
                "left_elbow": 100,
                "right_shoulder_roll": 150,
                "right_shoulder_pitch": 150,
                "right_shoulder_yaw": 100,
                "right_elbow": 100,
            },
            damping={
                "left_shoulder_roll": 5.0 + 66.849,
                "left_shoulder_pitch": 5.0 + 66.849,
                "left_shoulder_yaw": 5.0 + 26.1129,
                "left_elbow": 5.0 + 66.849,
                "right_shoulder_roll": 5.0 + 66.849,
                "right_shoulder_pitch": 5.0 + 66.849,
                "right_shoulder_yaw": 5.0 + 26.1129,
                "right_elbow": 5.0 + 66.849,
            },
        ),
        "toes": ImplicitActuatorCfg(
            joint_names_expr=[
                "left_toe_A",
                "left_toe_B",
                "right_toe_A",
                "right_toe_B",
            ],
            effort_limit_sim={
                "left_toe_A": 41,
                "left_toe_B": 41,
                "right_toe_A": 41,
                "right_toe_B": 41,
            },
            velocity_limit_sim=60.0,
            stiffness={
                "left_toe_A": 20,
                "left_toe_B": 20,
                "right_toe_A": 20,
                "right_toe_B": 20,
            },
            damping={
                "left_toe_A": 1.0 + 15.5532,
                "left_toe_B": 1.0 + 15.5532,
                "right_toe_A": 1.0 + 15.5532,
                "right_toe_B": 1.0 + 15.5532,
            },
        ),
        "passive": ImplicitActuatorCfg(
            joint_names_expr=[
                "left_shin",
                "left_tarsus",
                "left_heel_spring",
                "left_toe_pitch",
                "left_toe_roll",
                "right_shin",
                "right_tarsus",
                "right_heel_spring",
                "right_toe_pitch",
                "right_toe_roll",
            ],
            stiffness={
                "left_shin": 6000,
                "left_tarsus": 0,
                "left_heel_spring": 4375.0,
                "left_toe_pitch": 0.0,
                "left_toe_roll": 0.0,
                "right_shin": 6000,
                "right_tarsus": 0,
                "right_heel_spring": 4375.0,
                "right_toe_pitch": 0.0,
                "right_toe_roll": 0.0,
            },
            damping={
                "left_shin": 0.0,
                "left_tarsus": 0.0,
                "left_heel_spring": 0.0,
                "left_toe_pitch": 0.0,
                "left_toe_roll": 0.0,
                "right_shin": 0.0,
                "right_tarsus": 0,
                "right_heel_spring": 0.0,
                "right_toe_pitch": 0.0,
                "right_toe_roll": 0.0,
            },
            
        ),
    },
)
