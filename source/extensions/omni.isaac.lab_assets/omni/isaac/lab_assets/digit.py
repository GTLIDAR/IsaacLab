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

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg
from omni.isaac.lab.utils.assets import ISAACLAB_NUCLEUS_DIR

##
# Configuration
##

import os
import torch
torch.cuda.empty_cache()

full_path = os.path.dirname(os.path.realpath(__file__))


DIGITV3_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{full_path}/../../../assets/robot/digit_v3_july_3_with_value_range.usd", # d_i.usd
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.03077151),
        # joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
        joint_pos={
            ".*left_hip_roll": 3.65171317e-01,
            ".*left_hip_yaw": -6.58221569e-03,
            ".*left_hip_pitch": 3.16910843e-01,
            ".*left_knee": 3.57944829e-01,

            ".*left_shin": -0.0130148100,#
            ".*left_tarsus": -0.3311601,#
            ".*left_heel_spring": -0.01160161,#

            ".*left_toe_A": -1.32137105e-01,
            ".*left_toe_B": 1.24904386e-01,

            ".*left_toe_pitch": 0.13114588,#
            ".*left_toe_roll": -0.01159121,#

            "left_shoulder_roll": -1.50466737e-01,
            "left_shoulder_pitch": 1.09051174e+00,
            "left_shoulder_yaw": 3.43707570e-04,
            "left_elbow": -1.39091311e-01,

            ".*right_hip_roll": -3.65734576e-01,
            ".*right_hip_yaw": 6.42881761e-03,
            ".*right_hip_pitch": -3.16910843e-01,
            ".*right_knee": -3.58016735e-01,

            ".*right_shin":  0.01303884,#
            ".*right_tarsus": 0.33119604,#
            ".*right_heel_spring": 0.01160569,#

            ".*right_toe_A": 1.32006717e-01,
            ".*right_toe_B": -1.25034774e-01,

            ".*right_toe_pitch": -0.13114439,#
            ".*right_toe_roll": 0.01117851,#

            "right_shoulder_roll": 1.50437975e-01,
            "right_shoulder_pitch": -1.09045901e+00,
            "right_shoulder_yaw": -3.51377474e-04,
            "right_elbow": 1.39086517e-01,
        },
        # gear_ratio={".*": 1.0}
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*left_hip_roll",
                ".*hip_yaw",
                ".*hip_pitch",
                ".*knee",
            ],
            effort_limit=200.0,
            velocity_limit=10.0,
            stiffness={
                ".*hip_roll":800,
                ".*hip_yaw":600,
                ".*hip_pitch":800,
                ".*knee":1000,
            },
            damping={
                ".*hip_roll": 8.0,
                ".*hip_yaw": 8.0,
                ".*hip_pitch": 8.0,
                ".*knee": 5.0,
            },
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*shoulder_roll",
                ".*shoulder_pitch",
                ".*shoulder_yaw",
                ".*elbow",
            ],
            effort_limit=200.0,
            velocity_limit=10.0,
            stiffness={
                ".*shoulder_roll": 100,
                ".*shoulder_pitch": 100,
                ".*shoulder_yaw": 100,
                ".*elbow": 100,
            },
            damping={
                ".*shoulder_roll": 5.0,
                ".*shoulder_pitch": 5.0,
                ".*shoulder_yaw": 5.0,
                ".*elbow": 5.0,
            },
        ),
        "toes": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*toe_A",
                ".*toe_B",
            ],
            effort_limit=20.0,
            velocity_limit=10.0,
            stiffness={
                ".*left_toe_A":20,
                ".*left_toe_B":20,
            },
            damping={
                ".*left_toe_A": 1.0,
                ".*left_toe_B": 1.0,
            },
        ),
    },
)
