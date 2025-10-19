# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCfg
from isaaclab.sensors.ray_caster import patterns
from isaaclab.markers.config import RAY_CASTER_MARKER_CFG
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils


from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
)

from .env_cfg import (
    StudentObsCfg,
    TeacherObsCfg,
    DigitV3CommandsCfg,
    DigitV3TerminationsCfg,
    DigitV3RewardsCfg,
    DigitV3ActionCfg,
    DigitV3EventCfg,
)


from isaaclab_assets.robots.digit import DIGITV3_CFG  # isort: skip


@configclass
class DigitV3ObservationsCfg:
    """Observation specifications for the MDP."""

    # observation groups
    policy: TeacherObsCfg = TeacherObsCfg()


@configclass
class DigitV3RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    observations: DigitV3ObservationsCfg = DigitV3ObservationsCfg()
    actions: DigitV3ActionCfg = DigitV3ActionCfg()
    rewards: DigitV3RewardsCfg = DigitV3RewardsCfg()
    terminations: DigitV3TerminationsCfg = DigitV3TerminationsCfg()
    events: DigitV3EventCfg = DigitV3EventCfg()
    commands: DigitV3CommandsCfg = DigitV3CommandsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        
        self.enable_foot_terrain_vis = False  
        
        self.scene.env_spacing = 5.0
        self.sim.dt = 0.001
        self.decimation = 20
        self.sim.gravity = (0.0, 0.0, -9.806)
        self.sim.render_interval = self.decimation
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**26
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 2**22

        # Scene
        self.scene.robot = DIGITV3_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")  # type: ignore
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"

        # Visualizer configurations: Red for core, Blue for safe
        core_ray_marker_cfg = RAY_CASTER_MARKER_CFG.copy()
        core_ray_marker_cfg.prim_path = "/Visuals/FootRay/Core"
        core_ray_marker_cfg.markers["hit"].visual_material = sim_utils.PreviewSurfaceCfg(
            diffuse_color=(1.0, 0.0, 0.0)  # Red
        )
        
        safe_ray_marker_cfg = RAY_CASTER_MARKER_CFG.copy()
        safe_ray_marker_cfg.prim_path = "/Visuals/FootRay/Safe"
        safe_ray_marker_cfg.markers["hit"].visual_material = sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.0, 0.5, 1.0)  # Blue
        )

        self.scene.foot_scanner_left_core = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/left_toe_roll",
            update_period=self.sim.dt * self.decimation,
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.02)),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(
                resolution=0.02,
                size=[0.15, 0.15]
            ),
            max_distance=0.2,
            debug_vis=self.enable_foot_terrain_vis,
            visualizer_cfg=core_ray_marker_cfg,
            mesh_prim_paths=["/World/ground"],
        )
        
        self.scene.foot_scanner_right_core = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/right_toe_roll",
            update_period=self.sim.dt * self.decimation,
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.02)),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(
                resolution=0.02,
                size=[0.15, 0.15]
            ),
            max_distance=0.2,
            debug_vis=self.enable_foot_terrain_vis,
            visualizer_cfg=core_ray_marker_cfg,
            mesh_prim_paths=["/World/ground"],
        )

        self.scene.foot_scanner_left_safe = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/left_toe_roll",
            update_period=self.sim.dt * self.decimation,
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.02)),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(
                resolution=0.03,
                size=[0.24, 0.24]
            ),
            max_distance=0.25,
            debug_vis=self.enable_foot_terrain_vis,
            visualizer_cfg=safe_ray_marker_cfg,
            mesh_prim_paths=["/World/ground"],
        )

        self.scene.foot_scanner_right_safe = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/right_toe_roll",
            update_period=self.sim.dt * self.decimation,
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.02)),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(
                resolution=0.03,
                size=[0.24, 0.24]
            ),
            max_distance=0.25,
            debug_vis=self.enable_foot_terrain_vis,
            visualizer_cfg=safe_ray_marker_cfg,
            mesh_prim_paths=["/World/ground"],
        )

        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot",
            joint_names=[
                ".*_hip_.*",
                ".*_knee",
                ".*_toe.*",
                ".*_shoulder.*",
                ".*_elbow",
            ],
        )
        # Rewards
        self.rewards.dof_acc_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot",
            joint_names=[
                ".*_hip_.*",
                ".*_knee",
                ".*_shoulder.*",
                ".*_elbow",
            ],
        )

        self.rewards.undesired_contacts = None  # type: ignore
        # self.rewards.alive.weight = 0.0
        self.rewards.track_lin_vel_xy_exp.weight = 0.5
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        self.rewards.ang_vel_xy_l2.weight = -0.1
        self.rewards.dof_pos_limits.weight = -0.5
        self.rewards.termination_penalty.weight = -200
        self.rewards.feet_slide.weight = -1.0
        self.rewards.joint_deviation_hip.weight = -5.0
        self.rewards.flat_orientation_l2.weight = -10.0
        self.rewards.dof_torques_l2.weight = -1.0e-5
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.dof_acc_l2.weight = -1.25e-7

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (-0.3, 1.2)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.heading_command = False


@configclass
class DigitV3RoughEnvCfg_PLAY(DigitV3RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None  # type: ignore
        self.events.push_robot = None  # type: ignore

        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
