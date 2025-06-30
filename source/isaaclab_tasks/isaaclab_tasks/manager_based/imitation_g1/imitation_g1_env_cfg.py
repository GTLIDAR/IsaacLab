# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.envs.mdp import (
    JointPositionActionCfg,
    JointPositionToLimitsActionCfg,
    reset_joints_by_offset,
    time_out,
)
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

from . import mdp

##
# Pre-defined configs
##
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip

##
# Pre-defined configs
##

from isaaclab_assets.robots.unitree import G1_CFG

##
# Scene definition
##


@configclass
class ImitationG1SceneCfg(InteractiveSceneCfg):
    """Configuration for a G1 robot scene."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    # robot
    robot: ArticulationCfg = G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")  # type: ignore

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True
    )


##
# MDP settings
##

# --- Borrowed and extended from velocity G1 task ---
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.rough_env_cfg import (
    G1Rewards,
    G1ObservationsCfg,
    G1ActionsCfg,
)
from isaaclab.managers import (
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
    ObservationTermCfg as ObsTerm,
    ObservationGroupCfg as ObsGroup,
)
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from .mdp import qpos_imitation_l2


# --- Rewards ---
@configclass
class RewardsCfg(G1Rewards):
    # Borrow all velocity task rewards, then add imitation-specific ones
    qpos_imitation_l2 = RewTerm(
        func=qpos_imitation_l2, weight=5.0
    )  # New imitation reward
    # You can adjust the weight as needed


# --- Observations ---
@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(G1ObservationsCfg.PolicyCfg):
        # Borrow all velocity task observation terms
        # Optionally add imitation-specific terms here
        pass

    policy: PolicyCfg = PolicyCfg()
    student: PolicyCfg = PolicyCfg()
    teacher: PolicyCfg = PolicyCfg()


# --- Actions ---
@configclass
class ActionsCfg(G1ActionsCfg):
    pass  # Directly borrow from velocity task


# --- Events ---
@configclass
class EventCfg:
    # Borrow reset event structure from velocity task, but keep only relevant ones for imitation
    reset_robot_joints = EventTerm(
        func=reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "position_range": (-1.0, 1.0),
            "velocity_range": (-0.5, 0.5),
        },
    )
    # Add more events as needed


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=time_out, time_out=True)


##
# Environment configuration
##


@configclass
class ImitationG1EnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: ImitationG1SceneCfg = ImitationG1SceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    # Dataset settings
    dataset_type: str = "zarr"
    # Dataset and cache settings for ImitationRLEnv
    dataset_path: str = "/tmp/iltools_zarr"
    window_size: int = 64  # Window size for per-env cache
    batch_size: int = 1  # Batch size for Zarr prefetching
    device: str = "cuda"  # Torch device
    loader_type: str = "loco_mujoco"  # Loader type (required if Zarr does not exist)
    loader_kwargs: dict = {
        "env_name": "UnitreeG1",
        "task": "walk",
    }  # Loader kwargs (required if Zarr does not exist)
    replay_reference: bool = True
    # Reference joint names for the robot from the reference qpos order (this is the order of G1 in loco-mujoco)
    reference_joint_names: list[str] = [
        "root_z",
        "root_qw",
        "root_qx",
        "root_qy",
        "root_qz",
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "torso_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_pitch_joint",
        "left_elbow_roll_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_pitch_joint",
        "right_elbow_roll_joint",
    ]

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False
