from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.noise import UniformNoiseCfg as Unoise
from isaaclab.utils.noise import GaussianNoiseCfg as Gnoise
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
import isaaclab_tasks.manager_based.locomotion.velocity.config.digit_v3.mdp as digit_mdp


@configclass
class TeacherObsCfg(ObsGroup):
    """Observations for policy group."""

    # observation terms (order preserved)
    clock = ObsTerm(
        func=digit_mdp.clock,
    )
    base_lin_vel = ObsTerm(func=mdp.base_lin_vel)

    base_ang_vel = ObsTerm(func=mdp.base_ang_vel)

    projected_gravity = ObsTerm(func=mdp.projected_gravity)

    velocity_commands = ObsTerm(
        func=mdp.generated_commands,
        params={"command_name": "base_velocity"},
    )

    joint_pos = ObsTerm(func=mdp.joint_pos)

    joint_vel = ObsTerm(func=mdp.joint_vel)

    actions = ObsTerm(func=mdp.last_action)

    height_scan = ObsTerm(
        func=mdp.height_scan,
        params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        clip=(-1.0, 1.0),
    )
    
    foot_scan_left = ObsTerm(
        func=mdp.height_scan,
        params={"sensor_cfg": SceneEntityCfg("foot_scanner_left"), "offset": 0.0},
        clip=(-1.0, 1.0),
    )
    
    foot_scan_right = ObsTerm(
        func=mdp.height_scan,
        params={"sensor_cfg": SceneEntityCfg("foot_scanner_right"), "offset": 0.0},
        clip=(-1.0, 1.0),
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class StudentObsCfg(ObsGroup):
    """Observations for student group."""

    # observation terms (order preserved)
    clock = ObsTerm(
        func=digit_mdp.clock,
    )
    base_lin_vel = ObsTerm(
        func=mdp.base_lin_vel,
        scale=1,
        noise=Gnoise(mean=0.0, std=0.15),
    )
    base_ang_vel = ObsTerm(
        func=mdp.base_ang_vel,
        scale=1,
        noise=Gnoise(mean=0.0, std=0.15),
    )
    projected_gravity = ObsTerm(
        func=mdp.projected_gravity,
        noise=Gnoise(mean=0.0, std=0.075),
    )
    velocity_commands = ObsTerm(
        func=mdp.generated_commands,
        scale=1,
        params={"command_name": "base_velocity"},
    )
    joint_pos = ObsTerm(
        func=mdp.joint_pos,
        scale=1,
        noise=Gnoise(mean=0.0, std=0.175),
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "left_hip_roll",
                    "left_hip_yaw",
                    "left_hip_pitch",
                    "left_knee",
                    "left_toe_A",
                    "left_toe_B",
                    "right_hip_roll",
                    "right_hip_yaw",
                    "right_hip_pitch",
                    "right_knee",
                    "right_toe_A",
                    "right_toe_B",
                    "left_shoulder_roll",
                    "left_shoulder_pitch",
                    "left_shoulder_yaw",
                    "left_elbow",
                    "right_shoulder_roll",
                    "right_shoulder_pitch",
                    "right_shoulder_yaw",
                    "right_elbow",
                    "left_shin",
                    "left_tarsus",
                    "left_toe_pitch",
                    "left_toe_roll",
                    "left_heel_spring",
                    "right_shin",
                    "right_tarsus",
                    "right_toe_pitch",
                    "right_toe_roll",
                    "right_heel_spring",
                ],
                preserve_order=True,
            )
        },
    )

    joint_vel = ObsTerm(
        func=mdp.joint_vel,
        scale=1,
        noise=Gnoise(mean=0.0, std=0.175),
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "left_hip_roll",
                    "left_hip_yaw",
                    "left_hip_pitch",
                    "left_knee",
                    "left_toe_A",
                    "left_toe_B",
                    "right_hip_roll",
                    "right_hip_yaw",
                    "right_hip_pitch",
                    "right_knee",
                    "right_toe_A",
                    "right_toe_B",
                    "left_shoulder_roll",
                    "left_shoulder_pitch",
                    "left_shoulder_yaw",
                    "left_elbow",
                    "right_shoulder_roll",
                    "right_shoulder_pitch",
                    "right_shoulder_yaw",
                    "right_elbow",
                    "left_shin",
                    "left_tarsus",
                    "left_toe_pitch",
                    "left_toe_roll",
                    "left_heel_spring",
                    "right_shin",
                    "right_tarsus",
                    "right_toe_pitch",
                    "right_toe_roll",
                    "right_heel_spring",
                ],
                preserve_order=True,
            )
        },
    )
    actions = ObsTerm(func=mdp.last_action)

    height_scan = ObsTerm(
        func=mdp.height_scan,
        params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        clip=(-1.0, 1.0),
        noise=Gnoise(mean=0.0, std=0.175),
    )

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True
