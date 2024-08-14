from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.utils.noise import AdditiveGaussianNoiseCfg as Gnoise
from omni.isaac.lab.utils.noise import UniformNoiseCfg as Unoise
from omni.isaac.lab.utils import configclass

import omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp as mdp


@configclass
class TeacherObsCfg(ObsGroup):
    """Observations for teacher group."""

    # observation terms (order preserved)
    base_lin_vel = ObsTerm(
        func=mdp.base_lin_vel,  # type: ignore
        scale=1,
        noise=Gnoise(mean=0.0, std=0.05, operation="add"),
    )
    base_ang_vel = ObsTerm(
        func=mdp.base_ang_vel,  # type: ignore
        scale=1,
        noise=Gnoise(mean=0.0, std=0.05, operation="add"),
    )
    velocity_commands = ObsTerm(
        func=mdp.generated_commands,  # type: ignore
        scale=1,
        params={"command_name": "base_velocity"},
    )
    joint_pos = ObsTerm(
        func=mdp.joint_pos,  # type: ignore
        scale=1,
        noise=Gnoise(mean=0.0, std=0.05, operation="add"),
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
        func=mdp.joint_vel,  # type: ignore
        scale=1,
        noise=Gnoise(mean=0.0, std=0.05, operation="add"),
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
    actions = ObsTerm(func=mdp.last_action)  # type: ignore

    # # kp and kd. randomization scheme should be assigned from the event manager
    # pd_gain = ObsTerm(
    #     func=mdp.pd_gain,
    #     params={
    #         "asset_cfg": SceneEntityCfg(
    #             "robot",
    #             joint_names=[
    #                 "left_hip_roll",
    #                 "left_hip_yaw",
    #                 "left_hip_pitch",
    #                 "left_knee",
    #                 "left_toe_A",
    #                 "left_toe_B",
    #                 "right_hip_roll",
    #                 "right_hip_yaw",
    #                 "right_hip_pitch",
    #                 "right_knee",
    #                 "right_toe_A",
    #                 "right_toe_B",
    #                 "left_shoulder_roll",
    #                 "left_shoulder_pitch",
    #                 "left_shoulder_yaw",
    #                 "left_elbow",
    #                 "right_shoulder_roll",
    #                 "right_shoulder_pitch",
    #                 "right_shoulder_yaw",
    #                 "right_elbow",
    #             ],
    #             preserve_order=True,
    #         )
    #     },
    # )

    # # projected gravity
    # projected_gravity = ObsTerm(func=mdp.projected_gravity)

    # # height map
    # height_scan = ObsTerm(
    #     func=mdp.height_scan,
    #     params={"sensor_cfg": SceneEntityCfg("height_scanner")},
    #     # noise=Unoise(n_min=-0.1, n_max=0.1),
    #     clip=(-1.0, 1.0),
    # )

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True


@configclass
class StudentObsCfg(ObsGroup):
    """Observations for student group."""

    # observation terms (order preserved)
    base_lin_vel = ObsTerm(
        func=mdp.base_lin_vel, noise=Gnoise(mean=0.0, std=0.05, operation="add")  # type: ignore
    )
    base_ang_vel = ObsTerm(
        func=mdp.base_ang_vel, noise=Gnoise(mean=0.0, std=0.05, operation="add")  # type: ignore
    )
    velocity_commands = ObsTerm(
        func=mdp.generated_commands, params={"command_name": "base_velocity"}  # type: ignore
    )
    joint_pos = ObsTerm(
        func=mdp.joint_pos,  # type: ignore
        noise=Gnoise(mean=0.0, std=0.175, operation="add"),
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
        func=mdp.joint_vel,  # type: ignore
        noise=Gnoise(mean=0.0, std=0.05, operation="add"),
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
    actions = ObsTerm(func=mdp.last_action)  # type: ignore

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True
