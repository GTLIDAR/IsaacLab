from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.imitation.imitation_env_cfg import ImitationLearningEnvCfg
from isaaclab_tasks.manager_based.imitation.mdp import (
    bad_anchor_ori,
    bad_anchor_pos_z_only,
    bad_reference_body_pos_z_only,
    reference_anchor_ori_b,
    reference_anchor_pos_b,
    reference_global_anchor_orientation_error_exp,
    reference_global_anchor_position_error_exp,
    reference_global_body_angular_velocity_error_exp,
    reference_global_body_linear_velocity_error_exp,
    reference_motion_command,
    reference_relative_body_orientation_error_exp,
    reference_relative_body_position_error_exp,
    reset_joints_to_reference,
    robot_body_ori_b,
    robot_body_pos_b,
    track_joint_pos,
)

from isaaclab_assets.robots.unitree import G1_MINIMAL_CFG

VELOCITY_RANGE = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.2, 0.2),
    "roll": (-0.52, 0.52),
    "pitch": (-0.52, 0.52),
    "yaw": (-0.78, 0.78),
}

G1_IMITATION_JOINT_NAMES: list[str] = [
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

# Tracked body names in IsaacLab articulation naming.
G1_WBT_TRACKED_ASSET_BODY_NAMES: list[str] = [
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_pitch_link",
    # "left_palm_link",
    "right_shoulder_roll_link",
    "right_elbow_pitch_link",
    # "right_palm_link",
]

# Matching tracked body names in loco-mujoco reference metadata order.
G1_WBT_TRACKED_REFERENCE_BODY_NAMES: list[str] = [
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    # "left_wrist_roll_rubber_hand",
    "right_shoulder_roll_link",
    "right_elbow_link",
    # "right_wrist_roll_rubber_hand",
]

G1_EE_ASSET_BODY_NAMES: list[str] = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    # "left_wrist_yaw_link",
    # "right_wrist_yaw_link",
]

G1_EE_REFERENCE_BODY_NAMES: list[str] = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    # "left_wrist_roll_rubber_hand",
    # "right_wrist_roll_rubber_hand",
]

# Backward-compatible alias used by tooling/scripts that import the old name.
G1_WBT_TRACKED_BODY_NAMES: list[str] = G1_WBT_TRACKED_ASSET_BODY_NAMES

G1_WBT_UNDESIRED_CONTACT_PATTERN = "^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$).+$"

# Observation keys used by rlopt configs (flattened by IsaacLab wrapper).
G1_POLICY_OBS_KEYS: list[str] = ["policy"]
G1_VALUE_OBS_KEYS: list[str] = ["critic"]
G1_REWARD_OBS_KEYS: list[str] = ["critic"]


@configclass
class G1ObservationCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Policy observations aligned with tracking_env_cfg (reference-driven)."""

        reference_motion = ObsTerm(
            func=reference_motion_command,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=G1_IMITATION_JOINT_NAMES,
                )
            },
        )
        reference_anchor_ori_b = ObsTerm(
            func=reference_anchor_ori_b,
            params={"asset_cfg": SceneEntityCfg("robot"), "anchor_body_name": "torso_link"},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5))
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Privileged critic observations aligned with tracking_env_cfg."""

        reference_motion = ObsTerm(
            func=reference_motion_command,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=G1_IMITATION_JOINT_NAMES,
                )
            },
        )
        reference_anchor_pos_b = ObsTerm(
            func=reference_anchor_pos_b,
            params={"asset_cfg": SceneEntityCfg("robot"), "anchor_body_name": "torso_link"},
        )
        reference_anchor_ori_b = ObsTerm(
            func=reference_anchor_ori_b,
            params={"asset_cfg": SceneEntityCfg("robot"), "anchor_body_name": "torso_link"},
        )
        body_pos = ObsTerm(
            func=robot_body_pos_b,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=G1_WBT_TRACKED_ASSET_BODY_NAMES),
                "anchor_body_name": "torso_link",
            },
        )
        body_ori = ObsTerm(
            func=robot_body_ori_b,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=G1_WBT_TRACKED_ASSET_BODY_NAMES),
                "anchor_body_name": "torso_link",
            },
        )
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class G1EventCfg:
    """Tracking-style randomization/events with imitation reset hooks."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.6),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
        },
    )

    # reset_robot_joints = EventTerm(
    #     func=mdp.reset_joints_by_offset,
    #     mode="reset",
    #     params={
    #         "position_range": (-0.2, 0.2),
    #         "velocity_range": (-0.1, 0.1),
    #     },
    # )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "com_range": {"x": (-0.025, 0.025), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "yaw": (-0.2, 0.2)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints_to_reference = EventTerm(
        func=reset_joints_to_reference,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(1.0, 3.0),
        params={"velocity_range": VELOCITY_RANGE},
    )


@configclass
class G1RewardsCfg:
    """Reward terms aligned to unitree tracking_env_cfg."""

    # -- base
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    joint_torque = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1.0e-1)
    joint_limit = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )

    # -- tracking
    motion_global_anchor_pos = RewTerm(
        func=reference_global_anchor_position_error_exp,
        weight=0.5,
        params={"asset_cfg": SceneEntityCfg("robot"), "anchor_body_name": "torso_link", "std": 0.6},
    )
    motion_global_anchor_ori = RewTerm(
        func=reference_global_anchor_orientation_error_exp,
        weight=0.5,
        params={"asset_cfg": SceneEntityCfg("robot"), "anchor_body_name": "torso_link", "std": 1.0},
    )
    motion_body_pos = RewTerm(
        func=reference_relative_body_position_error_exp,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=G1_WBT_TRACKED_ASSET_BODY_NAMES),
            "reference_body_names": G1_WBT_TRACKED_REFERENCE_BODY_NAMES,
            "std": 1.0,
        },
    )
    motion_body_ori = RewTerm(
        func=reference_relative_body_orientation_error_exp,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=G1_WBT_TRACKED_ASSET_BODY_NAMES),
            "reference_body_names": G1_WBT_TRACKED_REFERENCE_BODY_NAMES,
            "std": 0.6,
        },
    )
    motion_body_lin_vel = RewTerm(
        func=reference_global_body_linear_velocity_error_exp,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=G1_WBT_TRACKED_ASSET_BODY_NAMES),
            "reference_body_names": G1_WBT_TRACKED_REFERENCE_BODY_NAMES,
            "std": 1.0,
        },
    )
    motion_body_ang_vel = RewTerm(
        func=reference_global_body_angular_velocity_error_exp,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=G1_WBT_TRACKED_ASSET_BODY_NAMES),
            "reference_body_names": G1_WBT_TRACKED_REFERENCE_BODY_NAMES,
            "std": 1.0,
        },
    )

    joint_tracking = RewTerm(
        func=track_joint_pos,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "left_hip_pitch_joint",
                    "right_hip_pitch_joint",
                    "torso_joint",
                    "left_hip_roll_joint",
                    "right_hip_roll_joint",
                    "left_shoulder_pitch_joint",
                    "right_shoulder_pitch_joint",
                    "left_hip_yaw_joint",
                    "right_hip_yaw_joint",
                    "left_shoulder_roll_joint",
                    "right_shoulder_roll_joint",
                    "left_knee_joint",
                    "right_knee_joint",
                    "left_shoulder_yaw_joint",
                    "right_shoulder_yaw_joint",
                    "left_ankle_pitch_joint",
                    "right_ankle_pitch_joint",
                    "left_elbow_pitch_joint",
                    "right_elbow_pitch_joint",
                    "left_ankle_roll_joint",
                    "right_ankle_roll_joint",
                    "left_elbow_roll_joint",
                    "right_elbow_roll_joint",
                ],
            ),
            "sigma": 0.5,
        },
    )

    # undesired_contacts = RewTerm(
    #     func=mdp.undesired_contacts,
    #     weight=-0.1,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[G1_WBT_UNDESIRED_CONTACT_PATTERN]),
    #         "threshold": 1.0,
    #     },
    # )


@configclass
class G1TerminationsCfg:
    """Termination terms aligned to unitree tracking_env_cfg."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    anchor_pos = DoneTerm(
        func=bad_anchor_pos_z_only,
        params={"asset_cfg": SceneEntityCfg("robot"), "anchor_body_name": "torso_link", "threshold": 1.0},
    )
    anchor_ori = DoneTerm(
        func=bad_anchor_ori,
        params={"asset_cfg": SceneEntityCfg("robot"), "anchor_body_name": "torso_link", "threshold": 1.5},
    )
    ee_body_pos = DoneTerm(
        func=bad_reference_body_pos_z_only,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=G1_EE_ASSET_BODY_NAMES),
            "reference_body_names": G1_EE_REFERENCE_BODY_NAMES,
            "threshold": 1.0,
        },
    )
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="torso_link"),
            "threshold": 1.0,
        },
    )
    base_too_low = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={
            "minimum_height": 0.5,
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
        },
    )


@configclass
class ImitationG1EnvCfg(ImitationLearningEnvCfg):
    observations = G1ObservationCfg()
    rewards = G1RewardsCfg()  # type: ignore
    terminations = G1TerminationsCfg()  # type: ignore
    events = G1EventCfg()

    # Dataset and cache settings for ImitationRLEnv
    # ref_dt = sim.dt * n_substeps (loco-mujoco TrajectoryHandler interpolates to this).
    # env_dt = self.sim.dt * self.decimation (Isaac control step, set in __post_init__).
    # Playback speed = ref_dt / env_dt.  1 ref step is consumed per 1 env step.
    # To slow down: lower n_substeps (smaller ref_dt -> more ref frames -> same motion takes more env steps).
    #   n_substeps=4 -> ref_dt=0.02  -> 1x    (normal)
    #   n_substeps=2 -> ref_dt=0.01  -> 0.5x  (2x slower)
    #   n_substeps=1 -> ref_dt=0.005 -> 0.25x (4x slower)
    device: str = "cuda"
    loader_type: str = "loco_mujoco"
    loader_kwargs: dict = {
        "env_name": "UnitreeG1",
        "n_substeps": 4,
        "dataset": {
            "trajectories": {
                "default": ["walk"],
                "amass": [],
                "lafan1": [],
            }
        },
        "control_freq": 50.0,
        "window_size": 4,
        "sim": {"dt": 0.005},
        "decimation": 4,
    }

    replay_reference: bool = False
    replay_only: bool = False
    refresh_zarr_dataset: bool = True
    reference_start_frame: int = 0

    visualize_reference_arrows: bool = True
    print_reference_velocity: bool = False
    print_reference_velocity_every: int = 50

    reference_joint_names: list[str] = [
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

    target_joint_names: list[str] = [
        "left_hip_pitch_joint",
        "right_hip_pitch_joint",
        "torso_joint",
        "left_hip_roll_joint",
        "right_hip_roll_joint",
        "left_shoulder_pitch_joint",
        "right_shoulder_pitch_joint",
        "left_hip_yaw_joint",
        "right_hip_yaw_joint",
        "left_shoulder_roll_joint",
        "right_shoulder_roll_joint",
        "left_knee_joint",
        "right_knee_joint",
        "left_shoulder_yaw_joint",
        "right_shoulder_yaw_joint",
        "left_ankle_pitch_joint",
        "right_ankle_pitch_joint",
        "left_elbow_pitch_joint",
        "right_elbow_pitch_joint",
        "left_ankle_roll_joint",
        "right_ankle_roll_joint",
        "left_elbow_roll_joint",
        "right_elbow_roll_joint",
        "left_five_joint",
        "left_three_joint",
        "left_zero_joint",
        "right_five_joint",
        "right_three_joint",
        "right_zero_joint",
        "left_six_joint",
        "left_four_joint",
        "left_one_joint",
        "right_six_joint",
        "right_four_joint",
        "right_one_joint",
        "left_two_joint",
        "right_two_joint",
    ]

    def __post_init__(self) -> None:
        super().__post_init__()  # type: ignore

        self.scene.robot = G1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")  # type: ignore
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
            self.scene.contact_forces.force_threshold = 10.0
            self.scene.contact_forces.debug_vis = True

        self.scene.height_scanner = None
