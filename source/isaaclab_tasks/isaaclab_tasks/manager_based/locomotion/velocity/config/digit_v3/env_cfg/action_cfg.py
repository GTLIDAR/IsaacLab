from isaaclab.utils import configclass
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp


@configclass
class DigitV3ActionCfg:
    """Action terms for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(  # type: ignore
        asset_name="robot",
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
        ],
        scale={
            "left_hip_roll": 0.9425 * 0.9,
            "left_hip_yaw": 0.6283 * 0.9,
            "left_hip_pitch": 0.9163 * 0.9,
            "left_knee": 1.1336 * 0.9,
            "left_toe_A": 0.8738 * 0.9,
            "left_toe_B": 0.8039 * 0.9,
            "right_hip_roll": 0.9425 * 0.9,
            "right_hip_yaw": 0.6283 * 0.9,
            "right_hip_pitch": 1.4399 * 0.9,
            "right_knee": 0.7671 * 0.9,
            "right_toe_A": 0.8738 * 0.9,
            "right_toe_B": 0.8039 * 0.9,
            "left_shoulder_roll": 1.1781 * 0.9,
            "left_shoulder_pitch": 2.2777 * 0.9,
            "left_shoulder_yaw": 1.5708 * 0.9,
            "left_elbow": 1.2174 * 0.9,
            "right_shoulder_roll": 1.1781 * 0.9,
            "right_shoulder_pitch": 2.2777 * 0.9,
            "right_shoulder_yaw": 1.5708 * 0.9,
            "right_elbow": 1.2174 * 0.9,
        },
        use_default_offset=True,
        preserve_order=True,
    )
    
    gait_phase = mdp.GaitPhaseActionCfg(  # type: ignore
        asset_name="robot",
        class_type=mdp.GaitPhaseAction,  # type: ignore
        min_period_s=0.515,
        max_period_s=0.765,
    )
    