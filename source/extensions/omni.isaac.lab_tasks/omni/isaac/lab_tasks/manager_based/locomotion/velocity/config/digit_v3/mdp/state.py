import torch
from omni.isaac.lab.assets import articulation
from omni.isaac.lab.assets.articulation.articulation import Articulation
from omni.isaac.lab.envs import ManagerBasedRLEnv
from omni.isaac.lab.managers import SceneEntityCfg


# applied torque
def applied_torque(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Applied torque."""
    robot = env.scene[asset_cfg.name]
    robot: Articulation
    return robot.data.applied_torque.to(env.device)


# stiffness and damping
def stiffness_and_damping(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Joint stiffness and damping."""
    robot = env.scene[asset_cfg.name]
    robot: Articulation
    return torch.cat([robot.data.joint_stiffness, robot.data.joint_damping], dim=-1).to(
        env.device
    )


# root state in world frame
def root_state_w(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Root state in world frame."""
    robot = env.scene[asset_cfg.name]
    robot: Articulation
    return torch.cat([robot.data.root_pos_w, robot.data.root_quat_w], dim=-1).to(
        env.device
    )


# acceleration
def acceleration(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Joint acceleration."""
    robot = env.scene[asset_cfg.name]
    robot: Articulation
    return robot.data.joint_acc.to(env.device)


# body state in world frame
def body_state_w(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Body state in world frame."""
    robot = env.scene[asset_cfg.name]
    robot: Articulation
    return robot.data.body_pos_w.flatten(start_dim=1).to(
        env.device
    )  # first dim is n_envs
