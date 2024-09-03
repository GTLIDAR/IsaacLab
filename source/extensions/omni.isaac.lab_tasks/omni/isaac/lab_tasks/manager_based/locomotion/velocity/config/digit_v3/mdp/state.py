import torch
from omni.isaac.lab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from omni.isaac.lab.managers import SceneEntityCfg


# ground reaction forces
def ground_reaction_forces(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Ground reaction forces."""
    # extract the used quantities (to enable type-hinting)
    robot = env.get_asset("robot")
    return robot.get_ground_reaction_forces().to(env.device)


# firctions
def friction(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Friction."""
    # extract the used quantities (to enable type-hinting)
    robot = env.get_asset("robot")
    return robot.get_friction().to(env.device)


# accelerations
def accelerations(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Accelerations."""
    # extract the used quantities (to enable type-hinting)
    robot = env.get_asset("robot")
    return robot.get_accelerations().to(env.device)
