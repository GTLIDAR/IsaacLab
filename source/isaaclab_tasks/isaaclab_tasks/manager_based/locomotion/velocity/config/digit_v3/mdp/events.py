from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal


from isaaclab.actuators import ImplicitActuator
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs.mdp.events import _randomize_prop_by_op  # type: ignore

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def randomize_actuator_gains(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    stiffness_distribution_params: tuple[float, float] | None = None,
    damping_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize the actuator gains in an articulation by adding, scaling, or setting random values.

    This function allows randomizing the actuator stiffness and damping gains.

    The function samples random values from the given distribution parameters and applies the operation to the joint properties.
    It then sets the values into the actuator models. If the distribution parameters are not provided for a particular property,
    the function does not modify the property.

    .. tip::
        For implicit actuators, this function uses CPU tensors to assign the actuator gains into the simulation.
        In such cases, it is recommended to use this function only during the initialization of the environment.

    Raises:
        NotImplementedError: If the joint indices are in explicit motor mode. This operation is currently
            not supported for explicit actuator models.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # resolve joint indices
    if asset_cfg.joint_ids == slice(None):
        joint_ids_list = range(asset.num_joints)
        joint_ids = slice(None)  # for optimization purposes
    else:
        joint_ids_list = asset_cfg.joint_ids
        joint_ids = torch.tensor(
            asset_cfg.joint_ids, dtype=torch.int, device=asset.device
        )

    # check if none of the joint indices are in explicit motor mode
    for joint_index in joint_ids_list:  # type: ignore
        for act_name, actuator in asset.actuators.items():
            # if joint indices are a slice (i.e., all joints are captured) or the joint index is in the actuator
            if (
                actuator.joint_indices == slice(None)
                or joint_index in actuator.joint_indices  # type: ignore
            ):
                if not isinstance(actuator, ImplicitActuator):
                    raise NotImplementedError(
                        "Event term 'randomize_actuator_stiffness_and_damping' is performed on asset"
                        f" '{asset_cfg.name}' on the joint '{asset.joint_names[joint_index]}' ('{joint_index}') which"
                        f" uses an explicit actuator model '{act_name}<{actuator.__class__.__name__}>'. This operation"
                        " is currently not supported for explicit actuator models."
                    )

    # sample joint properties from the given ranges and set into the physics simulation
    # -- stiffness
    if stiffness_distribution_params is not None:
        stiffness = asset.data.default_joint_stiffness.to(asset.device).clone()
        stiffness = _randomize_prop_by_op(
            stiffness,
            stiffness_distribution_params,
            env_ids,
            joint_ids,
            operation=operation,
            distribution=distribution,
        )[env_ids][:, joint_ids]
        asset.write_joint_stiffness_to_sim(
            stiffness,
            joint_ids=joint_ids,
            env_ids=env_ids,  # type: ignore
        )
    # -- damping
    if damping_distribution_params is not None:
        damping = asset.data.default_joint_damping.to(asset.device).clone()
        damping = _randomize_prop_by_op(
            damping,
            damping_distribution_params,
            env_ids,
            joint_ids,
            operation=operation,
            distribution=distribution,
        )[env_ids][:, joint_ids]
        asset.write_joint_damping_to_sim(damping, joint_ids=joint_ids, env_ids=env_ids)  # type: ignore
