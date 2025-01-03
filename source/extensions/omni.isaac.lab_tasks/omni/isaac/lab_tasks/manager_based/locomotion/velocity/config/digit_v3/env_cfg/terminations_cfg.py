from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils import configclass

import omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp as mdp
import omni.isaac.lab_tasks.manager_based.locomotion.velocity.config.digit_v3.mdp as digit_mdp
from omni.isaac.lab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    TerminationsCfg,
)
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm


@configclass
class DigitV3TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)  # type: ignore
    base_contact = DoneTerm(
        func=mdp.illegal_contact,  # type: ignore
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[".*base", ".*hip.*", ".*knee", ".*elbow"],
            ),
            "threshold": 1.0,
        },
    )

    base_too_low = DoneTerm(
        func=digit_mdp.root_height_below_minimum_adaptive,  # type: ignore
        params={
            "minimum_height": 0.4,
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=[
                    ".*toe_roll.*",
                ],
            ),
        },
    )
