from .event_cfg import DigitV3EventCfg
from .observation_cfg import TeacherObsCfg, StudentObsCfg
from .rewards_cfg import DigitV3RewardsCfg
from .terminations_cfg import DigitV3TerminationsCfg
from .command_cfg import DigitV3CommandsCfg
from .action_cfg import DigitV3ActionCfg

__all__ = [
    "TeacherObsCfg",
    "StudentObsCfg",
    "DigitV3RewardsCfg",
    "DigitV3ActionCfg",
    "DigitV3EventCfg",
    "DigitV3CommandsCfg",
    "DigitV3TerminationsCfg",
]
