from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers.action_manager import ActionTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from .actions_cfg import GaitPhaseActionCfg


class GaitPhaseAction(ActionTerm):
    """Action term to control gait phase dynamics by period only.

    Maps input action in [-1, 1] to period in [min_period_s, max_period_s].
    """

    cfg: GaitPhaseActionCfg

    def __init__(self, cfg: GaitPhaseActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        # single-dim raw/processed actions
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        # ranges
        self._min_period = torch.tensor(cfg.min_period_s, device=self.device)
        self._max_period = torch.tensor(cfg.max_period_s, device=self.device)

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def get_default_action(self) -> torch.Tensor:
        # neutral: zero
        return torch.zeros(self.num_envs, 1, device=self.device)

    def process_actions(self, actions: torch.Tensor):
        # store and clamp to [-1, 1]
        self._raw_actions[:] = actions
        self._processed_actions = torch.clamp(self._raw_actions, -1.0, 1.0)
        # processed value in [-1, 1]; mapping applied in apply_actions

    def apply_actions(self):
        # write period into environment buffers
        scale = (self._max_period - self._min_period) * 0.5
        center = (self._max_period + self._min_period) * 0.5
        period = center + scale * self._processed_actions.view(-1)
        period = torch.clamp(period, min=self._min_period.item(), max=self._max_period.item())
        self._env._phase_period_s[:] = period