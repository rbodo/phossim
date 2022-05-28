from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, TYPE_CHECKING

import gym
import numpy as np

if TYPE_CHECKING:
    from phossim.rendering import Viewer


@dataclass
class HumanAgentConfig:
    action_map: dict
    default_action: int = 0


class HumanAgent:
    def __init__(self, environment: gym.Env, viewer: Viewer,
                 config: HumanAgentConfig):
        self.environment = environment
        self.viewer = viewer
        self.action_map = config.action_map
        self.default_action = config.default_action
        assert len(self.action_map) <= self.environment.action_space.n

    def predict(self, observation: np.ndarray) -> Tuple[int, None]:
        self.viewer.render(observation)
        key = self.viewer.get_key()
        action = self.action_map.get(key, self.default_action)
        return action, None
