from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, TYPE_CHECKING

import gym
import numpy as np

if TYPE_CHECKING:
    from phossim.rendering import Display


@dataclass
class HumanAgentConfig:
    action_map: dict
    default_action: int = 0


class HumanAgent:
    def __init__(self, environment: gym.Env, display: Display,
                 config: HumanAgentConfig):
        self.environment = environment
        self.display = display
        self.action_map = config.action_map
        self.default_action = config.default_action
        assert len(self.action_map) <= self.environment.action_space.n

    def predict(self, observation: np.ndarray) -> Tuple[int, None]:
        key = self.display.render(observation)
        action = self.default_action
        if key != -1 and chr(key) in self.action_map:
            action = self.action_map[chr(key)]
        return action, None
