from dataclasses import dataclass
from typing import Tuple

import gym
import numpy as np

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
        if key != -1 and chr(key) in self.action_map:
            return self.action_map[chr(key)]
        return self.default_action, None
