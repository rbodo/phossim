from dataclasses import dataclass

import cv2
import gym
import numpy as np

from phossim.config import AbstractConfig


@dataclass
class HumanAgentConfig(AbstractConfig):
    action_map: dict
    default_action: int = 0


class HumanAgent:
    def __init__(self, environment: gym.Env, config: HumanAgentConfig):
        self.environment = environment
        self.action_map = config.action_map
        self.default_action = config.default_action
        assert len(self.action_map) <= self.environment.action_space.n

    # noinspection PyUnusedLocal
    def predict(self, observation: np.ndarray) -> int:
        key = cv2.waitKey(0)
        if key != -1:
            return self.action_map.get(chr(key), self.default_action)
        return self.default_action
