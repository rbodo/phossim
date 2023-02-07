from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import gym
import numpy as np
import cv2


@dataclass
class HumanAgentConfig:
    action_map: dict
    image_shape: Tuple[int, int, int]
    default_action: int = 0
    idp: int = 1240


class HumanAgent:
    def __init__(self, environment: gym.Env, config: HumanAgentConfig):
        self.environment = environment
        self.action_map = config.action_map
        self.default_action = config.default_action
        assert len(self.action_map) <= self.environment.action_space.n
        self.height, self.width, self.depth = config.image_shape
        self.d = config.idp // 2
        self.yrange = None
        self.xrange_left = None
        self.xrange_right = None

    def _set_range(self, frame: np.ndarray):
        # Get bottom left coordinates of screen to place image centrally.
        h, w, c = frame.shape
        y = (self.height - h) // 2
        x = (self.width - w) // 2
        self.yrange = range(y, y + h)
        self.xrange_left = range(x - self.d, x - self.d + w)
        self.xrange_right = range(x + self.d, x + self.d + w)

    def render(self, frame: np.ndarray) -> str:
        if self.yrange is None:
            self._set_range(frame)

        window = np.zeros((self.height, self.width, 1), 'uint8')

        # Center on left eye.
        window[np.ix_(self.yrange, self.xrange_left)] = frame
        # Center on right eye.
        window[np.ix_(self.yrange, self.xrange_right)] = frame

        cv2.imshow('human agent', window)
        return cv2.waitKey(1)  # in ms.

    def predict(self, observation: np.ndarray) -> Tuple[int, None]:
        key = self.render(observation)
        action = self.action_map.get(key, self.default_action)
        return action, None
