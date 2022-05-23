from dataclasses import dataclass

import numpy as np
import cv2
import gym

from phossim.transforms import Transform, TransformConfig


@dataclass
class CannyConfig(TransformConfig):
    sigma: float = 3
    threshold_low: float = 20
    threshold_high: float = 40


class CannyFilter(Transform):
    def __init__(self, env: gym.Env, config: CannyConfig):
        super().__init__(env, config)
        self.sigma = config.sigma
        self.threshold_low = config.threshold_low
        self.threshold_high = config.threshold_high

    def observation(self, observation):

        # Gaussian blur to remove noise.
        observation = cv2.GaussianBlur(observation, ksize=None,
                                       sigmaX=self.sigma)

        # Canny edge detection.
        observation = cv2.Canny(observation, self.threshold_low,
                                self.threshold_high)

        return np.atleast_3d(observation)
