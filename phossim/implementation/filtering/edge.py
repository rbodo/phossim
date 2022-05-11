from dataclasses import dataclass, asdict
from typing import Union

import numpy as np
import cv2
import gym
from gym.core import ObservationWrapper

from phossim.config import AbstractConfig
from phossim.implementation.filtering.preprocessing import (
    PreprocessingConfig, PreprocessingFilter)


@dataclass
class CannyConfig(AbstractConfig):
    sigma: float = 3
    threshold_low: float = 20
    threshold_high: float = 40


@dataclass
class PreprocessingCannyConfig(AbstractConfig):
    preprocessing_config: PreprocessingConfig
    canny_config: CannyConfig


def wrap_preprocessing_canny(
        environment: gym.Env,
        config: Union[AbstractConfig, PreprocessingCannyConfig]) -> gym.Env:

    environment = PreprocessingFilter(environment,
                                      **asdict(config.preprocessing_config))
    environment = CannyFilter(environment,
                              **asdict(config.canny_config))

    return environment


def wrap_canny(environment: gym.Env,
               config: Union[AbstractConfig, CannyConfig]) -> gym.Env:

    environment = CannyFilter(environment, **asdict(config))

    return environment


class CannyFilter(ObservationWrapper):
    def __init__(self,
                 env: gym.Env,
                 sigma: float,
                 threshold_low: float,
                 threshold_high: float):
        super().__init__(env)
        self.sigma = sigma
        self.threshold_low = threshold_low
        self.threshold_high = threshold_high

    def observation(self, observation):

        # Gaussian blur to remove noise.
        observation = cv2.GaussianBlur(observation, ksize=None,
                                       sigmaX=self.sigma)

        # Canny edge detection.
        observation = cv2.Canny(observation, self.threshold_low,
                                self.threshold_high)

        return np.atleast_3d(observation)
