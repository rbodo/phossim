from dataclasses import dataclass

import cv2
import numpy as np
import gym

from phossim.interface import TransformConfig, Transform


@dataclass
class GrayscaleConfig(TransformConfig):
    observation_space: gym.Space


class GrayscaleTransform(Transform):
    def __init__(self, env, config: GrayscaleConfig):
        super().__init__(env, config)
        self._observation_space = config.observation_space

    def observation(self, observation):
        return np.atleast_3d(cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY))


@dataclass
class ResizeConfig(TransformConfig):
    observation_space: gym.Space


class ResizeTransform(Transform):
    def __init__(self, env, config: ResizeConfig):
        super().__init__(env, config)
        self._observation_space = config.observation_space
        self._target_shape = self._observation_space.shape

    def observation(self, observation):
        return cv2.resize(observation, (self._target_shape[1],
                                        self._target_shape[0]))
