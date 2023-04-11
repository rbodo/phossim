from dataclasses import dataclass
from typing import Tuple

import gym
import habitat
import numpy as np


@dataclass
class AihabitatConfig:
    shape: Tuple[int, int, int]
    path_config: str


class Aihabitat(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, config: AihabitatConfig):

        super().__init__()
        _config = habitat.get_config(config.path_config)
        self.env = habitat.Env(config=_config)
        self._num_channels = 16
        self._num_actions = 4
        self.action_space = gym.spaces.Discrete(self._num_actions)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=config.shape, dtype=np.uint8)
        self._state = None

    def step(self, action):
        observations = self.env.step(action)
        observation = observations['rgb']
        reward = get_reward(observations)
        done = self.env.episode_over or reward
        self._state = observation.astype('uint8')
        print_state(observations)
        return observation, reward, done, {}

    def reset(self, **kwargs):
        observations = self.env.reset()
        print_state(observations)
        return observations['rgb']

    def render(self, mode='human'):
        if mode == 'rgb_array':
            return self._state
        else:
            super().render(mode)


def get_reward(observations):
    return observations['pointgoal_with_gps_compass'][0] < 2


def print_state(observations):
    print("Destination distance: {:3f}, theta (radians): {:.2f}".format(
        *observations['pointgoal_with_gps_compass']))
