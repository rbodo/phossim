from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import socket
import gym
from gym import spaces

from phossim.config import AbstractConfig


@dataclass
class NeurosmashConfig(AbstractConfig):
    ip: str = '127.0.0.1'
    port: int = 13000
    resolution: int = 96
    render_mode: Optional[str] = None


def get_neurosmash_environment(config: AbstractConfig) -> gym.Env:
    return NeurosmashEnv(**asdict(config))


class NeurosmashEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, ip: str, port: int, resolution: int,
                 render_mode: Optional[str] = None):

        if render_mode is not None:
            assert render_mode in self.metadata['render_modes']

        super().__init__()
        self._shape = (resolution, resolution, 3)
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=self._shape, dtype=np.uint8)
        self._client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._client.connect((ip, port))
        self._state = None

    def step(self, action):
        self._send(action)
        return self._receive()

    def reset(self):
        self._send(0, 1)
        observation, _, _, _ = self._receive()
        return observation

    def render(self, mode='human'):
        if mode == 'rgb_array':
            return self._state
        else:
            super().render(mode)

    def _send(self, action, phase=2):
        self._client.send(bytes([phase, action]))

    def _receive(self):
        data = self._client.recv(2 + np.prod(self._shape), socket.MSG_WAITALL)

        done = data[0]
        reward = data[1]
        observation = data[2:]
        self._state = np.array(list(observation), 'uint8').reshape(self._shape)

        return observation, reward, done, {}
