from dataclasses import dataclass
from typing import Optional

import numpy as np
import socket
import gym


@dataclass
class NeurosmashConfig:
    ip: str = '127.0.0.1'
    port: int = 13000
    resolution: int = 96
    render_mode: Optional[str] = None


class Neurosmash(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, config: NeurosmashConfig):

        if config.render_mode is not None:
            assert config.render_mode in self.metadata['render_modes']

        super().__init__()
        self._shape = (config.resolution, config.resolution, 3)
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(low=0, high=255,
                                                shape=self._shape,
                                                dtype=np.uint8)
        self._client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._client.connect((config.ip, config.port))
        self._state = None

    def step(self, action):
        self._send(action)
        return self._receive()

    def reset(self, **kwargs):
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
