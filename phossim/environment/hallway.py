from dataclasses import dataclass
from typing import Tuple

import numpy as np
import socket
import gym


@dataclass
class HallwayConfig:
    shape: Tuple[int, int, int]
    ip: str = '127.0.0.1'
    port: int = 13000


class Hallway(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, config: HallwayConfig):

        super().__init__()
        self._size = config.shape[0]  # Assumes quadratic resolution
        self._num_channels = 16
        self._num_actions = 3
        self.action_space = gym.spaces.Discrete(self._num_actions)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=config.shape, dtype=np.uint8)
        self._client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._client.connect((config.ip, config.port))
        self._state_dict = None

    def step(self, action):
        self._send(action)
        return self._receive()

    def reset(self, **kwargs):
        self._send(0, 1)
        observation, _, _, _ = self._receive()
        return observation

    def render(self, mode='human'):
        if mode == 'rgb_array':
            return self._state_dict['rgb']
        else:
            super().render(mode)

    def _send(self, action, phase=2):
        self._client.send(bytes([phase, action]))

    def _receive(self):
        data = self._client.recv(
            2 + self._size * self._size * self._num_channels,
            socket.MSG_WAITALL)

        done = bool(data[0])
        reward = data[1]
        state = data[2:]
        state = np.array(list(state), 'uint8').reshape((self._size, self._size,
                                                        self._num_channels))
        self._state_dict = {'rgb': state[..., :3],
                            'object_segmentation': state[..., 3:6],
                            'semantic_segmentation': state[..., 6:9],
                            'normals': state[..., 9:12],
                            'flow': state[..., 12:15],
                            'depth': state[..., 15]}

        return self._state_dict['rgb'], reward, done, {}
