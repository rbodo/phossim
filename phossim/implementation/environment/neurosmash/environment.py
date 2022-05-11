import numpy as np
import socket
from PIL import Image

from phossim.interface.environment import AbstractEnvironment


class Environment(AbstractEnvironment):
    def __init__(self,
                 ip: str = '127.0.0.1',
                 port: int = 13000,
                 resolution: int = 768,
                 timescale: int = 1,
                 **kwargs):

        super().__init__(**kwargs)

        self._client = None
        self._ip = ip
        self._port = port
        self._resolution = resolution
        self._timescale = timescale
        self._rewards_history = []

    def setup(self, **kwargs):
        self._client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._client.connect((self._ip, self._port))

    def render(self):
        assert self._state is not None, "The environment.step() method must " \
                                        "be called before attempting to " \
                                        "visualize the state."

        return Image.fromarray(
            np.array(self._state, 'uint8').reshape((self._resolution,
                                                    self._resolution, 3)))

    def reset_state(self):
        self._send(0, 1)
        self._receive()
        super().reset_state()

    def reset_history(self):
        self._rewards_history = []
        super().reset_history()

    def step(self, action, **kwargs):
        self._send(action)
        return self._receive()

    def _receive(self):
        data = self._client.recv(2 + 3 * self._resolution ** 2,
                                 socket.MSG_WAITALL)

        self._is_done = data[0]
        self._reward = data[1]
        self._state = list(data[2:])

        self._rewards_history.append(self._reward)

        return self._state

    def _send(self, action, phase=2):
        self._client.send(bytes([phase, action]))

    def get_rewards_history(self):
        return self._rewards_history
