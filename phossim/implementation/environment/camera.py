import os

import struct
from dataclasses import dataclass

import cv2
import gym
from dv import NetworkFrameInput

from phossim.config import AbstractConfig


@dataclass
class CameraConfig(AbstractConfig):
    observation_space: gym.Space


@dataclass
class DVSConfig(CameraConfig):
    ip: str = '127.0.0.1'
    port: int = 13000


class AbstractVideoStream(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, config: AbstractConfig):
        super().__init__()

        self.config = config
        self.stream = None
        self.frame = None

    def reset(self):
        return self.frame

    def render(self, mode: str = 'human'):
        if mode == 'rgb_array':
            return self.frame
        else:
            super().render(mode)

    def close(self):
        del self.stream


class CameraStream(AbstractVideoStream):
    """
    Class that continuously gets frames from a VideoCapture object.
    """

    def __init__(self, config):
        super().__init__(config)
        a = cv2.CAP_DSHOW if os.name == 'nt' else None  # Only for Windows
        self.stream = cv2.VideoCapture(config.cameradevice, a)
        grabbed, self.frame = self.stream.read()

    def step(self, action):
        grabbed, self.frame = self.stream.read()
        if not grabbed:
            self.close()
        return self.frame

    def close(self):
        self.stream.release()


class DVSFrameStream(AbstractVideoStream):
    """
    Class that continuously gets frames from a NetworkFrameInput object.
    """

    def __init__(self, config: DVSConfig):
        super().__init__(config)
        try:
            self.stream = NetworkFrameInput(address=config.ip,
                                            port=config.port)
        except ConnectionError as e:
            print(e)
            return
        self.frame = self.stream.__next__().image[:, :, 0]

    def step(self, action):
        try:
            self.frame = self.stream.__next__().image[:, :, 0]
        except struct.error or ConnectionAbortedError as e:
            print(e, '\n')
            self.close()
        return self.frame

    def close(self):
        if self.stream is not None:
            self.stream.__del__()
