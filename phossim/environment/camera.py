import os

import struct
from dataclasses import dataclass

import cv2
import gym
from dv import NetworkFrameInput


@dataclass
class CameraConfig:
    observation_space: gym.Space
    camera_id: int = 0


@dataclass
class DVSConfig(CameraConfig):
    ip: str = '127.0.0.1'
    port: int = 13000


class AbstractVideoStream(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self):
        super().__init__()

        self.stream = None
        self.frame = None

    def reset(self):
        self._update_frame()
        return self.frame

    def step(self, action):
        self._update_frame()
        return self.frame, 0, False, {}

    def _update_frame(self):
        raise NotImplementedError

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

    def __init__(self, config: CameraConfig):
        super().__init__()
        a = cv2.CAP_DSHOW if os.name == 'nt' else None  # Only for Windows
        self.stream = cv2.VideoCapture(config.camera_id, a)

    def _update_frame(self):
        grabbed, self.frame = self.stream.read()
        if not grabbed:
            self.close()

    def close(self):
        self.stream.release()


class DVSFrameStream(AbstractVideoStream):
    """
    Class that continuously gets frames from a NetworkFrameInput object.
    """

    def __init__(self, config: DVSConfig):
        super().__init__()
        self.stream = NetworkFrameInput(config.ip, config.port)
        self.action_space = gym.spaces.Discrete(1)
        self.observation_space = config.observation_space

    def _update_frame(self):
        try:
            self.frame = self.stream.__next__().image
        except struct.error or ConnectionAbortedError as e:
            print(e, '\n')
            self.close()

    def close(self):
        if self.stream is not None:
            self.stream.__del__()
