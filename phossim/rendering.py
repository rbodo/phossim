from dataclasses import dataclass
from threading import Thread, Event
from typing import List

import cv2
import numpy as np


@dataclass
class ViewerConfig:
    info_key: str
    name: str


class Viewer:
    def __init__(self, config: ViewerConfig):
        self.name = config.name
        self.info_key = config.info_key

    def render(self, frame: np.ndarray):
        cv2.imshow(self.name, frame)


class ViewerList:
    def __init__(self, viewers: List[Viewer]):
        self.viewers = viewers
        self._thread = Thread(target=self._render, name='viewer')
        self._info = {}
        self.key = None
        self.read_event = Event()
        self.stop_event = Event()

    def _render(self):
        while not self.stop_event.is_set():
            key = cv2.waitKey(1)  # in ms.
            if key != -1:   # Only store if user pressed key.
                self.key = chr(key)
            if self.read_event.is_set():
                self.read_event.clear()
                for viewer in self.viewers:
                    frame = self._info.get(viewer.info_key, None)
                    if frame is not None:
                        viewer.render(frame)

    def render(self, info: dict) -> str:
        self._info = info
        self.read_event.set()
        return self.get_key()

    def get_key(self) -> str:
        key = self.key
        self.key = None
        return key

    def start(self):
        self._thread.start()

    def stop(self):
        self.stop_event.set()
        self._thread.join()
        cv2.destroyAllWindows()


class ViewerListBlocking:
    def __init__(self, viewers: List[Viewer]):
        self.viewers = viewers

    def render(self, info: dict) -> str:
        for viewer in self.viewers:
            frame = info.get(viewer.info_key, None)
            if frame is not None:
                viewer.render(frame)
        key = cv2.waitKey(0)  # Wait indefinitely for user input.
        if key != -1:  # Only store if user pressed key.
            return chr(key)

    def start(self):
        pass

    @staticmethod
    def stop():
        cv2.destroyAllWindows()
