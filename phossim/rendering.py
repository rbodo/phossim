from dataclasses import dataclass
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

    def render(self, info: dict) -> str:
        for viewer in self.viewers:
            frame = info.get(viewer.info_key, None)
            if frame is not None:
                viewer.render(frame)

        return cv2.waitKey(1)  # in ms.

    @staticmethod
    def stop():
        cv2.destroyAllWindows()
