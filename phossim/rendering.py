from dataclasses import dataclass
from typing import Optional, Tuple, List

import cv2
import numpy as np

from phossim.pipeline import QUIT_KEY


@dataclass
class DisplayConfig:
    name: str
    info_key: str
    label: Optional[str] = None
    waitkey_delay: int = 1
    known_keys: Tuple[str] = (QUIT_KEY,)


class Display:
    def __init__(self, config: DisplayConfig):
        self.name = config.name
        self.info_key = config.info_key
        self._label = config.label
        self.stopped = False
        self._waitkey_delay: int = config.waitkey_delay  # in ms.
        self.known_keys = config.known_keys

    def stop(self):
        self.stopped = True
        cv2.destroyAllWindows()

    def render(self, frame: np.ndarray) -> int:
        cv2.imshow(self.name, frame)

        return cv2.waitKey(self._waitkey_delay)


class ScreenDisplay(Display):
    """Continuously show a frame on screen using a dedicated thread."""

    def render(self, frame: np.ndarray) -> int:
        frame = self.write_label_on_frame(frame)
        return super().render(frame)

    def write_label_on_frame(self, frame: np.ndarray):
        if self._label is not None:
            return cv2.putText(img=frame.copy(),
                               text=f'{self.name}: {self._label}',
                               org=(10, 20),
                               fontFace=cv2.FONT_HERSHEY_DUPLEX,
                               fontScale=0.5,
                               color=(255, 255, 255))


@dataclass
class VRDisplayConfig(DisplayConfig):
    size: Tuple[int, int] = (1600, 2880, 1)
    idp: int = 1240


class VRDisplay(Display):
    """Continuously show a frame (in VR style) using a dedicated thread."""

    def __init__(self, config: VRDisplayConfig):
        super().__init__(config)
        self.display_size = config.size
        self.d = config.idp // 2
        self._screen = np.zeros(self.display_size, 'uint8')
        self.yrange = None
        self.xrange_left = None
        self.xrange_right = None

    def set_range(self, frame: np.ndarray):
        # Get bottom left coordinates of screen to place image centrally.
        h, w, c = frame.shape
        y = (self.display_size[0] - h) // 2
        x = (self.display_size[1] - w) // 2
        self.yrange = range(y, y + h)
        self.xrange_left = range(x - self.d, x - self.d + w)
        self.xrange_right = range(x + self.d, x + self.d + w)
        cv2.imshow(self.name, self._screen)
        cv2.setWindowProperty(self.name, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN)

    def render(self, frame: np.ndarray) -> int:
        if self.yrange is None:
            self.set_range(frame)

        # Center on left eye.
        self._screen[np.ix_(self.yrange, self.xrange_left)] = frame
        # Center on right eye.
        self._screen[np.ix_(self.yrange, self.xrange_right)] = frame

        return super().render(self._screen)


class DisplayList:
    def __init__(self, displays: List[Display]):
        self.displays = displays

    def render(self, info: dict) -> int:
        for display in self.displays:
            frame = info.get(display.info_key, None)
            if frame is not None:
                key = display.render(frame)
                if key >= 0 and chr(key) in display.known_keys:
                    return key

    def stop(self):
        for display in self.displays:
            display.stop()
