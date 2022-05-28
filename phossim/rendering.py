from dataclasses import dataclass
from typing import Optional, List, Tuple

import cv2
import numpy as np
import pyglet
from pyglet.window import key


def get_display(spec: Optional[str] = None) -> pyglet.canvas.Display:
    """Convert a display specification (such as :0) into an actual Display
    object.

    Pyglet only supports multiple Displays on Linux.
    """
    if spec is None:
        return pyglet.canvas.get_display()
        # returns already available pyglet_display,
        # if there is no pyglet display available then it creates one
    elif isinstance(spec, str):
        return pyglet.canvas.Display(spec)
    else:
        raise RuntimeError("Invalid display specification: {}. (Must be a "
                           "string like :0 or None.)".format(spec))


def get_window(width: int, height: int, display: pyglet.canvas.Display,
               **kwargs) -> pyglet.window.Window:
    """
    Will create a pyglet window from the display specification provided.
    """
    screen = display.get_screens()  # available screens
    config = screen[0].get_best_config()  # selecting the first screen
    context = config.create_context(None)  # create GL context

    return pyglet.window.Window(
        width=width,
        height=height,
        display=display,
        config=config,
        context=context,
        **kwargs
    )


@dataclass
class ViewerConfig:
    shape: Tuple[int, int, int]
    name: str
    info_key: str
    label: Optional[str] = None
    display_id: Optional[str] = None


class Viewer:
    def __init__(self, config: ViewerConfig):
        self.height, self.width, self.depth = config.shape
        self.name = config.name
        self.info_key = config.info_key
        self._label = config.label
        self._format = 'L' if self.depth == 1 else 'RGB'

        display = get_display(config.display_id)

        self.window = get_window(width=self.width, height=self.height,
                                 display=display, caption=self.name)
        self.window.on_close = self.on_close
        self.window.on_key_press = self.on_key_press
        self._key = None

    # noinspection PyUnusedLocal
    def on_key_press(self, symbol, modifier):
        if symbol == key.ESCAPE:
            self.on_close()
        self._key = symbol

    def on_close(self):
        self.window.has_exit = True
        self.window.close()

    def render(self, frame: np.ndarray):
        self.window.switch_to()
        self.window.clear()
        self.window.dispatch_events()
        # frame = self.write_label_on_frame(frame)
        frame = np.flipud(frame).copy()
        image = (pyglet.gl.GLubyte * frame.size).from_buffer(frame)
        image = pyglet.image.ImageData(frame.shape[1], frame.shape[0],
                                       self._format, image)
        image.blit(0, 0)
        self.window.flip()

    def write_label_on_frame(self, frame: np.ndarray):
        if self._label is not None:
            return cv2.putText(img=frame.copy(),
                               text=f'{self.name}: {self._label}',
                               org=(10, 20),
                               fontFace=cv2.FONT_HERSHEY_DUPLEX,
                               fontScale=0.5,
                               color=(255, 255, 255))

    def get_key(self):
        k = self._key
        self._key = None
        return k


@dataclass
class VRViewerConfig(ViewerConfig):
    idp: int = 1240


class VRViewer(Viewer):
    """Continuously show a frame in VR style."""

    def __init__(self, config: VRViewerConfig):
        super().__init__(config)
        self.d = config.idp // 2
        self._screen = np.zeros((self.height, self.width, 1), 'uint8')
        self.yrange = None
        self.xrange_left = None
        self.xrange_right = None

    def set_range(self, frame: np.ndarray):
        # Get bottom left coordinates of screen to place image centrally.
        h, w, c = frame.shape
        y = (self.height - h) // 2
        x = (self.width - w) // 2
        self.yrange = range(y, y + h)
        self.xrange_left = range(x - self.d, x - self.d + w)
        self.xrange_right = range(x + self.d, x + self.d + w)

    def render(self, frame: np.ndarray):
        if self.window.has_exit:
            return

        if self.yrange is None:
            self.set_range(frame)

        # Center on left eye.
        self._screen[np.ix_(self.yrange, self.xrange_left)] = frame
        # Center on right eye.
        self._screen[np.ix_(self.yrange, self.xrange_right)] = frame

        super().render(self._screen)


class ViewerList:
    def __init__(self, viewers: List[Viewer]):
        self.viewers = viewers
        self.has_exit = False

    def render(self, info: dict):
        for viewer in self.viewers:
            frame = info.get(viewer.info_key, None)
            viewer.render(frame)
            if viewer.window.has_exit:
                self.stop()
                break

    def stop(self):
        for viewer in self.viewers:
            viewer.window.close()
        self.has_exit = True
