from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np
import gym
from gym.core import ObservationWrapper


@dataclass
class PreprocessingConfig:
    target_size: Tuple[int, int]
    zoom: float = 1


class PreprocessingFilter(ObservationWrapper):
    def __init__(self,
                 env: gym.Env,
                 target_size: Tuple[int, int],
                 zoom: float = 1):
        super().__init__(env)
        self.target_size = target_size
        self.zoom = zoom

    def observation(self, observation):

        # If not done yet, convert to grayscale.
        if observation.ndim == 3 and observation.shape[2] == 3:
            observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
        if self.zoom > 1 or observation.shape != self.target_size:
            observation = center_crop(observation, self.zoom, self.target_size)
        return observation


def center_crop(img, zoom=1.9, resize_to=None, square_crop=False):
    """Center-crop and resize img to height x width."""

    h, w = img.shape

    if square_crop:
        d = min(w, h)
        m1 = int((d - d / zoom) / 2)
        m2 = int((d - d / zoom + abs(w - h)) / 2)

        # Check if we need to flip temporarily so the first axis is always the
        # smaller one.
        do_flip = h > w
        if do_flip:
            img = np.transpose(img)
        img = img[m1:-m1-1, m2:-m2-1]
        if do_flip:
            img = np.transpose(img)
    else:
        y = int((h - h / zoom) / 2)
        x = int((w - w / zoom) / 2)
        img = img[y:-y-1, x:-x-1]

    if resize_to is not None:
        img = cv2.resize(img, resize_to[::-1])

    return img
