from dataclasses import dataclass
from typing import Tuple, Optional

import cv2
import gym
import numpy as np

from phossim.transforms import Transform, TransformConfig


@dataclass
class PhospheneSimulationConfig(TransformConfig):
    phosphene_resolution: Tuple[int, int] = (32, 32)
    phosphene_intensity: float = 8
    normalize_phosphenes: bool = True
    sigma: float = 1
    jitter: float = 0.4
    intensity_var: float = 0.8
    aperture: float = 0.66
    info_key = 'phosphenes'
    observation_space: Optional[gym.Space] = None


class PhospheneSimulation(Transform):
    def __init__(self, env: gym.Env, config: PhospheneSimulationConfig):
        """
        Phosphene simulator class to create gaussian-based phosphene
        simulations from a stimulus pattern.
        """

        super().__init__(env, config)
        if config.observation_space is not None:
            self.observation_space = config.observation_space
        size = env.observation_space.shape[:-1]
        phosphene_resolution = config.phosphene_resolution
        self.intensity = config.phosphene_intensity
        self._normalize = config.normalize_phosphenes
        if self._normalize and self.intensity:
            print("Warning: Phosphenes intensity is normalized; "
                  "'phosphene_intensity parameter is ignored.")
        self.sigma = config.sigma
        self.grid = create_regular_grid(phosphene_resolution,
                                        size, config.jitter,
                                        config.intensity_var)
        # relative aperture > dilation kernel size
        aperture = config.aperture
        aperture = np.round(aperture *
                            size[0] / phosphene_resolution[0]).astype(int)
        self.dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                         (aperture, aperture))

    def observation(self, observation):
        """Returns phosphene simulation (image), given an activation mask."""

        mask = cv2.dilate(observation, self.dilation_kernel, iterations=1)
        phosphenes = self.grid * mask
        phosphenes = cv2.GaussianBlur(phosphenes, ksize=None,
                                      sigmaX=self.sigma) * self.intensity

        if self._normalize:
            phosphenes = 255 * phosphenes / (phosphenes.max() or 1)

        return np.atleast_3d(phosphenes).astype('uint8')


def create_regular_grid(phosphene_resolution, size, jitter, intensity_var):
    """Returns regular eqiodistant phosphene grid of shape <size> with
    resolution <phosphene_resolution> for variable phosphene intensity with
    jittered positions"""

    grid = np.zeros(size)
    phosphene_spacing = np.divide(size, phosphene_resolution)
    xrange = np.linspace(0, size[0], phosphene_resolution[0], endpoint=False) \
        + phosphene_spacing[0] / 2
    yrange = np.linspace(0, size[1], phosphene_resolution[1], endpoint=False) \
        + phosphene_spacing[1] / 2
    for x in xrange:
        for y in yrange:
            deviation = \
                jitter * (2 * np.random.rand(2) - 1) * phosphene_spacing
            intensity = intensity_var * (np.random.rand() - 0.5) + 1
            rx = \
                np.clip(np.round(x + deviation[0]), 0, size[0] - 1).astype(int)
            ry = \
                np.clip(np.round(y + deviation[1]), 0, size[1] - 1).astype(int)
            grid[rx, ry] = intensity
    return grid
