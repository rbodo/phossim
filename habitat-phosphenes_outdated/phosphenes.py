import copy
from typing import Dict

import cv2
import numpy as np
import torch
from gym.spaces import Box

from habitat import get_config
from habitat.core import spaces
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.obs_transformers import ObservationTransformer
from habitat_baselines.utils.common import get_image_height_width


def overwrite_gym_box_shape(box: Box, shape) -> Box:
    if box.shape == shape:
        return box
    low = box.low if np.isscalar(box.low) else np.min(box.low)
    high = box.high if np.isscalar(box.high) else np.max(box.high)
    return Box(low=low, high=high, shape=shape, dtype=box.dtype)


@baseline_registry.register_obs_transformer()
class GrayScale(ObservationTransformer):
    def __init__(self):
        super().__init__()
        self.transformed_sensor = 'rgb'

    def forward(self, observations: Dict[str, torch.Tensor]
                ) -> Dict[str, torch.Tensor]:
        if self.transformed_sensor in observations:
            observations[self.transformed_sensor] = self._transform_obs(
                observations[self.transformed_sensor])
        return observations

    @staticmethod
    def _transform_obs(observation):
        device = observation.device

        observation = observation.cpu().numpy()

        frames = []
        for frame in observation:
            # Render the image
            # cv2.waitKey(1)
            # cv2.imshow("Before processing", frame)
            # cv2.waitKey(1)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Render the image
            # cv2.waitKey(1)
            # cv2.imshow("Gray Scale", frame)
            # cv2.waitKey(1)

            frames.append(frame)

        observation = torch.as_tensor(np.expand_dims(frames, -1),
                                      device=device)

        return observation

    def transform_observation_space(self, observation_space: spaces.Dict,
                                    **kwargs):
        key = self.transformed_sensor
        observation_space = copy.deepcopy(observation_space)

        h, w = get_image_height_width(observation_space[key],
                                      channels_last=True)
        new_shape = (h, w, 1)
        observation_space[key] = overwrite_gym_box_shape(
            observation_space[key], new_shape)
        return observation_space

    @classmethod
    def from_config(cls, config: get_config):
        # c = config.rl.policy.obs_transform.GrayScale
        return cls()


@baseline_registry.register_obs_transformer()
class EdgeFilter(ObservationTransformer):

    def __init__(self, sigma, threshold_low, threshold_high):
        super().__init__()
        self.sigma = sigma
        self.threshold_low = threshold_low
        self.threshold_high = threshold_high
        self.transformed_sensor = 'rgb'

    def forward(self, observations: Dict[str, torch.Tensor]
                ) -> Dict[str, torch.Tensor]:
        key = self.transformed_sensor
        if key in observations:
            observations[key] = self._transform_obs(observations[key])
        return observations

    def _transform_obs(self, observation: torch.Tensor):
        device = observation.device

        observation = observation.cpu().numpy()

        frames = []
        for frame in observation:

            # Gaussian blur to remove noise.
            frame = cv2.GaussianBlur(frame, ksize=None, sigmaX=self.sigma)

            # Canny edge detection.
            frame = cv2.Canny(frame, self.threshold_low, self.threshold_high)

            # Render the image
            # cv2.waitKey(1)
            # cv2.imshow("Edges", frame)
            # cv2.waitKey(1)

            # Copy grayscale image on each RGB channel so we can reuse
            # pre-trained net.
            # frames.append(np.tile(np.expand_dims(frame, -1), 3))
            frames.append(np.expand_dims(frame, -1))

        observation = torch.as_tensor(np.array(frames), device=device)

        return observation

    # def transform_observation_space(self, observation_space: spaces.Dict,
    #                                 **kwargs):
    #     key = self.transformed_sensor
    #     observation_space = copy.deepcopy(observation_space)
    #
    #     h, w = get_image_height_width(observation_space[key],
    #                                   channels_last=True)
    #     new_shape = (h, w, 3)
    #     observation_space[key] = overwrite_gym_box_shape(
    #         observation_space[key], new_shape)
    #     return observation_space

    @classmethod
    def from_config(cls, config: get_config):
        return cls(config.sigma, config.threshold_low, config.threshold_high)


@baseline_registry.register_obs_transformer()
class Phosphenes(ObservationTransformer):
    def __init__(self, size, phosphene_resolution, sigma):
        super().__init__()
        self.size = size
        self.phosphene_resolution = phosphene_resolution
        self.sigma = sigma
        jitter = 0.4
        intensity_var = 0.8
        aperture = 0.66
        self.transformed_sensor = 'rgb'
        self.grid = create_regular_grid((phosphene_resolution,
                                         phosphene_resolution),
                                        size, jitter, intensity_var)
        # relative aperture > dilation kernel size
        aperture = np.round(aperture *
                            size[0] / phosphene_resolution).astype(int)
        self.dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                         (aperture, aperture))

    @classmethod
    def from_config(cls, config: get_config):
        return cls(config.size, config.resolution, config.sigma)

    def forward(self, observations: Dict[str, torch.Tensor]
                ) -> Dict[str, torch.Tensor]:
        key = self.transformed_sensor
        if key in observations:
            observations[key] = self._transform_obs(observations[key])
        return observations

    def _transform_obs(self, observation):
        device = observation.device

        observation = observation.cpu().numpy()

        frames = []
        for frame in observation:
            mask = cv2.dilate(frame, self.dilation_kernel, iterations=1)
            phosphenes = self.grid * mask
            phosphenes = cv2.GaussianBlur(phosphenes, ksize=None,
                                          sigmaX=self.sigma)
            # Copy grayscale image on each RGB channel so we can reuse
            # pre-trained net.
            phosphenes = 255 * phosphenes / (phosphenes.max() or 1)
            frames.append(np.tile(np.expand_dims(phosphenes, -1), 3))

            # Render the image
            # cv2.waitKey(1)
            # cv2.imshow("Phosphenes", phosphenes)
            # cv2.waitKey(1)

        phosphenes = torch.as_tensor(np.array(frames, 'uint8'), device=device)

        return phosphenes

    def transform_observation_space(self, observation_space: spaces.Dict,
                                    **kwargs):
        key = self.transformed_sensor
        observation_space = copy.deepcopy(observation_space)

        h, w = get_image_height_width(observation_space[key],
                                      channels_last=True)
        new_shape = (h, w, 3)
        observation_space[key] = overwrite_gym_box_shape(
            observation_space[key], new_shape)
        return observation_space


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

@baseline_registry.register_obs_transformer()
class BlackScreen(ObservationTransformer):
    def __init__(self):
        super().__init__()
        self.transformed_sensor = 'rgb'

    def forward(self, observations: Dict[str, torch.Tensor]
                ) -> Dict[str, torch.Tensor]:
        if self.transformed_sensor in observations:
            observations[self.transformed_sensor] = self._transform_obs(
                observations[self.transformed_sensor])
        return observations

    @staticmethod
    def _transform_obs(observation):
        device = observation.device

        observation = observation.cpu().numpy()

        # Create a black screen of the same dimensions as the input frames
        black_screen = np.zeros_like(observation)
        black_screen = np.mean(black_screen, axis=3, keepdims=True)
        observation = torch.as_tensor(black_screen, device=device)

        return observation

    def transform_observation_space(self, observation_space: spaces.Dict,
                                    **kwargs):
        key = self.transformed_sensor
        observation_space = copy.deepcopy(observation_space)

        h, w = get_image_height_width(observation_space[key],
                                      channels_last=True)
        new_shape = (h, w, 1)
        observation_space[key] = overwrite_gym_box_shape(
            observation_space[key], new_shape)
        return observation_space

    @classmethod
    def from_config(cls, config: get_config):
        # c = config.rl.policy.obs_transform.GrayScale
        return cls()
