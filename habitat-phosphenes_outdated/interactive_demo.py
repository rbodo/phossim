import copy
import os
import sys
from pathlib import Path

import cv2
import numpy as np

import habitat
from habitat import get_config
from habitat.core import spaces
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat_baselines.common.obs_transformers import ObservationTransformer
from habitat_baselines.utils.common import get_image_height_width
from phosphenes import create_regular_grid, overwrite_gym_box_shape

FORWARD_KEY = "w"
LEFT_KEY = "a"
RIGHT_KEY = "d"
FINISH = "q"


class GrayScale(ObservationTransformer):
    def __init__(self):
        super().__init__()
        self.transformed_sensor = 'rgb'

    def forward(self, observations):
        if self.transformed_sensor in observations:
            observations[self.transformed_sensor] = self._transform_obs(
                observations[self.transformed_sensor])
        return observations

    @staticmethod
    def _transform_obs(observation):
        return cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)

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
        return cls()


class Rescale(ObservationTransformer):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape
        self.transformed_sensor = 'rgb'

    def forward(self, observations):
        if self.transformed_sensor in observations:
            observations[self.transformed_sensor] = self._transform_obs(
                observations[self.transformed_sensor])
        return observations

    def _transform_obs(self, observation):
        return cv2.resize(observation, self.shape)

    def transform_observation_space(self, observation_space: spaces.Dict,
                                    **kwargs):
        key = self.transformed_sensor
        observation_space = copy.deepcopy(observation_space)

        new_shape = self.shape + (1,)
        observation_space[key] = overwrite_gym_box_shape(
            observation_space[key], new_shape)
        return observation_space

    @classmethod
    def from_config(cls, config: get_config):
        return cls(config.shape)


class EdgeFilter(ObservationTransformer):

    def __init__(self, sigma, threshold_low, threshold_high):
        super().__init__()
        self.sigma = sigma
        self.threshold_low = threshold_low
        self.threshold_high = threshold_high
        self.transformed_sensor = 'rgb'

    def forward(self, observations):
        key = self.transformed_sensor
        if key in observations:
            observations[key] = self._transform_obs(observations[key])
        return observations

    def _transform_obs(self, observation):
        # Gaussian blur to remove noise.
        observation = cv2.GaussianBlur(observation, ksize=None,
                                       sigmaX=self.sigma)

        # Canny edge detection.
        return cv2.Canny(observation, self.threshold_low, self.threshold_high)

    @classmethod
    def from_config(cls, config: get_config):
        c = config.RL.POLICY.OBS_TRANSFORMS.EDGE_FILTER
        return cls(c.SIGMA, c.THRESHOLD_LOW, c.THRESHOLD_HIGH)


class Phosphenes(ObservationTransformer):
    def __init__(self, size, phosphene_resolution, sigma):
        super().__init__()
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
        c = config.RL.POLICY.OBS_TRANSFORMS.PHOSPHENES
        return cls(c.SIZE, c.RESOLUTION, c.SIGMA)

    def forward(self, observations):
        key = self.transformed_sensor
        if key in observations:
            observations[key] = self._transform_obs(observations[key])
        return observations

    def _transform_obs(self, observation):
        mask = cv2.dilate(observation, self.dilation_kernel, iterations=1)
        phosphenes = self.grid * mask
        phosphenes = cv2.GaussianBlur(phosphenes, ksize=None,
                                      sigmaX=self.sigma)
        phosphenes = 255 * phosphenes / (phosphenes.max() or 1)

        return np.array(phosphenes, 'uint8')


def transform_rgb_bgr(observation):
    observation['rgb'] = observation['rgb'][:, :, [2, 1, 0]]
    return observation


def example():
    # config = habitat.get_config("habitat-lab/habitat/config/benchmark/nav/pointnav/pointnav_gibson.yaml")
    config = habitat.get_config("benchmark/nav/pointnav/pointnav_habitat_test.yaml")
    config.defrost()
    config['DATASET']['SPLIT'] = 'val'
    config.freeze()
    env = habitat.Env(config=config)
    shape = (512, 512)
    grayscale = GrayScale()
    rescale = Rescale(shape)
    edge = EdgeFilter(3, 25, 50)
    phosphene_simulator = Phosphenes(shape, 32, 3)

    def transform_phosphenes(observation):
        observation = grayscale.forward(observation)
        observation = rescale.forward(observation)
        observation = edge.forward(observation)
        observation = phosphene_simulator(observation)
        return observation

    use_phosphenes = True
    transform = transform_phosphenes if use_phosphenes else transform_rgb_bgr

    print("Environment creation successful")
    observations = env.reset()
    print("Destination, distance: {:3f}, theta(radians): {:.2f}".format(
        observations["pointgoal_with_gps_compass"][0],
        observations["pointgoal_with_gps_compass"][1]))
    cv2.imshow("RGB", transform(observations)['rgb'])

    print("Agent stepping around inside environment.")

    action = None
    count_steps = 0
    while not env.episode_over:
        keystroke = cv2.waitKey(0)

        if keystroke == ord(FORWARD_KEY):
            action = HabitatSimActions.MOVE_FORWARD
            print("action: FORWARD")
        elif keystroke == ord(LEFT_KEY):
            action = HabitatSimActions.TURN_LEFT
            print("action: LEFT")
        elif keystroke == ord(RIGHT_KEY):
            action = HabitatSimActions.TURN_RIGHT
            print("action: RIGHT")
        elif keystroke == ord(FINISH):
            action = HabitatSimActions.STOP
            print("action: FINISH")
        else:
            print("INVALID KEY")
            continue

        observations = env.step(action)
        count_steps += 1

        print("Destination, distance: {:3f}, theta(radians): {:.2f}".format(
            observations["pointgoal_with_gps_compass"][0],
            observations["pointgoal_with_gps_compass"][1]))
        cv2.imshow("RGB", transform(observations)['rgb'])

    print("Episode finished after {} steps.".format(count_steps))

    if (
        action == HabitatSimActions.STOP
        and observations["pointgoal_with_gps_compass"][0] < 0.2
    ):
        print("you successfully navigated to destination point")
    else:
        print("your navigation was unsuccessful")


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '8'
    os.chdir(Path('~/Internship/PyCharm_projects/habitat-lab/').expanduser())
    example()
    sys.exit()
