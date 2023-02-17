from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Union, List, Tuple, Type, Optional, Callable

import cv2
import gym
import numpy as np
from stable_baselines3.common.monitor import Monitor


@dataclass
class TransformConfig:
    info_key: Union[str, None]


class Transform(gym.ObservationWrapper):
    def __init__(self, env, config: TransformConfig):
        super().__init__(env)
        self.config = config
        self._ndarray_to_render = None

    def observation(self, observation):
        return observation

    def step(self, action):
        observation, reward, done, info = super().step(action)
        info[self.config.info_key] = observation
        self._ndarray_to_render = observation
        return observation, reward, done, info

    def reset(self, **kwargs):
        observation = super().reset(**kwargs)
        self._ndarray_to_render = observation
        return observation

    def render(self, mode="human", **kwargs):
        if mode == 'rgb_array':
            if self._ndarray_to_render.shape[-1] == 1:
                return cv2.cvtColor(self._ndarray_to_render,
                                    cv2.COLOR_GRAY2RGB)
            else:
                return self._ndarray_to_render


def wrap_transforms(environment: gym.Env,
                    transforms: List[Tuple[Type[Transform],
                                           TransformConfig]]) -> gym.Env:
    for transform_class, transform_config in transforms:
        environment = transform_class(environment, transform_config)
    # check_env(environment, skip_render_check=False)
    return environment


@dataclass
class TimeLimitConfig:
    max_episode_steps: int = 100


@dataclass
class MonitorConfig:
    filename: str = None


class TimeLimitTransform(gym.wrappers.TimeLimit):
    def __init__(self, env, config: TimeLimitConfig):
        super().__init__(env, **asdict(config))


class MonitorTransform(Monitor):
    def __init__(self, env, config: MonitorConfig):
        super().__init__(env, **asdict(config))


@dataclass
class RecordingConfig:
    video_folder: Path
    episode_trigger: Optional[Callable[[int], bool]] = None
    step_trigger: Optional[Callable[[int], bool]] = None
    video_length: int = 0
    name_prefix: str = None


class RecordingTransform(gym.wrappers.RecordVideo):
    def __init__(self, env: gym.Env, config: RecordingConfig):
        if 'render_modes' in env.metadata:  # Fixing gym bug
            env.metadata['render.modes'] = env.metadata['render_modes']
        super().__init__(env, **asdict(config))


@dataclass
class GrayscaleConfig(TransformConfig):
    shape: Tuple[int, int, int]


class GrayscaleTransform(Transform):
    def __init__(self, env, config: GrayscaleConfig):
        super().__init__(env, config)
        self._observation_space = gym.spaces.Box(
            low=0, high=255, shape=config.shape, dtype=np.uint8)

    def observation(self, observation):
        return np.atleast_3d(cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY))


@dataclass
class ResizeConfig(TransformConfig):
    shape: Tuple[int, int, int]


class ResizeTransform(Transform):
    def __init__(self, env, config: ResizeConfig):
        super().__init__(env, config)
        self._observation_space = gym.spaces.Box(
            low=0, high=255, shape=config.shape, dtype=np.uint8)
        self._target_shape = self._observation_space.shape

    def observation(self, observation):
        return cv2.resize(observation, (self._target_shape[1],
                                        self._target_shape[0]))


@dataclass
class VrDisplayConfig(TransformConfig):
    shape: Tuple[int, int, int]
    idp: int = 1240


class VrDisplayTransform(Transform):
    def __init__(self, env: gym.Env, config: VrDisplayConfig):
        super().__init__(env, config)
        self._observation_space = gym.spaces.Box(
            low=0, high=255, shape=config.shape, dtype=np.uint8)
        d = config.idp // 2
        height_in, width_in = env.observation_space.shape[:2]
        height_out, width_out, _ = self.observation_space.shape
        y = (height_out - height_in) // 2
        x = (width_out - width_in) // 2
        self.yrange = range(y, y + height_in)
        self.xrange_left = range(x - d, x - d + width_in)
        self.xrange_right = range(x + d, x + d + width_in)
        self.display = self.observation_space.sample() * 0

    def observation(self, observation):
        # Center on left eye.
        self.display[np.ix_(self.yrange, self.xrange_left)] = observation
        # Center on right eye.
        self.display[np.ix_(self.yrange, self.xrange_right)] = observation

        return self.display
