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
    observation_space: gym.Space


class GrayscaleTransform(Transform):
    def __init__(self, env, config: GrayscaleConfig):
        super().__init__(env, config)
        self._observation_space = config.observation_space

    def observation(self, observation):
        return np.atleast_3d(cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY))


@dataclass
class ResizeConfig(TransformConfig):
    observation_space: gym.Space


class ResizeTransform(Transform):
    def __init__(self, env, config: ResizeConfig):
        super().__init__(env, config)
        self._observation_space = config.observation_space
        self._target_shape = self._observation_space.shape

    def observation(self, observation):
        return cv2.resize(observation, (self._target_shape[1],
                                        self._target_shape[0]))
