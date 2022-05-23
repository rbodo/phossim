from dataclasses import dataclass, asdict
from typing import Union, List, Tuple, Type

import cv2
import gym
from gym.utils.env_checker import check_env
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
                    transforms: List[Tuple[Type[Transform], TransformConfig]],
                    skip_render_check: bool = False) -> gym.Env:
    for transform_class, transform_config in transforms:
        environment = transform_class(environment, transform_config)
    environment.reset()
    check_env(environment, skip_render_check=skip_render_check)
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
