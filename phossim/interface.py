from dataclasses import dataclass
from pathlib import Path

import cv2
import gym
from gym.utils.env_checker import check_env
from stable_baselines3.common.base_class import BaseAlgorithm

from phossim.config import Config, AbstractConfig


def get_environment(config: Config):
    environment = config.environment_getter(config.environment_config)
    check_env(environment, skip_render_check=True)
    return environment


@dataclass
class TransformConfig(AbstractConfig):
    info_key: str


@dataclass
class Transform(gym.ObservationWrapper):
    def __init__(self, env, config: TransformConfig):
        super().__init__(env)
        self.config = config
        self._ndarray_to_render = None

    def observation(self, observation):
        return observation

    def step(self, action):
        observation, reward, done, info = super().step(action)
        observation = self.observation(observation)
        info[self.config.info_key] = observation
        self._ndarray_to_render = observation
        return observation, reward, done, info

    def render(self, mode="human", **kwargs):
        if mode == 'rgb_array':
            return cv2.cvtColor(self._ndarray_to_render, cv2.COLOR_GRAY2RGB)


def wrap_transforms(environment: gym.Env, config: Config):
    for transform_class, transform_config in config.transform_configs:
        environment = transform_class(environment, transform_config)
        check_env(environment, skip_render_check=False)
    return environment


@dataclass
class AgentConfig(AbstractConfig):
    path_model: Path


def get_agent(environment: gym.Env, config: Config) -> BaseAlgorithm:
    agent = config.agent_getter(environment, config.agent_config)
    return agent
