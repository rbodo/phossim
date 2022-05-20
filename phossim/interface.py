from dataclasses import dataclass

import gym
from gym.utils.env_checker import check_env
from stable_baselines3.common.policies import BasePolicy

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

    def observation(self, observation):
        return observation

    def step(self, action):
        observation, reward, done, info = super().step(action)
        observation = self.observation(observation)
        info[self.config.info_key] = observation
        return observation, reward, done, info


def wrap_transforms(environment: gym.Env, config: Config):
    for transform_class, transform_config in config.transform_configs:
        environment = transform_class(environment, transform_config)
        check_env(environment, skip_render_check=True)
    return environment


def get_agent(environment: gym.Env, config: Config) -> BasePolicy:
    agent = config.agent_getter(environment, config.agent_config)
    return agent
