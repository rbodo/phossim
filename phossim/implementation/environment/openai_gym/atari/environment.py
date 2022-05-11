from dataclasses import dataclass
from typing import Union

import gym
from stable_baselines3.common.atari_wrappers import AtariWrapper

from phossim.config import AbstractConfig
from phossim.implementation.environment.openai_gym.environment import \
    get_gym_environment, GymConfig


@dataclass
class AtariConfig(AbstractConfig):
    gym_config: GymConfig
    atari_kwargs: dict


def get_atari_environment(config: Union[AbstractConfig,
                                        AtariConfig]) -> gym.Env:
    env = get_gym_environment(config.gym_config)
    env = AtariWrapper(env, **config.atari_kwargs)
    return env
