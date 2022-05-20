from dataclasses import dataclass, field
from typing import Union, Optional

import gym
from stable_baselines3.common.atari_wrappers import AtariWrapper

from phossim.config import AbstractConfig
from phossim.implementation.environment.openai_gym.environment import \
    get_gym_environment, GymConfig


@dataclass
class AtariConfig(AbstractConfig):
    gym_config: GymConfig
    atari_kwargs: Optional[dict] = field(default_factory=dict)


def get_atari_environment(config: Union[AbstractConfig,
                                        AtariConfig]) -> gym.Env:
    env = get_gym_environment(config.gym_config)
    env = AtariWrapper(env, **config.atari_kwargs)
    return env
