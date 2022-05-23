from dataclasses import dataclass, field
from typing import Optional

import gym
from stable_baselines3.common.atari_wrappers import AtariWrapper


@dataclass
class GymConfig:
    gym_id: str
    kwargs: dict


def get_gym_environment(config: GymConfig) -> gym.Env:
    return gym.make(config.gym_id, **config.kwargs)


@dataclass
class AtariConfig:
    gym_config: GymConfig
    atari_kwargs: Optional[dict] = field(default_factory=dict)


def get_atari_environment(config: AtariConfig) -> gym.Env:
    env = get_gym_environment(config.gym_config)
    env = AtariWrapper(env, **config.atari_kwargs)
    return env
