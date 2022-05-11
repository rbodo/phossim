from dataclasses import dataclass

import gym


@dataclass
class GymConfig:
    gym_id: str
    kwargs: dict


def get_gym_environment(config: GymConfig) -> gym.Env:
    return gym.make(config.gym_id, **config.kwargs)
