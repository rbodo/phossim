import gym
from gym.utils.env_checker import check_env

from phossim.config import Config


def wrap_stimulus_generation(environment: gym.Env, config: Config) -> gym.Env:
    environment = config.stimulus_generation_wrapper(environment,
                                                     config.filtering)
    check_env(environment, skip_render_check=True)
    return environment
