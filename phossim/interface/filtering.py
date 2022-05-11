import gym
from gym.utils.env_checker import check_env

from phossim.config import Config


def wrap_filtering(environment: gym.Env, config: Config) -> gym.Env:
    environment = config.filtering_wrapper(environment, config.filtering)
    check_env(environment, skip_render_check=True)
    return environment
