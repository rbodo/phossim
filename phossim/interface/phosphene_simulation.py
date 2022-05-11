import gym
from gym.utils.env_checker import check_env

from phossim.config import Config


def wrap_phosphene_simulation(environment: gym.Env, config: Config) -> gym.Env:
    environment = config.phosphene_simulation_wrapper(
        environment, config.phosphene_simulation)
    check_env(environment, skip_render_check=True)
    return environment
