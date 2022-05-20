import gym
from gym.utils.env_checker import check_env

from phossim.config import Config
from phossim.utils import add_observation_to_info


def wrap_filtering(environment: gym.Env, config: Config) -> gym.Env:
    environment = config.filtering_wrapper(environment, config.filtering)
    check_env(environment, skip_render_check=True)
    return environment


class FilteringWrapper(gym.ObservationWrapper):
    def observation(self, observation):
        super().observation(observation)

    @add_observation_to_info('filtered_observation')
    def step(self, action):
        return super().step(action)
