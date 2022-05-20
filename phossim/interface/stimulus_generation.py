import gym
from gym.utils.env_checker import check_env

from phossim.config import Config
from phossim.utils import add_observation_to_info


def wrap_stimulus_generation(environment: gym.Env, config: Config) -> gym.Env:
    environment = config.stimulus_generation_wrapper(environment,
                                                     config.filtering)
    check_env(environment, skip_render_check=True)
    return environment


class StimulusWrapper(gym.ObservationWrapper):
    def observation(self, observation):
        super().observation(observation)

    @add_observation_to_info('stimulus')
    def step(self, action):
        return super().step(action)
