import gym
from gym.utils.env_checker import check_env

from phossim.config import Config
from phossim.utils import add_observation_to_info


def wrap_phosphene_simulation(environment: gym.Env, config: Config) -> gym.Env:
    environment = config.phosphene_simulation_wrapper(
        environment, config.phosphene_simulation)
    check_env(environment, skip_render_check=True)
    return environment


class PhospheneWrapper(gym.ObservationWrapper):
    def observation(self, observation):
        super().observation(observation)

    @add_observation_to_info('phosphenes')
    def step(self, action):
        return super().step(action)
