import gym
from stable_baselines3.common.policies import BasePolicy

from phossim.config import Config


def get_agent(environment: gym.Env, config: Config) -> BasePolicy:
    agent = config.agent_getter(environment, config.agent)
    return agent
