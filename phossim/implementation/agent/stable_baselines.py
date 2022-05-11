from dataclasses import dataclass
from typing import Union

import gym
import stable_baselines3
from stable_baselines3.common.policies import BasePolicy

from phossim.config import AbstractConfig


@dataclass
class AgentConfig(AbstractConfig):
    model_id: str
    policy_id: str
    kwargs: dict


def get_agent(environment: gym.Env,
              config: Union[AbstractConfig, AgentConfig]) -> BasePolicy:
    agent_class = getattr(stable_baselines3, config.model_id)
    agent = agent_class(config.policy_id, environment, **config.kwargs)
    return agent
