from dataclasses import dataclass, field
from typing import Optional

import gym
import stable_baselines3
from stable_baselines3.common.base_class import BaseAlgorithm

from phossim.config import AbstractConfig
from phossim.interface import AgentConfig


@dataclass
class TrainingConfig(AbstractConfig):
    total_timesteps: int
    log_interval: int = 100
    tb_log_name: str = 'run'
    eval_env: Optional[gym.Env] = None
    eval_freq: int = -1
    n_eval_episodes: int = 5
    eval_log_path: Optional[str] = None
    reset_num_timesteps: bool = True


@dataclass
class StableBaselineAgentConfig(AgentConfig):
    model_id: str
    policy_id: str
    kwargs: Optional[dict] = field(default_factory=dict)


def get_agent(environment: gym.Env,
              config: StableBaselineAgentConfig) -> BaseAlgorithm:
    agent_class = getattr(stable_baselines3, config.model_id)
    agent = agent_class(config.policy_id, environment, **config.kwargs)
    return agent
