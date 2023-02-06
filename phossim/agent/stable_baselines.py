from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import gym
import stable_baselines3
from stable_baselines3.common.base_class import BaseAlgorithm


@dataclass
class TrainingConfig:
    total_timesteps: int
    log_interval: int = 50
    tb_log_name: str = 'run'
    reset_num_timesteps: bool = True

    def asdict(self):
        return asdict(self)


@dataclass
class AgentConfig:
    path_model: Path
    model_id: str
    policy_id: str
    kwargs: Optional[dict] = field(default_factory=dict)


def get_agent(environment: gym.Env, config: AgentConfig) -> BaseAlgorithm:
    agent_class = getattr(stable_baselines3, config.model_id)
    agent = agent_class(config.policy_id, environment, **config.kwargs)
    return agent
