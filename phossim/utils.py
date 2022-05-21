from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Callable

import gym
from gym.wrappers import TimeLimit, RecordVideo
from stable_baselines3.common.monitor import Monitor

from phossim.config import Config, AbstractConfig


@dataclass
class RecordingConfig(AbstractConfig):
    video_folder: Path
    episode_trigger: Optional[Callable[[int], bool]] = None
    step_trigger: Optional[Callable[[int], bool]] = None
    video_length: int = 0
    name_prefix: str = None


class RecordingTransform(RecordVideo):
    def __init__(self, env: gym.Env, config: RecordingConfig):
        env.metadata['render.modes'] = env.metadata['render_modes']
        super().__init__(env, **asdict(config))


def wrap_common(environment: gym.Env, config: Config) -> gym.Env:
    environment = TimeLimit(environment, config.max_episode_steps)
    environment = Monitor(environment)
    return environment
