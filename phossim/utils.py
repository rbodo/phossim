from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Callable

import gym
from gym.wrappers import TimeLimit, RecordEpisodeStatistics, RecordVideo
from stable_baselines3.common.monitor import Monitor

from phossim.config import Config, AbstractConfig


@dataclass
class RecordingConfig(AbstractConfig):
    video_folder: Path
    episode_trigger: Optional[Callable[[int], bool]] = None
    step_trigger: Optional[Callable[[int], bool]] = None
    video_length: int = 0
    name_prefix: str = ''


def wrap_common(environment: gym.Env, config: Config) -> gym.Env:
    environment = TimeLimit(environment, config.max_episode_steps)
    environment = RecordEpisodeStatistics(
        environment, config.record_episode_statistics_deque_size)
    environment = Monitor(environment, config.monitor_filename,
                          info_keywords=config.info_keywords)
    environment = RecordVideo(environment, **asdict(config.recording_config))
    return environment
