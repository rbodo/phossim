from __future__ import annotations
import pathlib
from dataclasses import dataclass
from typing import Callable, Tuple, Optional, List, Type, TYPE_CHECKING

import gym
from stable_baselines3.common.policies import BasePolicy
import torch

if TYPE_CHECKING:
    from phossim.rendering import DisplayConfig, Display
    from phossim.interface import Transform, TransformConfig

USE_CUDA = False
DEVICE = 'cuda:0' if USE_CUDA else 'cpu'
DTYPE = torch.float32

QUIT_KEY = 'q'
KNOWN_KEYS = [ord(QUIT_KEY)]


@dataclass
class AbstractConfig:
    pass


@dataclass
class Config(AbstractConfig):
    """Main configuration class.

    Note:
        Have to separate default values from non-default values in two config
        classes to be able to inherit from the config. When upgrading to python
         3.10, a single config can be used by specifying
         @dataclass(kw_only=True). See also stackoverflow.com/questions/
         51575931/class-inheritance-in-python-3-7-data
    """

    environment_getter: Callable[[AbstractConfig], gym.Env]
    agent_getter: Callable[[gym.Env, AbstractConfig], BasePolicy]

    environment_config: AbstractConfig
    transform_configs: List[Tuple[Type[Transform], TransformConfig]]
    agent_config: AbstractConfig
    display_configs: List[Tuple[Type[Display], DisplayConfig]]

    filepath_output_data: pathlib.Path = None
    seed: int = 42
    max_num_episodes: int = 100
    max_episode_steps: int = 100
    record_episode_statistics_deque_size: int = 100
    monitor_filename: Optional[str] = None
    info_keywords: Tuple[str, ...] = ()

    def apply_key(self, key: int):
        pass
