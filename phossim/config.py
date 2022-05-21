from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Tuple, List, Type, TYPE_CHECKING, Union

import gym
from stable_baselines3.common.base_class import BaseAlgorithm
import torch

if TYPE_CHECKING:
    from phossim.rendering import DisplayConfig, Display
    from phossim.interface import Transform, TransformConfig, AgentConfig

USE_CUDA = True
gpu = '4'
DEVICE = 'cuda:0' if USE_CUDA else 'cpu'
DTYPE = torch.float32

os.environ['CUDA_VISIBLE_DEVICES'] = gpu

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

    environment_getter: Union[Type[gym.Env], Callable[[...], gym.Env]]
    agent_getter: Callable[[gym.Env, ...], BaseAlgorithm]

    environment_config: AbstractConfig
    transform_configs: List[Tuple[Type[Transform], TransformConfig]]
    agent_config: AgentConfig
    display_configs: List[Tuple[Type[Display], DisplayConfig]]
    training_config: AbstractConfig

    max_num_episodes: int = 100
    max_episode_steps: int = 100

    def apply_key(self, key: int):
        pass
