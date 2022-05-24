import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Type, Optional

import gym
import numpy as np

from phossim.pipeline import BasePipeline
from phossim.environment.hallway import HallwayConfig, Hallway
from phossim.transforms import (Transform, TransformConfig, TimeLimitConfig,
                                MonitorConfig, MonitorTransform,
                                TimeLimitTransform, wrap_transforms)
from phossim.filtering.preprocessing import GrayscaleTransform, GrayscaleConfig
from phossim.filtering.edge import CannyFilter, CannyConfig
from phossim.phosphene_simulation.basic import (PhospheneSimulation,
                                                PhospheneSimulationConfig)
from phossim.recording import RecordingConfig, RecordingTransform
from phossim.agent.stable_baselines import (get_agent, TrainingConfig,
                                            StableBaselineAgentConfig)
from phossim.rendering import (DisplayConfig, ScreenDisplay, DisplayList,
                               Display)


@dataclass
class Config:
    environment_config: HallwayConfig
    transforms: List[Tuple[Type[Transform], TransformConfig]]
    agent_config: StableBaselineAgentConfig
    displays: List[Display]
    device: Optional[str] = 'cpu'


class Pipeline(BasePipeline):
    def __init__(self, config: Config):
        super().__init__()
        self.environment = Hallway(config.environment_config)
        self.environment = wrap_transforms(self.environment, config.transforms)
        self.agent = get_agent(self.environment, config.agent_config)
        self.renderer = DisplayList(config.displays)


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '4'
    device = 'cuda:0'
    input_key = 'input'
    filter_key = 'filtered_observation'
    phosphene_key = 'phosphenes'
    path_base = Path('~/Data/phosphenes/hallway').expanduser()
    path_recording = path_base.joinpath('recording')
    path_tensorboard = path_base.joinpath('log')
    path_model = path_base.joinpath('models/A2C_hallway')
    path_model.parent.mkdir(parents=True, exist_ok=True)
    video_length = 300
    def recording_trigger(episode): return episode % 10000 == 0

    size = 128
    shape = (size, size, 3)
    shape_gray = (size, size, 1)
    observation_space = gym.spaces.Box(low=0, high=255,
                                       shape=shape, dtype=np.uint8)
    observation_space_gray = gym.spaces.Box(low=0, high=255,
                                            shape=shape_gray, dtype=np.uint8)

    environment_config = HallwayConfig(observation_space, size=size)

    transforms = [
        (Transform, TransformConfig(input_key)),
        (RecordingTransform, RecordingConfig(path_recording,
                                             episode_trigger=recording_trigger,
                                             video_length=video_length,
                                             name_prefix='input')),
        (GrayscaleTransform, GrayscaleConfig(None, observation_space_gray)),
        (CannyFilter, CannyConfig(filter_key, sigma=1)),
        (RecordingTransform, RecordingConfig(path_recording,
                                             episode_trigger=recording_trigger,
                                             video_length=video_length,
                                             name_prefix='filtered')),
        (PhospheneSimulation, PhospheneSimulationConfig(phosphene_key)),
        (RecordingTransform, RecordingConfig(path_recording,
                                             episode_trigger=recording_trigger,
                                             video_length=video_length,
                                             name_prefix='phosphenes')),
        (TimeLimitTransform, TimeLimitConfig()),
        (MonitorTransform, MonitorConfig())
    ]

    agent_config = StableBaselineAgentConfig(
        path_model, 'A2C', 'CnnPolicy', {'tensorboard_log': path_tensorboard})

    displays = [
         ScreenDisplay(DisplayConfig(input_key, input_key, 'hallway')),
         ScreenDisplay(DisplayConfig(filter_key, filter_key, 'canny')),
         ScreenDisplay(DisplayConfig(phosphene_key, phosphene_key, 'basic')),
    ]

    config = Config(environment_config,
                    transforms,
                    agent_config,
                    displays,
                    device)

    pipeline = Pipeline(config)

    training_config = TrainingConfig(int(1e6))
    pipeline.agent.learn(**training_config.asdict())
    pipeline.agent.save(config.agent_config.path_model)

    pipeline.run()


if __name__ == '__main__':
    main()
    sys.exit()
