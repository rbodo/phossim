import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Type, Optional

import gym
import numpy as np

from phossim.pipeline import BasePipeline
from phossim.environment.camera import CameraConfig, CameraStream
from phossim.transforms.common import (
    Transform, TransformConfig, wrap_transforms, ResizeConfig, RecordingConfig,
    RecordingTransform, GrayscaleTransform, GrayscaleConfig, ResizeTransform)
from phossim.transforms.edge import CannyFilter, CannyConfig
from phossim.transforms.phosphenes.basic import (PhospheneSimulation,
                                                 PhospheneSimulationConfig)
from phossim.agent.human import HumanAgentConfig, HumanAgent
from phossim.rendering import Viewer, ViewerConfig, ViewerList


@dataclass
class Config:
    environment_config: CameraConfig
    transforms: List[Tuple[Type[Transform], TransformConfig]]
    agent_config: HumanAgentConfig
    viewers: List[Viewer]
    device: Optional[str] = 'cpu'


class Pipeline(BasePipeline):
    def __init__(self, config: Config):
        super().__init__()
        self.environment = CameraStream(config.environment_config)
        self.environment = wrap_transforms(self.environment, config.transforms)
        self.agent = HumanAgent(self.environment, config.agent_config)
        self.renderer = ViewerList(config.viewers)


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '4'
    device = 'cuda:0'
    input_key = 'input'
    filter_key = 'filtered_observation'
    phosphene_key = 'phosphenes'
    path_base = Path('~/Data/phosphenes/dvs_human').expanduser()
    path_recording = path_base.joinpath('recording')
    video_length = 300
    def recording_trigger(episode): return episode % 10000 == 0

    size_in = 128
    size_out = 512
    shape_in = (size_in, size_in, 3)
    shape_resized = (size_out, size_out, 3)
    shape_gray = (size_out, size_out, 1)
    observation_space_in = gym.spaces.Box(low=0, high=255, shape=shape_in,
                                          dtype=np.uint8)
    observation_space_resized = gym.spaces.Box(low=0, high=255,
                                               shape=shape_resized,
                                               dtype=np.uint8)
    observation_space_gray = gym.spaces.Box(low=0, high=255, shape=shape_gray,
                                            dtype=np.uint8)

    environment_config = CameraConfig(observation_space_in)

    transforms = [
        (Transform, TransformConfig(input_key)),
        (ResizeTransform, ResizeConfig(None, observation_space_resized)),
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
    ]

    agent_config = HumanAgentConfig({}, (1600, 2880, 1))

    displays = [
         Viewer(ViewerConfig(input_key, 'hallway')),
         Viewer(ViewerConfig(filter_key, 'canny')),
         Viewer(ViewerConfig(phosphene_key, 'basic')),
    ]

    config = Config(environment_config,
                    transforms,
                    agent_config,
                    displays,
                    device)

    pipeline = Pipeline(config)

    pipeline.run()


if __name__ == '__main__':
    main()
    sys.exit()
