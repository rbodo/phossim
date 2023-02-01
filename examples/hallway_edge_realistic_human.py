import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Type, Optional

import gym
import numpy as np
from pyglet.window import key

from phossim.pipeline import BasePipeline
from phossim.environment.hallway import HallwayConfig, Hallway
from phossim.transforms import Transform, TransformConfig, wrap_transforms
from phossim.filtering.preprocessing import (GrayscaleTransform, ResizeConfig,
                                             GrayscaleConfig, ResizeTransform)
from phossim.filtering.edge import CannyFilter, CannyConfig
from phossim.phosphene_simulation.realistic import (PhospheneSimulation,
                                                    PhospheneSimulationConfig)
from phossim.recording import RecordingConfig, RecordingTransform
from phossim.agent.human import HumanAgent, HumanAgentConfig
from phossim.rendering import (Viewer, ViewerConfig, ViewerList, VRViewer,
                               VRViewerConfig)


@dataclass
class Config:
    environment_config: HallwayConfig
    transforms: List[Tuple[Type[Transform], TransformConfig]]
    agent_config: HumanAgentConfig
    viewers: List[Viewer]
    vr_viewer: VRViewer
    device: Optional[str] = 'cpu'


class Pipeline(BasePipeline):
    def __init__(self, config: Config):
        super().__init__()
        self.environment = Hallway(config.environment_config)
        self.environment = wrap_transforms(self.environment, config.transforms)
        self.agent = HumanAgent(self.environment, config.vr_viewer,
                                config.agent_config)
        self.renderer = ViewerList(config.viewers)


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '4'
    device = 'cuda:0'
    input_key = 'input'
    filter_key = 'filtered_observation'
    phosphene_key = 'phosphenes'
    path_base = Path('~/Data/phosphenes/hallway_human').expanduser()
    path_recording = path_base.joinpath('recording')
    video_length = 300
    def recording_trigger(episode): return episode % 10000 == 0

    num_phosphenes = 256
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
    observation_space_phosphenes = gym.spaces.Box(low=0, high=255,
                                                  shape=shape_gray,
                                                  dtype=np.uint8)

    environment_config = HallwayConfig(observation_space_in, size=size_in)

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
        (PhospheneSimulation,
         PhospheneSimulationConfig(phosphene_key, observation_space_phosphenes,
                                   num_phosphenes)),
        (RecordingTransform, RecordingConfig(path_recording,
                                             episode_trigger=recording_trigger,
                                             video_length=video_length,
                                             name_prefix='phosphenes')),
    ]

    agent_config = HumanAgentConfig({key.W: 0, key.A: 1, key.D: 2})

    displays = [
         Viewer(ViewerConfig(shape_in, input_key, input_key, 'hallway')),
         Viewer(ViewerConfig(shape_gray, filter_key, filter_key, 'canny')),
         Viewer(ViewerConfig(shape_gray, phosphene_key, phosphene_key,
                             'basic')),
    ]

    vr_display = VRViewer(VRViewerConfig((1600, 2880, 1), 'phosphenes_vr',
                                         phosphene_key))

    config = Config(environment_config,
                    transforms,
                    agent_config,
                    displays,
                    vr_display,
                    device)

    pipeline = Pipeline(config)

    pipeline.run()


if __name__ == '__main__':
    main()
    sys.exit()
