import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Type, Optional

from phossim.pipeline import BasePipeline
from phossim.environment.camera import DVSConfig, DVSFrameStream
from phossim.transforms.common import (
    Transform, TransformConfig, wrap_transforms, RecordingConfig,
    RecordingTransform, VrDisplayTransform, VrDisplayConfig)
from phossim.transforms.phosphenes.basic import (PhospheneSimulation,
                                                 PhospheneSimulationConfig)
from phossim.agent.human import HumanAgentConfig, HumanAgent
from phossim.rendering import Viewer, ViewerConfig, ViewerList


@dataclass
class Config:
    environment_config: DVSConfig
    transforms: List[Tuple[Type[Transform], TransformConfig]]
    agent_config: HumanAgentConfig
    viewers: List[Viewer]
    device: Optional[str] = 'cpu'


class Pipeline(BasePipeline):
    def __init__(self, config: Config):
        super().__init__()
        self.environment = DVSFrameStream(config.environment_config)
        self.environment = wrap_transforms(self.environment, config.transforms)
        self.renderer = ViewerList(config.viewers)
        self.agent = HumanAgent(self.renderer, config.agent_config)


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    device = 'cuda:0'
    input_key = 'input'
    phosphene_key = 'phosphenes'
    vr_key = 'vr_display'
    path_base = Path('~/Data/phosphenes/dvs_human').expanduser()
    path_recording = path_base.joinpath('recording')
    video_length = 300
    def recording_trigger(episode): return episode % 10000 == 0

    shape = (260, 346, 1)
    shape_vr = (1600, 2880, 1)

    environment_config = DVSConfig(shape, port=6666)

    transforms = [
        (Transform, TransformConfig(input_key)),
        (RecordingTransform, RecordingConfig(path_recording,
                                             episode_trigger=recording_trigger,
                                             video_length=video_length,
                                             name_prefix='input')),
        (PhospheneSimulation, PhospheneSimulationConfig(phosphene_key)),
        (RecordingTransform, RecordingConfig(path_recording,
                                             episode_trigger=recording_trigger,
                                             video_length=video_length,
                                             name_prefix='phosphenes')),
        (VrDisplayTransform, VrDisplayConfig(vr_key, shape_vr))
    ]

    agent_config = HumanAgentConfig({})

    displays = [
        Viewer(ViewerConfig(input_key, 'dvs')),
        Viewer(ViewerConfig(phosphene_key, 'basic')),
        Viewer(ViewerConfig(vr_key, 'vr_display'))
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
