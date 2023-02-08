import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Type, Optional

from phossim.pipeline import BasePipeline
from phossim.environment.hallway import HallwayConfig, Hallway
from phossim.transforms.common import (
    Transform, TransformConfig, wrap_transforms, RecordingConfig,
    RecordingTransform, GrayscaleTransform, GrayscaleConfig,
    TimeLimitTransform, TimeLimitConfig, MonitorTransform, MonitorConfig)
from phossim.transforms.edge import CannyFilter, CannyConfig
from phossim.transforms.phosphenes.basic import (PhospheneSimulation,
                                                 PhospheneSimulationConfig)
from phossim.agent.stable_baselines import (get_agent, TrainingConfig,
                                            AgentConfig)
from phossim.rendering import Viewer, ViewerConfig, ViewerList


@dataclass
class Config:
    environment_config: HallwayConfig
    transforms: List[Tuple[Type[Transform], TransformConfig]]
    agent_config: AgentConfig
    viewers: List[Viewer]
    device: Optional[str] = 'cpu'


class Pipeline(BasePipeline):
    def __init__(self, config: Config):
        super().__init__()
        self.environment = Hallway(config.environment_config)
        self.environment = wrap_transforms(self.environment, config.transforms)
        self.agent = get_agent(self.environment, config.agent_config)
        self.renderer = ViewerList(config.viewers)


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

    environment_config = HallwayConfig(shape)

    transforms = [
        (Transform, TransformConfig(input_key)),
        (RecordingTransform, RecordingConfig(path_recording,
                                             episode_trigger=recording_trigger,
                                             video_length=video_length,
                                             name_prefix='input')),
        (GrayscaleTransform, GrayscaleConfig(None, shape_gray)),
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

    agent_config = AgentConfig(
        path_model, 'A2C', 'CnnPolicy', {'tensorboard_log': path_tensorboard})

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

    training_config = TrainingConfig(int(1e6))
    pipeline.agent.learn(**training_config.asdict())
    pipeline.agent.save(config.agent_config.path_model)

    pipeline.run()


if __name__ == '__main__':
    main()
    sys.exit()
