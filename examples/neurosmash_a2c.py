import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Type, Optional

from phossim.environment.neurosmash import Neurosmash, NeurosmashConfig
from phossim.pipeline import BasePipeline
from phossim.transforms import (Transform, TransformConfig, TimeLimitConfig,
                                MonitorConfig, MonitorTransform,
                                TimeLimitTransform, wrap_transforms)
from phossim.recording import RecordingConfig, RecordingTransform
from phossim.agent.stable_baselines import (get_agent, TrainingConfig,
                                            AgentConfig)
from phossim.rendering import (DisplayConfig, ScreenDisplay, DisplayList,
                               Display)


@dataclass
class Config:
    environment_config: NeurosmashConfig
    transforms: List[Tuple[Type[Transform], TransformConfig]]
    agent_config: AgentConfig
    displays: List[Display]
    device: Optional[str] = 'cpu'


class Pipeline(BasePipeline):
    def __init__(self, config: Config):
        super().__init__()
        self.environment = Neurosmash(config.environment_config)
        self.environment = wrap_transforms(self.environment, config.transforms)
        self.agent = get_agent(self.environment, config.agent_config)
        self.renderer = DisplayList(config.displays)


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '4'
    device = 'cuda:0'
    input_key = 'input'
    path_base = Path('~/Data/phosphenes/neurosmash').expanduser()
    path_recording = path_base.joinpath('recording')
    path_tensorboard = path_base.joinpath('log')
    path_model = path_base.joinpath('models/A2C_neurosmash')
    path_model.parent.mkdir(exist_ok=True)
    video_length = 300
    def recording_trigger(episode): return episode % 10000 == 0

    environment_config = NeurosmashConfig()

    transforms = [
        (Transform, TransformConfig(input_key)),
        (RecordingTransform, RecordingConfig(path_recording,
                                             episode_trigger=recording_trigger,
                                             video_length=video_length,
                                             name_prefix='input')),
        (TimeLimitTransform, TimeLimitConfig()),
        (MonitorTransform, MonitorConfig())
    ]

    agent_config = AgentConfig(
        path_model, 'A2C', 'CnnPolicy', {'tensorboard_log': path_tensorboard})

    displays = [
         ScreenDisplay(DisplayConfig(input_key, input_key, 'neurosmash')),
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
