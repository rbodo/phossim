import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Type, Optional

from torch import nn

from phossim.pipeline import BasePipeline
from phossim.environment.openai_gym import (GymConfig, AtariConfig,
                                            get_atari_environment)
from phossim.transforms.common import (
    Transform, TransformConfig, wrap_transforms, RecordingConfig,
    RecordingTransform, TimeLimitTransform, TimeLimitConfig, MonitorTransform,
    MonitorConfig)
from phossim.transforms.e2e import AutoencoderFilter, AutoencoderConfig
from phossim.transforms.phosphenes.basic import (PhospheneSimulation,
                                                 PhospheneSimulationConfig)
from phossim.agent.e2e import TrainingConfig, AgentConfig, E2ePPO
from phossim.rendering import Viewer, ViewerConfig, ViewerList


@dataclass
class Config:
    environment_config: AtariConfig
    transforms: List[Tuple[Type[Transform], TransformConfig]]
    agent_config: AgentConfig
    viewers: List[Viewer]
    device: Optional[str] = 'cpu'


class Pipeline(BasePipeline):
    def __init__(self, config: Config):
        super().__init__()
        self.environment = get_atari_environment(config.environment_config)
        self.environment = wrap_transforms(self.environment, config.transforms)
        self.agent = E2ePPO(config.agent_config.policy_id, self.environment,
                            **config.agent_config.kwargs)
        self.renderer = ViewerList(config.viewers)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 1)

    def forward(self, x):
        return nn.ReLU()(self.conv(x))


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 1)

    def forward(self, x):
        return nn.ReLU()(self.conv(x))


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '4'
    device = 'cuda:0'
    input_key = 'input'
    filter_key = 'filtered_observation'
    phosphene_key = 'phosphenes'
    path_base = Path('~/Data/phosphenes/atari').expanduser()
    path_recording = path_base.joinpath('recording')
    path_tensorboard = path_base.joinpath('log')
    path_model = path_base.joinpath('models/PPO_breakout')
    path_model.parent.mkdir(exist_ok=True)
    video_length = 300
    def recording_trigger(episode): return episode % 10000 == 0

    shape = (84, 84, 1)

    environment_config = AtariConfig(
        GymConfig('ALE/Breakout-v5',
                  {'render_mode': 'rgb_array',
                   'frameskip': 1,  # Handled by AtariWrapper
                   'repeat_action_probability': 0,
                   'full_action_space': True}))

    transforms = [
        (Transform, TransformConfig(input_key)),
        (RecordingTransform, RecordingConfig(path_recording,
                                             episode_trigger=recording_trigger,
                                             video_length=video_length,
                                             name_prefix='input')),
        (AutoencoderFilter, AutoencoderConfig(filter_key, shape, Encoder(),
                                              Decoder())),
        (RecordingTransform, RecordingConfig(path_recording,
                                             episode_trigger=recording_trigger,
                                             video_length=video_length,
                                             name_prefix='filtered')),
        (PhospheneSimulation, PhospheneSimulationConfig(phosphene_key, shape)),
        (RecordingTransform, RecordingConfig(path_recording,
                                             episode_trigger=recording_trigger,
                                             video_length=video_length,
                                             name_prefix='phosphenes')),
        (TimeLimitTransform, TimeLimitConfig()),
        (MonitorTransform, MonitorConfig())
    ]

    agent_config = AgentConfig(
        path_model, 'MlpPolicy', {'tensorboard_log': path_tensorboard})

    displays = [
         Viewer(ViewerConfig(input_key, 'gym')),
         Viewer(ViewerConfig(filter_key, 'e2e')),
         Viewer(ViewerConfig(phosphene_key, 'basic')),
    ]

    config = Config(environment_config,
                    transforms,
                    agent_config,
                    displays,
                    device)

    pipeline = Pipeline(config)

    training_config = TrainingConfig(int(1e7))
    pipeline.agent.learn(**training_config.asdict())
    pipeline.agent.save(config.agent_config.path_model)

    pipeline.run()


if __name__ == '__main__':
    main()
    sys.exit()
