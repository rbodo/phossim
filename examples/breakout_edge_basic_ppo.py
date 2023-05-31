import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Type, Optional

from phossim.pipeline import BasePipeline
from phossim.environment.openai_gym import (GymConfig, AtariConfig,
                                            get_atari_environment)
from phossim.transforms.common import (
    Transform, TransformConfig, wrap_transforms, RecordingConfig,
    RecordingTransform, TimeLimitTransform, TimeLimitConfig, MonitorTransform,
    MonitorConfig)
from phossim.transforms.edge import CannyFilter, CannyConfig
from phossim.transforms.phosphenes.basic import (PhospheneSimulation,
                                                 PhospheneSimulationConfig)
from phossim.agent.stable_baselines import (get_agent, TrainingConfig,
                                            AgentConfig)
from phossim.rendering import Viewer, ViewerList, ViewerConfig


@dataclass
class Config:
    environment_config: AtariConfig
    transforms: List[Tuple[Type[Transform], TransformConfig]]
    agent_config: AgentConfig
    displays: List[Viewer]
    device: Optional[str] = 'cpu'


class Pipeline(BasePipeline):
    def __init__(self, config: Config):
        super().__init__()
        self.environment = get_atari_environment(config.environment_config)
        self.environment = wrap_transforms(self.environment, config.transforms)
        self.agent = get_agent(self.environment, config.agent_config)
        self.renderer = ViewerList(config.displays)


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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

    def recording_trigger(episode): return episode % 1000 == 0

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
        path_model, 'PPO', 'MlpPolicy', {'tensorboard_log': path_tensorboard})

    displays = [
        Viewer(ViewerConfig(input_key, 'input')),
        Viewer(ViewerConfig(filter_key, 'canny')),
        Viewer(ViewerConfig(phosphene_key, 'phosphenes')),
    ]

    config = Config(environment_config,
                    transforms,
                    agent_config,
                    displays,
                    device)

    pipeline = Pipeline(config)

    training_config = TrainingConfig(int(1e2))
    pipeline.agent.learn(**training_config.asdict())
    pipeline.agent.save(config.agent_config.path_model)

    pipeline.run()


if __name__ == '__main__':
    main()
    sys.exit()
