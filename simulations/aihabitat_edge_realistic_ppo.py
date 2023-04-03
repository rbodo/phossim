# System imports
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Type, Optional

# Third party libraries (Habitat)
from habitat.sims.habitat_simulator.actions import HabitatSimActions

# Project imports
from phossim.pipeline import InteractivePipeline
from phossim.pipeline import BasePipeline
from phossim.environment.aihabitat import AihabitatConfig, Aihabitat
from phossim.transforms.common import (
    Transform, TransformConfig, wrap_transforms, ResizeTransform, ResizeConfig,
    RecordingTransform, RecordingConfig, GrayscaleTransform, GrayscaleConfig,
    VrDisplayTransform, VrDisplayConfig, TimeLimitTransform, TimeLimitConfig, MonitorTransform,
    MonitorConfig)
from phossim.transforms.edge import CannyFilter, CannyConfig
from phossim.transforms.phosphenes.basic import (PhospheneSimulation,
                                                     PhospheneSimulationConfig)
from phossim.agent.human import HumanAgent, HumanAgentConfig
from phossim.agent.stable_baselines import (get_agent, TrainingConfig,
                                            AgentConfig)
from phossim.rendering import Viewer, ViewerList, ViewerConfig, ViewerListBlocking

@dataclass
class Config:
    environment_config: AihabitatConfig
    transforms: List[Tuple[Type[Transform], TransformConfig]]
    agent_config: AgentConfig
    displays: List[Viewer]
    device: Optional[str] = 'cpu'


class Pipeline(BasePipeline):
    def __init__(self, config: Config):
        super().__init__()
        self.environment = Aihabitat(config.environment_config)
        self.environment = wrap_transforms(self.environment, config.transforms)
        # self.renderer = ViewerListBlocking(config.displays)
        self.renderer = ViewerList(config.displays)
        self.agent = get_agent(self.environment, config.agent_config)


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    device = 'cuda:0'
    input_key = 'input'
    filter_key = 'filtered_observation'
    phosphene_key = 'phosphenes'

    path_base = Path('~/Data/phosphenes/aihabitat_ppo').expanduser()
    path_recording = path_base.joinpath('recording')
    path_tensorboard = path_base.joinpath('log')
    path_model = path_base.joinpath('models/PPO_breakout')
    path_config = 'benchmark/nav/pointnav/pointnav_habitat_test.yaml'  # VR environment?
    path_model.parent.mkdir(exist_ok=True)

    video_length = 300

    def recording_trigger(episode): return episode % 10000 == 0

    num_phosphenes = 256
    shape_in = (256, 256, 3)
    shape_resized = (512, 512, 3)
    shape_gray = (512, 512, 1)
    shape_vr = (1600, 2880, 1)

    environment_config = AihabitatConfig(shape_in, path_config)

    transforms = [
        (Transform, TransformConfig(input_key)),
        #(ResizeTransform, ResizeConfig(None, shape_resized)),
        (RecordingTransform, RecordingConfig(path_recording,
                                             episode_trigger=recording_trigger,
                                             video_length=video_length,
                                             name_prefix='input')),
        # (GrayscaleTransform, GrayscaleConfig(None, shape_gray)),
        (CannyFilter, CannyConfig(filter_key, sigma=1)),
        (RecordingTransform, RecordingConfig(path_recording,
                                             episode_trigger=recording_trigger,
                                             video_length=video_length,
                                             name_prefix='filtered')),
        (PhospheneSimulation, PhospheneSimulationConfig(phosphene_key)), # Basic
        #(PhospheneSimulation, PhospheneSimulationConfig(phosphene_key, shape_gray[:-1], num_phosphenes)), # Realistic
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

    training_config = TrainingConfig(int(1e7))
    pipeline.agent.learn(**training_config.asdict())
    pipeline.agent.save(config.agent_config.path_model)

    pipeline.run()


if __name__ == '__main__':
    main()
    sys.exit()



