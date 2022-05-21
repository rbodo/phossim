import sys
from pathlib import Path

import gym
import numpy as np

from phossim.implementation.filtering.edge import CannyFilter, CannyConfig
from phossim.implementation.filtering.preprocessing import \
    GrayscaleTransform, GrayscaleConfig
from phossim.implementation.phosphene_simulation.basic import \
    PhospheneSimulationBasic, BasicPhospheneSimulationConfig
from phossim.interface import Transform, TransformConfig
from phossim.pipeline import evaluate, train
from phossim.config import Config
from phossim.implementation.environment.hallway import HallwayConfig, \
    HallwayEnv
from phossim.implementation.agent.stable_baselines import get_agent, \
    StableBaselineAgentConfig, TrainingConfig
from phossim.utils import RecordingConfig, RecordingTransform
from phossim.rendering import DisplayConfig, ScreenDisplay


if __name__ == '__main__':
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

    transform_configs = [
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
        (PhospheneSimulationBasic,
         BasicPhospheneSimulationConfig(phosphene_key,
                                        image_size=(size, size))),
        (RecordingTransform, RecordingConfig(path_recording,
                                             episode_trigger=recording_trigger,
                                             video_length=video_length,
                                             name_prefix='phosphenes')),
    ]

    agent_config = StableBaselineAgentConfig(
        path_model, 'A2C', 'CnnPolicy', {'tensorboard_log': path_tensorboard})

    display_configs = [
         (ScreenDisplay, DisplayConfig(input_key, input_key, 'hallway')),
         (ScreenDisplay, DisplayConfig(filter_key, filter_key, 'canny')),
         (ScreenDisplay, DisplayConfig(phosphene_key, phosphene_key, 'basic')),
    ]

    training_config = TrainingConfig(int(1e6))

    config = Config(environment_getter=HallwayEnv,
                    agent_getter=get_agent,
                    environment_config=environment_config,
                    transform_configs=transform_configs,
                    agent_config=agent_config,
                    display_configs=display_configs,
                    training_config=training_config,
                    )

    train(config)

    evaluate(config)

    sys.exit()
