import sys
from pathlib import Path

import gym
import numpy as np

from phossim.agent.human import HumanAgentConfig, HumanAgent
from phossim.filtering import CannyFilter, CannyConfig
from phossim.filtering import \
    GrayscaleTransform, GrayscaleConfig, ResizeTransform, ResizeConfig
from phossim.phosphene_simulation import \
    PhospheneSimulationBasic, BasicPhospheneSimulationConfig
from phossim.transforms import Transform, TransformConfig
from phossim.pipeline import evaluate
from phossim.config import Config
from phossim.environment import HallwayConfig, \
    HallwayEnv
from phossim.recording import RecordingConfig, RecordingTransform
from phossim.rendering import DisplayConfig, ScreenDisplay


if __name__ == '__main__':
    input_key = 'input'
    filter_key = 'filtered_observation'
    phosphene_key = 'phosphenes'
    path_base = Path('~/Data/phosphenes/hallway_human').expanduser()
    path_recording = path_base.joinpath('recording')
    path_tensorboard = path_base.joinpath('log')
    video_length = 300
    def recording_trigger(episode): return episode % 10000 == 0

    size_in = 128
    size_out = 512
    shape_in = (size_in, size_in, 3)
    shape_resized = (size_out, size_out, 3)
    shape_gray = (size_out, size_out, 1)
    observation_space_in = gym.spaces.Box(low=0, high=255,
                                          shape=shape_in, dtype=np.uint8)
    observation_space_resized = gym.spaces.Box(low=0, high=255,
                                               shape=shape_resized,
                                               dtype=np.uint8)
    observation_space_gray = gym.spaces.Box(low=0, high=255,
                                            shape=shape_gray, dtype=np.uint8)

    environment_config = HallwayConfig(observation_space_in, size=size_in)

    transform_configs = [
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
        (PhospheneSimulationBasic,
         BasicPhospheneSimulationConfig(phosphene_key,
                                        image_size=(size_out, size_out))),
        (RecordingTransform, RecordingConfig(path_recording,
                                             episode_trigger=recording_trigger,
                                             video_length=video_length,
                                             name_prefix='phosphenes')),
    ]

    agent_config = HumanAgentConfig({'w': 0, 'a': 1, 'd': 2})

    display_configs = [
         (ScreenDisplay, DisplayConfig(input_key, input_key, 'hallway')),
         (ScreenDisplay, DisplayConfig(filter_key, filter_key, 'canny')),
         (ScreenDisplay, DisplayConfig(phosphene_key, phosphene_key, 'basic')),
    ]

    config = Config(environment_getter=HallwayEnv,
                    agent_getter=HumanAgent,
                    environment_config=environment_config,
                    transform_configs=transform_configs,
                    agent_config=agent_config,
                    display_configs=display_configs,
                    )

    evaluate(config)

    sys.exit()
