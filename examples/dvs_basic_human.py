import sys
from pathlib import Path

import gym
import numpy as np

from phossim.implementation.agent.human import HumanAgentConfig, HumanAgent
from phossim.implementation.environment.camera import DVSConfig, DVSFrameStream
from phossim.implementation.phosphene_simulation.basic import \
    PhospheneSimulationBasic, BasicPhospheneSimulationConfig
from phossim.interface import Transform, TransformConfig
from phossim.pipeline import evaluate
from phossim.config import Config
from phossim.utils import RecordingConfig, RecordingTransform
from phossim.rendering import DisplayConfig, ScreenDisplay


if __name__ == '__main__':
    input_key = 'input'
    filter_key = 'filtered_observation'
    phosphene_key = 'phosphenes'
    path_base = Path('~/Data/phosphenes/dvs_human').expanduser()
    path_recording = path_base.joinpath('recording')
    path_tensorboard = path_base.joinpath('log')
    video_length = 300
    def recording_trigger(episode): return episode % 10000 == 0

    shape = (346, 280, 1)
    observation_space_in = gym.spaces.Box(low=0, high=255, shape=shape,
                                          dtype=np.uint8)

    environment_config = DVSConfig(observation_space_in)

    transform_configs = [
        (Transform, TransformConfig(input_key)),
        (RecordingTransform, RecordingConfig(path_recording,
                                             episode_trigger=recording_trigger,
                                             video_length=video_length,
                                             name_prefix='input')),
        (PhospheneSimulationBasic,
         BasicPhospheneSimulationConfig(phosphene_key, image_size=shape[:-1])),
        (RecordingTransform, RecordingConfig(path_recording,
                                             episode_trigger=recording_trigger,
                                             video_length=video_length,
                                             name_prefix='phosphenes')),
    ]

    agent_config = HumanAgentConfig({})

    display_configs = [
         (ScreenDisplay, DisplayConfig(input_key, input_key, 'dvs')),
         (ScreenDisplay, DisplayConfig(phosphene_key, phosphene_key, 'basic')),
    ]

    config = Config(environment_getter=DVSFrameStream,
                    agent_getter=HumanAgent,
                    environment_config=environment_config,
                    transform_configs=transform_configs,
                    agent_config=agent_config,
                    display_configs=display_configs,
                    )

    evaluate(config)

    sys.exit()
