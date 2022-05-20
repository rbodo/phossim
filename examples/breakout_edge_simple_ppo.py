import sys
from pathlib import Path

from phossim.interface import Transform, TransformConfig
from phossim.pipeline import main
from phossim.config import Config
from phossim.implementation.environment.openai_gym.environment import GymConfig
from phossim.implementation.environment.openai_gym.atari.environment import \
    get_atari_environment, AtariConfig
from phossim.implementation.agent.stable_baselines import get_agent, \
    AgentConfig
from phossim.implementation.filtering.edge import CannyConfig, CannyFilter
from phossim.implementation.phosphene_simulation.basic import \
    BasicPhospheneSimulationConfig, PhospheneSimulationBasic
from phossim.utils import RecordingConfig, RecordingTransform
from phossim.rendering import DisplayConfig, ScreenDisplay


if __name__ == '__main__':
    input_key = 'input'
    filter_key = 'filtered_observation'
    phosphene_key = 'phosphenes'
    path_recording = Path('/home/rbodo/Data/phosphenes/atari/recording')
    video_length = 100
    def recording_trigger(step): return step == 1

    environment_config = AtariConfig(
        GymConfig('ALE/Breakout-v5',
                  {'render_mode': 'rgb_array',
                   'frameskip': 1,  # Handled by AtariWrapper
                   'repeat_action_probability': 0,
                   'full_action_space': True}))

    transform_configs = [
        (Transform, TransformConfig(input_key)),
        (RecordingTransform, RecordingConfig(path_recording,
                                             step_trigger=recording_trigger,
                                             video_length=video_length,
                                             name_prefix='input')),
        (CannyFilter, CannyConfig(filter_key, sigma=1)),
        (RecordingTransform, RecordingConfig(path_recording,
                                             step_trigger=recording_trigger,
                                             video_length=video_length,
                                             name_prefix='filtered')),
        (PhospheneSimulationBasic,
         BasicPhospheneSimulationConfig(phosphene_key, image_size=(84, 84))),
        (RecordingTransform, RecordingConfig(path_recording,
                                             step_trigger=recording_trigger,
                                             video_length=video_length,
                                             name_prefix='phosphenes')),
    ]

    agent_config = AgentConfig('PPO', 'MlpPolicy', {})

    display_configs = [
         (ScreenDisplay, DisplayConfig(input_key, input_key, 'gym')),
         (ScreenDisplay, DisplayConfig(filter_key, filter_key, 'identity')),
         (ScreenDisplay, DisplayConfig(phosphene_key, phosphene_key,
                                       'identity')),
    ]

    filepath_output_data = Path('/home/rbodo/Data/phosphenes/atari')

    config = Config(environment_getter=get_atari_environment,
                    agent_getter=get_agent,
                    environment_config=environment_config,
                    transform_configs=transform_configs,
                    agent_config=agent_config,
                    display_configs=display_configs,
                    filepath_output_data=filepath_output_data)

    main(config)

    sys.exit()
