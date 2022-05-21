import sys
from pathlib import Path

from phossim.interface import Transform, TransformConfig
from phossim.pipeline import evaluate, train
from phossim.config import Config
from phossim.implementation.environment.openai_gym import GymConfig, \
    AtariConfig, get_atari_environment
from phossim.implementation.agent.stable_baselines import get_agent, \
    StableBaselineAgentConfig, TrainingConfig
from phossim.implementation.filtering.edge import CannyConfig, CannyFilter
from phossim.implementation.phosphene_simulation.basic import \
    BasicPhospheneSimulationConfig, PhospheneSimulationBasic
from phossim.utils import RecordingConfig, RecordingTransform
from phossim.rendering import DisplayConfig, ScreenDisplay


if __name__ == '__main__':
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

    environment_config = AtariConfig(
        GymConfig('ALE/Breakout-v5',
                  {'render_mode': 'rgb_array',
                   'frameskip': 1,  # Handled by AtariWrapper
                   'repeat_action_probability': 0,
                   'full_action_space': True}))

    transform_configs = [
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
        (PhospheneSimulationBasic,
         BasicPhospheneSimulationConfig(phosphene_key, image_size=(84, 84))),
        (RecordingTransform, RecordingConfig(path_recording,
                                             episode_trigger=recording_trigger,
                                             video_length=video_length,
                                             name_prefix='phosphenes')),
    ]

    agent_config = StableBaselineAgentConfig(
        path_model, 'PPO', 'MlpPolicy', {'tensorboard_log': path_tensorboard})

    display_configs = [
         (ScreenDisplay, DisplayConfig(input_key, input_key, 'gym')),
         (ScreenDisplay, DisplayConfig(filter_key, filter_key, 'canny')),
         (ScreenDisplay, DisplayConfig(phosphene_key, phosphene_key, 'basic')),
    ]

    training_config = TrainingConfig(int(1e7))

    config = Config(environment_getter=get_atari_environment,
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
