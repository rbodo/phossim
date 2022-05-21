import sys
from pathlib import Path

from phossim.interface import Transform, TransformConfig
from phossim.pipeline import evaluate, train
from phossim.config import Config
from phossim.implementation.environment.neurosmash import \
    NeurosmashConfig, get_neurosmash_environment
from phossim.implementation.agent.stable_baselines import get_agent, \
    StableBaselineAgentConfig, TrainingConfig
from phossim.utils import RecordingConfig, RecordingTransform
from phossim.rendering import DisplayConfig, ScreenDisplay


if __name__ == '__main__':
    input_key = 'input'
    filter_key = 'filtered_observation'
    phosphene_key = 'phosphenes'
    path_base = Path('~/Data/phosphenes/neurosmash').expanduser()
    path_recording = path_base.joinpath('recording')
    path_tensorboard = path_base.joinpath('log')
    path_model = path_base.joinpath('models/A2C_neurosmash')
    path_model.parent.mkdir(exist_ok=True)
    video_length = 300
    def recording_trigger(episode): return episode % 10000 == 0

    environment_config = NeurosmashConfig()

    transform_configs = [
        (Transform, TransformConfig(input_key)),
        (RecordingTransform, RecordingConfig(path_recording,
                                             episode_trigger=recording_trigger,
                                             video_length=video_length,
                                             name_prefix='input')),
    ]

    agent_config = StableBaselineAgentConfig(
        path_model, 'A2C', 'CnnPolicy', {'tensorboard_log': path_tensorboard})

    display_configs = [
         (ScreenDisplay, DisplayConfig(input_key, input_key, 'neurosmash')),
    ]

    training_config = TrainingConfig(int(1e6))

    config = Config(environment_getter=get_neurosmash_environment,
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
