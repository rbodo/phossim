import sys
from pathlib import Path

from phossim.implementation.environment.openai_gym.environment import GymConfig
from phossim.pipeline import main
from phossim.config import Config
from phossim.implementation.agent.stable_baselines import get_agent, \
    AgentConfig
from phossim.implementation.filtering.edge import CannyConfig, wrap_canny
from phossim.implementation.environment.openai_gym.atari.environment import \
    get_atari_environment, AtariConfig
from phossim.implementation.phosphene_simulation.basic import \
    BasicPhospheneSimulationConfig, wrap_phosphene_simulation
from phossim.implementation.stimulus_generation.stimulus_generation import \
    IdentityConfig, wrap_stimulus_generation
from phossim.utils import RecordingConfig


if __name__ == '__main__':
    image_size = (84, 84)
    config = Config(
        get_atari_environment,
        wrap_canny,
        wrap_stimulus_generation,
        wrap_phosphene_simulation,
        get_agent,
        AtariConfig(GymConfig('ALE/Breakout-v5',
                              {'render_mode': 'human',
                               'frameskip': 1,  # Handled by AtariWrapper
                               'repeat_action_probability': 0,
                               'full_action_space': True}),
                    {}),
        CannyConfig(sigma=3),
        IdentityConfig(),
        BasicPhospheneSimulationConfig(image_size),
        AgentConfig('PPO', 'MlpPolicy', {}),
        RecordingConfig(Path('/home/rbodo/Data/phosphenes/atari/recording')),
        filepath_output_data=Path('/home/rbodo/Data/phosphenes/atari'),
        image_size=image_size
    )

    main(config)

    sys.exit()
