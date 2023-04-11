

# export PYTHONPATH="/scratch/big/home/carsan/Internship/PyCharm_projects/Phossim"

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Type, Optional

import gym
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.env_util import make_vec_env

from phossim.pipeline import BasePipeline
from phossim.environment.openai_gym import (GymConfig, AtariConfig,
                                            get_atari_environment)
from phossim.transforms.common import (
    Transform, TransformConfig, wrap_transforms, RecordingConfig,
    RecordingTransform, TimeLimitTransform, TimeLimitConfig, MonitorTransform,
    MonitorConfig)
from phossim.transforms.e2e import AutoencoderFilter, AutoencoderConfig
from phossim.transforms.phosphenes.realistic import (
    PhospheneSimulationTorch, PhospheneSimulationConfig)
from phossim.agent.e2e import (TrainingConfig, AgentConfig, E2ePPO,
                               PhospheneEncoderDecoder, Encoder, Decoder)
from phossim.rendering import Viewer, ViewerConfig, ViewerList

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


@dataclass
class Config:
    environment: gym.Env
    transforms: List[Tuple[Type[Transform], TransformConfig]]
    agent: BaseAlgorithm
    viewers: List[Viewer]
    device: Optional[str] = 'cpu'


class Pipeline(BasePipeline):
    def __init__(self, config: Config):
        super().__init__()
        self.environment = config.environment
        self.environment = wrap_transforms(self.environment, config.transforms)
        self.agent = config.agent
        self.renderer = ViewerList(config.viewers)


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    device = 'cuda:1'
    input_key = 'input'
    filter_key = 'filtered_observation'
    phosphene_key = 'phosphenes'
    path_base = Path('~/Data/phosphenes/atari').expanduser()
    path_recording = path_base.joinpath('recording')
    path_tensorboard = path_base.joinpath('log')
    path_model = path_base.joinpath('models/PPO_breakout')
    path_model.parent.mkdir(exist_ok=True)
    video_length = 300
    num_coordinates = 256
    batch_size = 64
    def recording_trigger(episode): return episode % 10000 == 0

    shape_input = (84, 84, 1)
    shape_phosphenes = (168, 168)

    environment_config = AtariConfig(
        GymConfig('ALE/Breakout-v5',
                  {'render_mode': 'rgb_array',
                   'frameskip': 1,  # Handled by AtariWrapper
                   'repeat_action_probability': 0,
                   'full_action_space': False}))

    # Training
    ##########

    phosphene_simulator = PhospheneSimulationTorch(PhospheneSimulationConfig(
        phosphene_key, shape_phosphenes, num_coordinates,
        batch_size=batch_size))
    num_electrodes = phosphene_simulator.sim.num_phosphenes
    agent_config = AgentConfig(Encoder(num_electrodes).to(device),
                               Decoder().to(device), phosphene_simulator,
                               device)

    # noinspection PyTypeChecker
    environment_vec = make_vec_env(get_atari_environment, batch_size,
                                   env_kwargs={'config': environment_config})
    agent = E2ePPO('MlpPolicy', environment_vec, verbose=1, device=device,
                   batch_size=batch_size,
                   tensorboard_log=str(path_tensorboard), policy_kwargs={
                       'features_extractor_class': PhospheneEncoderDecoder,
                       'features_extractor_kwargs': {'config': agent_config}})

    training_config = TrainingConfig(int(1e7))
    agent.learn(**training_config.asdict())
    agent.save(path_model)

    # Evaluation
    ############

    # Create single (not vectorized) environment for evaluation.
    environment = get_atari_environment(environment_config)

    # Load trained agent.
    agent = E2ePPO.load(path_model, environment, device)

    transforms = [
        (Transform, TransformConfig(input_key)),
        (RecordingTransform, RecordingConfig(path_recording,
                                             episode_trigger=recording_trigger,
                                             video_length=video_length,
                                             name_prefix='input')),
        (AutoencoderFilter, AutoencoderConfig(filter_key, shape_input,
                                              Encoder(num_coordinates),
                                              Decoder())),
        (RecordingTransform, RecordingConfig(path_recording,
                                             episode_trigger=recording_trigger,
                                             video_length=video_length,
                                             name_prefix='filtered')),
        (TimeLimitTransform, TimeLimitConfig()),
        (MonitorTransform, MonitorConfig())
    ]

    displays = [
         Viewer(ViewerConfig(input_key, 'gym')),
         Viewer(ViewerConfig(filter_key, 'e2e')),
         Viewer(ViewerConfig(phosphene_key, 'basic')),
    ]

    config = Config(environment,
                    transforms,
                    agent,
                    displays,
                    device)
    pipeline = Pipeline(config)
    pipeline.run()


def print_model_summary(agent, agent_config, batch_size, shape_input):
    from torchinfo import summary
    
    p = PhospheneEncoderDecoder(agent.observation_space, agent_config)
    summary(p, input_size=(batch_size,)+agent.observation_space.shape,
            col_names=['input_size', 'output_size'])
    summary(agent.policy, input_size=(batch_size, 1) + shape_input[:-1],
            col_names=['input_size', 'output_size'])
        
        
if __name__ == '__main__':
    main()
    sys.exit()
