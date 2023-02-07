import numpy as np
from dataclasses import dataclass

import gym
import os
import torch
from dynaphos.cortex_models import \
    get_visual_field_coordinates_from_cortex_full
from dynaphos.simulator import GaussianSimulator
from dynaphos.utils import (load_params, to_numpy, load_coordinates_from_yaml,
                            Map)

from phossim import phosphene_simulation
from phossim.transforms import Transform, TransformConfig


@dataclass
class PhospheneSimulationConfig(TransformConfig):
    observation_space: gym.Space
    num_phosphenes: int
    intensity_decay: float = 0.4
    dtype: torch.dtype = torch.float32
    device: str = 'cpu'


class PhospheneSimulation(Transform):
    def __init__(self, env: gym.Env, config: PhospheneSimulationConfig):
        super().__init__(env, config)

        self._observation_space = config.observation_space
        resolution = self._observation_space.shape[:-1]
        path_module = os.path.dirname(phosphene_simulation.__file__)
        params = load_params(os.path.join(path_module, 'params.yaml'))
        params['thresholding']['use_threshold'] = False
        params['run']['resolution'] = resolution
        params['display']['screen_resolution'] = resolution
        coordinates_cortex = load_coordinates_from_yaml(
            os.path.join(path_module, 'grid_coords_dipole_valid.yaml'),
            n_coordinates=config.num_phosphenes)
        coordinates_cortex = Map(*coordinates_cortex)
        coordinates_visual_field = \
            get_visual_field_coordinates_from_cortex_full(
                params['cortex_model'], coordinates_cortex)
        self.sim = GaussianSimulator(params, coordinates_visual_field)

    def observation(self, observation):
        """
        Return phosphenes (2d) based on current stimulation and previous state.
        """

        stimulus = self.sim.sample_stimulus(np.moveaxis(observation, -1, 0))
        phosphenes = self.sim(stimulus)
        phosphenes *= 255  # Has been clipped to 1 before.
        return np.atleast_3d(to_numpy(phosphenes)).astype('uint8')
