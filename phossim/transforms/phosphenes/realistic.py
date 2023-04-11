from typing import Tuple, Optional

import numpy as np
from dataclasses import dataclass

import gym
import os

import pathlib
import torch
from dynaphos.cortex_models import \
    get_visual_field_coordinates_from_cortex_full
from dynaphos.simulator import GaussianSimulator
from dynaphos.utils import (load_params, to_numpy, load_coordinates_from_yaml,
                            Map)
# Dynaphos is a library developed in Neural Coding lab, check paper and git repository.

from phossim.transforms.common import Transform, TransformConfig


@dataclass
class PhospheneSimulationConfig(TransformConfig):
    resolution: Tuple[int, int]
    num_electrodes: int
    intensity_decay: float = 0.4
    dtype: torch.dtype = torch.float32
    device: str = 'cpu'
    batch_size: Optional[int] = 0  # Use no batch dimension.


class PhospheneSimulation(Transform):
    def __init__(self, env: gym.Env, config: PhospheneSimulationConfig):
        super().__init__(env, config)  # init Transform class, which initialize gym.ObservationWrapper with env

        self._observation_space = gym.spaces.Box(
            low=0, high=255, shape=config.resolution + (1,), dtype=np.uint8)
        path_module = pathlib.Path(__file__).parent.resolve()
        params = load_params(os.path.join(path_module, 'params.yaml'))
        params['thresholding']['use_threshold'] = False
        params['run']['resolution'] = config.resolution
        params['display']['screen_resolution'] = config.resolution
        coordinates_cortex = load_coordinates_from_yaml(
            os.path.join(path_module, 'grid_coords_dipole_valid.yaml'),
            n_coordinates=config.num_electrodes)
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


class PhospheneSimulationTorch(torch.nn.Module):
    def __init__(self, config: PhospheneSimulationConfig):
        super().__init__()

        path_module = pathlib.Path(__file__).parent.resolve()
        params = load_params(os.path.join(path_module, 'params.yaml'))
        params['thresholding']['use_threshold'] = False
        params['run']['batch_size'] = config.batch_size
        params['run']['resolution'] = config.resolution
        coordinates_cortex = load_coordinates_from_yaml(
            os.path.join(path_module, 'grid_coords_dipole_valid.yaml'),
            n_coordinates=config.num_electrodes)
        coordinates_cortex = Map(*coordinates_cortex)
        coordinates_visual_field = \
            get_visual_field_coordinates_from_cortex_full(
                params['cortex_model'], coordinates_cortex)
        self.sim = GaussianSimulator(params, coordinates_visual_field)

    def forward(self, x):
        """
        Return phosphenes (2d) based on current stimulation and previous state.
        """

        # Need to reset for each forward pass, otherwise torch will throw an
        # error when computing the gradient (the simulator is stateful, but
        # each gradient computation discards the graph by default, so the next
        # call would not find older states.) The reset here is problematic
        # because it removes any temporal component of the phosphene model. In
        # a supervised learning scenario, one could do a forward pass on a
        # sequence of frames, do the backward pass, and then reset. In RL, one
        # usually presents samples randomly. Would need to use an RL
        # implementation for RNN policies, which train on sequences.
        self.sim.reset()

        phosphenes = self.sim(x)
        return torch.atleast_3d(phosphenes)
