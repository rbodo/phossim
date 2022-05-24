from dataclasses import dataclass

import gym
import numpy as np
import torch

from phossim.transforms import Transform, TransformConfig


@dataclass
class PhospheneSimulationConfig(TransformConfig):
    observation_space: gym.Space
    intensity_decay: float = 0.4
    dtype: torch.dtype = torch.float32
    device: str = 'cpu'


class PhospheneSimulation(Transform):
    def __init__(self, env: gym.Env, config: PhospheneSimulationConfig):
        super().__init__(env, config)

        self._observation_space = config.observation_space
        self.intensity_decay = config.intensity_decay
        self.dtype = config.dtype
        self.device = config.device
        num_phosphenes = self.env.observation_space.shape[0]
        self.neural_activation = torch.zeros(num_phosphenes, dtype=self.dtype,
                                             device=self.device)
        self.gaussian_filters = None

    def get_gaussian_filters(self):
        """Generate gaussian activation maps, based on sigmas and phosphene
        mapping."""

        phosphene_sizes = self.env.phosphene_sizes
        phosphene_map = self.env.phosphene_map

        alpha = 1 / (phosphene_sizes * np.sqrt(np.pi))
        beta = 1 / (2 * phosphene_sizes ** 2)

        return torch.exp(-phosphene_map ** 2 * beta) * alpha

    def _update(self, stimulus_pattern):
        """Adjust state as function of previous state and current stimulation.
        """

        # TODO: adjust temporal properties here
        self.neural_activation = \
            stimulus_pattern + self.intensity_decay * self.neural_activation

        # TODO: adjust temporal properties here
        # self.phosphene_sizes = self.phosphene_sizes

    def observation(self, observation):
        """Return phosphenes (2d) based on current stimulation and previous
        state (self.neural_activation, self.phosphene_sizes)."""

        # Update current state according to current stimulation and previous
        # state.
        self._update(observation)

        # Todo: If phosphene properties change over time, the gaussian filters
        #       need to be updated every time the simulator is called.
        if self.gaussian_filters is None:
            self.gaussian_filters = self.get_gaussian_filters()

        # Generate phosphenes by summing across gaussians.
        phosphenes = torch.tensordot(self.neural_activation,
                                     self.gaussian_filters, 1)

        phosphenes = phosphenes.clip(0, 500)
        phosphenes = 255 * phosphenes / phosphenes.max()

        return np.atleast_3d(phosphenes.cpu().numpy().astype('uint8'))
