from dataclasses import dataclass

import gym
import numpy as np
import torch

from phossim.transforms import Transform, TransformConfig
from phossim.phosphene_simulation.realistic import get_phosphene_map


@dataclass
class IdentityConfig(TransformConfig):
    pass


class IdentityStimulusGenerator(Transform):
    def observation(self, observation):
        return observation


class RealisticStimulusGenerator(Transform):
    def __init__(self, env: gym.Env, config: TransformConfig):
        super().__init__(env, config)
        self.receptive_field_size = config.RECEPTIVE_FIELD_SIZE
        self.use_relative_receptive_field_size = \
            config.USE_RELATIVE_RECEPTIVE_FIELD_SIZE

        num_phosphenes = np.prod(config.PHOSPHENE_RESOLUTION)
        self.phosphene_map, self.phosphene_sizes = \
            get_phosphene_map(num_phosphenes, config.IMAGE_SIZE)
        config.phosphene_map = self.phosphene_map
        config.phosphene_sizes = self.phosphene_sizes

        self.electrode_map = self.get_electrode_map()

    def get_electrode_map(self):
        """Generate electrode map."""

        rf = self.receptive_field_size
        if self.use_relative_receptive_field_size:
            rf *= self.phosphene_sizes

        return torch.less(self.phosphene_map, rf).float()

    def observation(self, observation):
        """Create stimulation vector from sensor signal via electrode map."""

        sensor_signal = torch.tensor(observation, dtype=self.config.DTYPE,
                                     device=self.config.DEVICE)

        return torch.tensordot(self.electrode_map, sensor_signal)
