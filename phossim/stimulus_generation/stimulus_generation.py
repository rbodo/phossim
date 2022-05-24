from dataclasses import dataclass
from typing import Tuple

import gym
import numpy as np
import torch

from phossim.transforms import Transform, TransformConfig


@dataclass
class StimulusGeneratorConfig(TransformConfig):
    observation_space: gym.Space
    phosphene_resolution: Tuple[int, int]
    receptive_field_size: float = 4.0
    use_relative_receptive_field_size: bool = True
    dtype: torch.dtype = torch.float32
    device: str = 'cpu'


class StimulusGenerator(Transform):
    def __init__(self, env: gym.Env, config: StimulusGeneratorConfig):
        super().__init__(env, config)

        self._observation_space = config.observation_space
        self.phosphene_resolution = config.phosphene_resolution
        self.image_size = env.observation_space.shape[:-1]
        self.receptive_field_size = config.receptive_field_size
        self.use_relative_receptive_field_size = \
            config.use_relative_receptive_field_size
        self.dtype = config.dtype
        self.device = config.device

        self.phosphene_map = None
        self.phosphene_sizes = None
        self.electrode_map = None

        self.set_phosphene_map()
        self.set_electrode_map()

    def set_phosphene_map(self):
        num_phosphenes = int(np.prod(self.phosphene_resolution))
        h, w = self.image_size

        # Cartesian coordinate system for the visual field
        x = torch.arange(h, dtype=self.dtype, device=self.device)
        y = torch.arange(w, dtype=self.dtype, device=self.device)
        grid = torch.meshgrid(x, y, indexing='ij')

        # Polar coordinates
        d_min = min([h, w])
        phi = 2 * torch.pi * torch.rand(num_phosphenes, dtype=self.dtype,
                                        device=self.device)
        r = d_min / 2 * torch.rand(num_phosphenes, dtype=self.dtype,
                                   device=self.device) ** 2

        # Convert to cartesian indices
        xmax = w - 1
        ymax = h - 1
        x_offset = torch.round(r * torch.cos(phi) + xmax / 2)
        y_offset = torch.round(r * torch.sin(phi) + ymax / 2)
        x_offset = torch.clip(x_offset, 0, xmax)
        y_offset = torch.clip(y_offset, 0, ymax)
        x_offset = torch.reshape(x_offset, (-1, 1, 1))
        y_offset = torch.reshape(y_offset, (-1, 1, 1))

        # Calculate distance map for every element wrt center of phosphene
        self.phosphene_map = torch.sqrt(
            (torch.unsqueeze(grid[0], 0) - y_offset) ** 2 +
            (torch.unsqueeze(grid[1], 0) - x_offset) ** 2)

        # Sigma at start of simulation
        self.phosphene_sizes = get_eccentricity_scaling(r / d_min)
        self.phosphene_sizes = torch.reshape(self.phosphene_sizes, (-1, 1, 1))

    def set_electrode_map(self):
        """Generate electrode map."""

        rf = self.receptive_field_size
        if self.use_relative_receptive_field_size:
            rf *= self.phosphene_sizes

        self.electrode_map = torch.less(self.phosphene_map, rf).float()

    def observation(self, observation):
        """Create stimulation vector from sensor signal via electrode map."""

        sensor_signal = torch.tensor(observation, dtype=self.dtype,
                                     device=self.device)

        return torch.tensordot(self.electrode_map, sensor_signal).squeeze()


def get_eccentricity_scaling(r):
    """Spatial phosphene characteristics"""
    return 2 * r + 0.5  # TODO: PLAUSIBLE CORTICAL MAGNIFICATION
