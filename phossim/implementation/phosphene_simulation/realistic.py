import gym
import numpy as np
import torch

from phossim.config import DTYPE, DEVICE
from phossim.interface import Transform, TransformConfig


def get_eccentricity_scaling(r):
    """Spatial phosphene characteristics"""
    return 2 * r + 0.5  # TODO: PLAUSIBLE CORTICAL MAGNIFICATION


def get_phosphene_map(num_phosphenes, resolution):
    # Cartesian coordinate system for the visual field
    x = torch.arange(resolution[0], dtype=DTYPE, device=DEVICE)
    y = torch.arange(resolution[1], dtype=DTYPE, device=DEVICE)
    grid = torch.meshgrid(x, y, indexing='ij')

    d_min = min(resolution)

    # Polar coordinates
    phi = 2 * torch.pi * torch.rand(num_phosphenes, dtype=DTYPE, device=DEVICE)
    r = d_min / 2 * torch.rand(num_phosphenes, dtype=DTYPE, device=DEVICE) ** 2

    # Convert to cartesian indices
    xmax = resolution[1] - 1
    ymax = resolution[0] - 1
    x_offset = torch.round(r * torch.cos(phi) + xmax / 2)
    y_offset = torch.round(r * torch.sin(phi) + ymax / 2)
    x_offset = torch.clip(x_offset, 0, xmax)
    y_offset = torch.clip(y_offset, 0, ymax)
    x_offset = torch.reshape(x_offset, (-1, 1, 1))
    y_offset = torch.reshape(y_offset, (-1, 1, 1))

    # Calculate distance map for every element wrt center of phosphene
    phosphene_map = torch.sqrt((torch.unsqueeze(grid[0], 0) - y_offset) ** 2 +
                               (torch.unsqueeze(grid[1], 0) - x_offset) ** 2)
    # Sigma at start of simulation
    phosphene_sizes = get_eccentricity_scaling(r / d_min)

    return phosphene_map, torch.reshape(phosphene_sizes, (-1, 1, 1))


class GaussianSimulator(Transform):
    def __init__(self, env: gym.Env, config: TransformConfig):
        """
        phosphene_map: ndarray
            A stack of phosphene mappings (i, j, k)
        phosphene_sizes: ndarray
            A vector of phosphene sizes at start of simulation (i,)
        intensity_decay: float

        Dimensions i = n_phosphenes, j = pixels_y , k = pixels_x
        """

        super().__init__(env, config)
        self.phosphene_map = config.phosphene_map
        self.phosphene_sizes = config.phosphene_sizes
        self.intensity_decay = config.PHOSPHENE_INTENSITY_DECAY
        self.neural_activation = torch.zeros(len(self.phosphene_map),
                                             dtype=DTYPE, device=DEVICE)
        self.gaussian_filters = None
        # Not needed any more; save space
        del config.phosphene_map
        del config.phosphene_sizes

    def get_gaussian_filters(self):
        """Generate gaussian activation maps, based on sigmas and phosphene
        mapping."""

        alpha = 1 / (self.phosphene_sizes * np.sqrt(np.pi))
        beta = 1 / (2 * self.phosphene_sizes ** 2)

        return torch.exp(-self.phosphene_map ** 2 * beta) * alpha

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

        return phosphenes.cpu().numpy().astype('uint8')
