from dataclasses import dataclass
from typing import Tuple

import gym
import torch
from torch import nn

from phossim.transforms.common import Transform, TransformConfig


@dataclass
class AutoencoderConfig(TransformConfig):
    shape: Tuple[int, int, int]
    encoder: nn.Module
    decoder: nn.Module
    dtype: torch.dtype = torch.float32
    device: torch.device = torch.device('cpu')


class AutoencoderFilter(Transform):
    def __init__(self, env: gym.Env, config: AutoencoderConfig):
        super().__init__(env, config)
        self._observation_space = gym.spaces.Box(
            low=0, high=1, shape=config.shape, dtype=float)
        self.dtype = config.dtype
        self.device = config.device
        self.encoder = config.encoder
        self.decoder = config.decoder
        self.encoded = None
        self.decoded = None
        self.loss_function = nn.MSELoss()
        self.reconstruction_loss = None

    def observation(self, observation):
        observation = torch.as_tensor(observation, dtype=self.dtype,
                                      device=self.device)
        observation = torch.movedim(observation, -1, 0)
        self.encoded = self.encoder.forward(observation)
        self.decoded = self.decoder.forward(self.encoded)
        self.reconstruction_loss = self.loss_function(self.decoded,
                                                      observation)

        return torch.movedim(self.encoded, 0, -1).detach().numpy()
