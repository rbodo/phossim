import sys
from typing import Dict, List, Tuple, Type, Union

import gym
from torch import nn, Tensor

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import MlpExtractor


class CnnFeatureExtractor(nn.Module):
    # This class would contain the encoder, decoder, and phosphene model.
    # But since the phosphene model is implemented as an ObservationWrapper,
    # we don't go with this approach to avoid code duplication.
    def __init__(self, observation_space: gym.spaces.Space):
        super().__init__()
        self.observation_space = observation_space
        self.conv = nn.Conv2d(3, 2, 1)
        self.features_dim = (observation_space.shape[1] *
                             observation_space.shape[2] * 2)

    def forward(self, x):
        return nn.ReLU()(self.conv(x))


class FlattenMlpExtractor(MlpExtractor):
    # This class simply inserts a flatten layer because SB3 assumes flattened
    # input to the policy.
    def __init__(self, feature_dim: int,
                 net_arch: List[Union[int, Dict[str, List[int]]]],
                 activation_fn: Type[nn.Module]):
        super().__init__(feature_dim, net_arch, activation_fn)
        self.flatten = nn.Flatten()

    def forward(self, features: Tensor) -> Tuple[Tensor, Tensor]:
        return super().forward(self.flatten(features))

    def forward_actor(self, features: Tensor) -> Tensor:
        return super().forward_actor(self.flatten(features))

    def forward_critic(self, features: Tensor) -> Tensor:
        return super().forward_critic(self.flatten(features))


class CustomActorCriticPolicy(ActorCriticPolicy):
    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = FlattenMlpExtractor(self.features_dim, [10],
                                                 nn.Tanh)


model = PPO(CustomActorCriticPolicy, "Breakout-v4", verbose=1,
            policy_kwargs={'features_extractor_class': CnnFeatureExtractor,
                           'features_extractor_kwargs': {}})
model.learn(3)

sys.exit()
