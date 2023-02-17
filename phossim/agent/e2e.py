from dataclasses import dataclass, asdict
from typing import Optional

import gym
import numpy as np
import torch
import torch as th
from torch import nn
from torch.nn.functional import mse_loss
from stable_baselines3 import PPO
from stable_baselines3.common.utils import explained_variance


@dataclass
class TrainingConfig:
    total_timesteps: int
    log_interval: int = 1
    tb_log_name: str = 'run'
    eval_env: Optional[gym.Env] = None
    eval_freq: int = -1
    n_eval_episodes: int = 5
    eval_log_path: Optional[str] = None
    reset_num_timesteps: bool = True

    def asdict(self):
        return asdict(self)


@dataclass
class AgentConfig:
    encoder: nn.Module
    decoder: nn.Module
    phosphene_simulator: nn.Module
    device: str


class Encoder(nn.Module):
    def __init__(self, num_electrodes):
        super().__init__()
        self.num_electrodes = num_electrodes
        nonlinearity = nn.ReLU()
        pool = nn.AvgPool2d(2)
        conv_kwargs = dict(kernel_size=3, stride=1, padding=1)
        self.layers = nn.ModuleList([
            nn.Conv2d(1, 16, **conv_kwargs),
            nonlinearity,
            pool,
            nn.Conv2d(16, 32, **conv_kwargs),
            nonlinearity,
            pool,
            nn.Conv2d(32, 16, **conv_kwargs),
            nonlinearity,
            nn.Conv2d(16, 1, **conv_kwargs),
            nn.Sigmoid(),
            nn.Flatten()
        ])

    def forward(self, x):
        for module in self.layers:
            x = module(x)
        return x[:, :self.num_electrodes]


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        nonlinearity = nn.ReLU()
        pool = nn.AvgPool2d(2)
        conv_kwargs = dict(kernel_size=3, stride=1, padding=1)
        self.layers = nn.ModuleList([
            nn.Conv2d(1, 16, **conv_kwargs),
            nonlinearity,
            pool,
            nn.Conv2d(16, 32, **conv_kwargs),
            nonlinearity,
            nn.Conv2d(32, 16, **conv_kwargs),
            nonlinearity,
            nn.Conv2d(16, 1, **conv_kwargs),
            nn.Sigmoid(),
        ])

    def forward(self, x):
        for module in self.layers:
            x = module(x)
        return x


class EncoderDecoder(nn.Module):
    def __init__(self, observation_space: gym.spaces.Space,
                 config: AgentConfig):
        super().__init__()
        self.observation_space = observation_space
        self.encoder = config.encoder
        self.decoder = config.decoder
        self.device = config.device
        self.loss_function = nn.MSELoss()
        self.reconstruction_loss = None
        self.flatten = nn.Flatten()
        self.features_dim = self.get_feature_dim()

    def get_feature_dim(self):
        # Get dummy input.
        batch_size = self.phosphene_simulator.sim.batch_size
        x = torch.zeros((batch_size,) + self.observation_space.shape,
                        dtype=torch.float32, device=self.device)
        # Compute model output.
        with torch.no_grad():
            y = self.encoder(x)
        # Return number of neurons in output layer.
        return np.prod(y.shape[1:])

    def forward(self, x):
        stimulus = self.encoder.forward(x)
        decoded = self.decoder.forward(stimulus)
        self.reconstruction_loss = self.loss_function(decoded, x)

        return self.flatten(stimulus)


class PhospheneEncoderDecoder(nn.Module):
    def __init__(self, observation_space: gym.spaces.Space,
                 config: AgentConfig):
        super().__init__()
        self.phosphene_simulator = config.phosphene_simulator
        self.observation_space = observation_space
        self.encoder = config.encoder
        self.decoder = config.decoder
        self.device = config.device
        self.loss_function = nn.MSELoss()
        self.reconstruction_loss = None
        self.flatten = nn.Flatten()
        self.features_dim = self.get_feature_dim()

    def get_feature_dim(self):
        # Get dummy input.
        batch_size = self.phosphene_simulator.sim.params['run']['batch_size']
        x = torch.zeros((batch_size,) + self.observation_space.shape,
                        dtype=torch.float32, device=self.device)
        # Compute model output.
        with torch.no_grad():
            y = self.phosphene_simulator(self.encoder(x))
        # Return number of neurons in output layer.
        return np.prod(y.shape[1:])

    def forward(self, x):
        stimulus = self.encoder.forward(x)
        phosphenes = self.phosphene_simulator(stimulus)
        decoded = self.decoder.forward(torch.unsqueeze(phosphenes, 1))
        self.reconstruction_loss = self.loss_function(decoded, x)

        return self.flatten(phosphenes)


class E2ePPO(PPO):
    def train(self) -> None:
        """ Update policy using the currently gathered rollout buffer."""

        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = \
                self.clip_range_vf(self._current_progress_remaining)
        else:
            clip_range_vf = None

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        approx_kl_divs = None
        loss = None

        continue_training = True

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, gym.spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                advantages = (advantages - advantages.mean()) / \
                             (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the
                # first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range,
                                                      1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) >
                                         clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the different between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf,
                        clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = (policy_loss + self.ent_coef * entropy_loss +
                        self.vf_coef * value_loss)

                loss = (loss +
                        self.policy.features_extractor.reconstruction_loss)
                # If we had implemented the e2e encoder as an observation
                # wrapper, would use this line:
                # loss = loss + self.env.venv.envs[0].reconstruction_loss

                # Calculate approximate form of reverse KL Divergence for
                # early stopping see issue #417:
                # https://github.com/DLR-RM/stable-baselines3/issues/417 and
                # discussion in PR #419:
                # https://github.com/DLR-RM/stable-baselines3/pull/419 and
                # Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1)
                                            - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None \
                        and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching"
                              f" max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(),
                                            self.max_grad_norm)
                self.policy.optimizer.step()

            if not continue_training:
                break

        self._n_updates += self.n_epochs
        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(),
            self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std",
                               th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates,
                           exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
