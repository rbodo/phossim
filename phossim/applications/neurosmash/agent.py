import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from phossim.interface.agent import AbstractAgent


class Agent(AbstractAgent):
    def __init__(self, model, **kwargs: dict):
        super().__init__(**kwargs)
        self._rng = np.random.default_rng()
        self.critic_value_history = []
        self.action_probs_history = []
        self.model = model
        self._num_actions = model.output_shape[0][1]

    def step(self, state, **kwargs):
        """Possible return values (actions to take):

        0: do nothing
        1: go left
        2: go right
        3: random motion
        # Todo: No straight motion?
        """

        shape = list(self.model.input_shape)
        shape[0] = 1  # Batch size
        state = np.reshape(state, shape)
        state = tf.convert_to_tensor(state)

        # Predict action probabilities and estimated future rewards
        # from environment state
        action_probs, critic_value = self.model(state)
        self.critic_value_history.append(critic_value[0, 0])

        # Sample action from action probability distribution
        self._action = self._rng.choice(self._num_actions,
                                        p=np.squeeze(action_probs))
        self.action_probs_history.append(tf.math.log(
            action_probs[0, self._action]))

        return self._action

    def reset_history(self):
        self.critic_value_history = []
        self.action_probs_history = []


def get_model(input_shape, num_actions):

    inputs = layers.Input(input_shape)
    common = layers.Conv2D(32, 3, strides=2, activation='relu')(inputs)
    common = layers.MaxPool2D()(common)
    common = layers.Conv2D(32, 3, strides=2, activation='relu')(common)
    common = layers.MaxPool2D()(common)
    common = layers.Flatten()(common)
    action = layers.Dense(num_actions, activation='softmax')(common)
    critic = layers.Dense(1)(common)

    model = keras.Model(inputs=inputs, outputs=[action, critic])
    model.summary()

    return model


def train(environment, model, num_episodes=10):
    gamma = 0.99  # Discount factor for past rewards

    # Smallest number such that 1.0 + eps != 1.0
    eps = np.finfo(np.float32).eps.item()

    optimizer = keras.optimizers.Adam(learning_rate=0.01)
    huber_loss = keras.losses.Huber()
    agent = Agent(model)

    for episode in range(num_episodes):
        state = environment.reset()
        agent.reset_history()

        with tf.GradientTape() as tape:
            while not environment.is_done:

                # Apply the sampled action in our environment
                action = agent.step(state)
                environment.step(action)

            # Calculate expected value from rewards. At each timestep, what was
            # the total reward received after that timestep? Rewards in the
            # past are discounted by multiplying them with gamma. These are
            # the labels for the critic.
            rewards_history = []
            discounted_sum = 0
            for r in reversed(environment.get_rewards_history()):
                discounted_sum = r + gamma * discounted_sum
                rewards_history.insert(0, discounted_sum)

            # Normalize.
            rewards_history = ((np.array(rewards_history)
                                - np.mean(rewards_history)) /
                               (np.std(rewards_history) + eps))

            # Calculate loss values to update the network.
            history = zip(agent.action_probs_history,
                          agent.critic_value_history,
                          rewards_history)
            total_loss = 0
            for log_prob, reward_expected, reward_received in history:
                actor_loss = -log_prob * (reward_received - reward_expected)
                critic_loss = huber_loss(tf.expand_dims(reward_expected, 0),
                                         tf.expand_dims(reward_received, 0))
                total_loss += actor_loss + critic_loss

            # Backpropagation.
            grads = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
