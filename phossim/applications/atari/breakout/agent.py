import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from phossim.interface.agent import AbstractAgent


class Agent(AbstractAgent):
    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self._epsilon = 1.0  # Epsilon greedy parameter
        self._epsilon_min = 0.1  # Minimum epsilon greedy parameter
        self._epsilon_max = 1.0  # Maximum epsilon greedy parameter
        # Rate at which to reduce chance of random action being taken
        self._epsilon_interval = (self._epsilon_max - self._epsilon_min)
        # Number of frames to take random action and observe output
        self._epsilon_random_frames = 50000
        # Number of frames for exploration
        self._epsilon_greedy_frames = 1000000.0
        self.num_actions = model.output_shape[1]
        self._rng = np.random.default_rng()

    def step(self, state, **kwargs):
        # Use epsilon-greedy for exploration
        if 'frame_count' in kwargs and (
                kwargs['frame_count'] < self._epsilon_random_frames or
                self._epsilon > self._rng.random()):
            # Take random action
            self._action = self._rng.choice(self.num_actions)
            # Decay probability of taking random action
            self._epsilon -= \
                self._epsilon_interval / self._epsilon_greedy_frames
            self._epsilon = max(self._epsilon, self._epsilon_min)
        else:
            # Predict action Q-values
            # From environment state
            state_tensor = tf.convert_to_tensor(state)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_probs = self.model(state_tensor, training=False)
            # Take best action
            self._action = tf.argmax(action_probs[0]).numpy()

        return self._action


def get_model(input_shape, num_actions):
    # Network defined by the Deepmind paper
    inputs = layers.Input(input_shape)

    # Convolutions on the frames on the screen
    layer1 = layers.Conv2D(32, 8, strides=4, activation='relu')(inputs)
    layer2 = layers.Conv2D(64, 4, strides=2, activation='relu')(layer1)
    layer3 = layers.Conv2D(64, 3, strides=1, activation='relu')(layer2)

    layer4 = layers.Flatten()(layer3)

    layer5 = layers.Dense(512, activation='relu')(layer4)
    action = layers.Dense(num_actions, activation='linear')(layer5)

    return keras.Model(inputs=inputs, outputs=action)


def train(environment, model_q, model_target, num_episodes=10000):
    visualize = False
    gamma = 0.99  # Discount factor for past rewards
    batch_size = 32  # Size of batch taken from replay buffer
    max_steps_per_episode = 10000

    # In the Deepmind paper they use RMSProp however then Adam optimizer
    # improves training time
    optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)

    # Experience replay buffers
    action_history = []
    state_history = []
    rewards_history = []
    done_history = []
    episode_reward_history = []
    running_reward = 0
    frame_count = 0

    # Maximum replay length
    # Note: The Deepmind paper suggests 1000000 however this causes memory
    # issues
    max_memory_length = 100000
    # Train the model after 4 actions
    update_after_actions = 4
    # How often to update the target network
    update_target_network = 10000
    # Using huber loss for stability
    loss_function = keras.losses.Huber()

    rng = np.random.default_rng()

    agent = Agent(model_q)

    for episode in range(num_episodes):
        state = environment.reset()
        episode_reward = 0

        for timestep in range(1, max_steps_per_episode):
            if len(state_history) == 0:
                state_history.append(state)

            if visualize:
                environment.visualize()

            frame_count += 1

            action = agent.step(state, frame_count=frame_count)

            # Apply the sampled action in our environment.
            state = environment.step(action)

            reward = environment.get_current_reward()
            episode_reward += reward

            # Save actions and states in replay buffer.
            action_history.append(action)
            state_history.append(state)
            done_history.append(environment.is_done)
            rewards_history.append(reward)
            len_history = len(done_history)

            # Update every fourth frame and once we are past the first batch.
            if frame_count % update_after_actions == 0 \
                    and len_history > batch_size:
                # Get indices of samples for replay buffers
                indices = rng.choice(len_history, size=batch_size)

                # Sample from replay buffer
                state_shape = (len_history,) + model_q.input_shape[1:]
                state_sample = np.zeros(state_shape)
                state_next_sample = np.zeros(state_shape)
                rewards_sample = np.zeros(len_history)
                action_sample = np.zeros(len_history)
                done_sample = np.zeros(len_history)
                for i, j in enumerate(indices):
                    state_sample[i] = state_history[j]
                    state_next_sample[i] = state_history[j + 1]
                    rewards_sample[i] = rewards_history[j]
                    action_sample[i] = action_history[j]
                    done_sample[i] = done_history[j]
                # done_sample = tf.convert_to_tensor(done_sample)

                # Build the updated Q-values for the sampled future states
                # Use the target model for stability
                future_rewards = model_target.predict(state_next_sample)
                # Q value = reward + discount factor * expected future reward
                updated_q_values = rewards_sample + gamma * tf.reduce_max(
                    future_rewards, axis=1)

                # If final frame set the last value to -1
                updated_q_values = \
                    updated_q_values * (1 - done_sample) - done_sample

                # Create a mask so we only calculate loss on the updated
                # Q-values
                masks = tf.one_hot(action_sample, agent.num_actions)

                with tf.GradientTape() as tape:
                    # Train the model on the states and updated Q-values
                    q_values = model_q(state_sample)

                    # Apply the masks to the Q-values to get the Q-value for
                    # action taken
                    q_action = tf.reduce_sum(tf.multiply(q_values, masks), 1)
                    # Calculate loss between new Q-value and old Q-value
                    loss = loss_function(updated_q_values, q_action)

                # Backpropagation
                grads = tape.gradient(loss, model_q.trainable_variables)
                optimizer.apply_gradients(zip(grads,
                                              model_q.trainable_variables))

            if frame_count % update_target_network == 0:
                # update the the target network with new weights
                model_target.set_weights(model_q.get_weights())
                # Log details
                template = "running reward: {:.2f} at episode {}, frame " \
                           "count {}"
                print(template.format(running_reward, episode, frame_count))

            # Limit the state and reward history
            if len(rewards_history) > max_memory_length:
                rewards_history.pop(0)
                state_history.pop(0)
                action_history.pop(0)
                done_history.pop(0)

            if environment.is_done:
                break

        # Update running reward to check condition for solving
        episode_reward_history.append(episode_reward)
        if len(episode_reward_history) > 100:
            episode_reward_history.pop(0)
        running_reward = np.mean(episode_reward_history)

        if running_reward > 40:  # Condition to consider the task solved
            print("Solved at episode {}!".format(episode))
            break
