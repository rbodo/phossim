from dataclasses import dataclass

import gym
import habitat
from habitat.sims.habitat_simulator.actions import HabitatSimActions

FORWARD_KEY = 'w'
LEFT_KEY = 'a'
RIGHT_KEY = 'd'
FINISH = 'f'


@dataclass
class AihabitatConfig:
    observation_space: gym.Space
    path_config: str


class Aihabitat(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, config: AihabitatConfig):

        super().__init__()
        _config = habitat.get_config(config.path_config)
        self.env = habitat.Env(config=_config)
        self._num_channels = 16
        self._num_actions = 4
        self.action_space = gym.spaces.Discrete(self._num_actions)
        self.observation_space = config.observation_space
        self._state = None

    def step(self, action):
        observations = self.env.step(key2action(action))
        observation = observations['rgb']
        reward = get_reward(observations)
        done = self.env.episode_over or reward
        self._state = observation.astype('uint8')
        print_state(observations)
        return observation, reward, done, {}

    def reset(self, **kwargs):
        observations = self.env.reset()
        print_state(observations)
        return observations['rgb']

    def render(self, mode='human'):
        if mode == 'rgb_array':
            return self._state
        else:
            super().render(mode)


def key2action(key):
    if key == ord(FORWARD_KEY):
        action = HabitatSimActions.move_forward
        print('action: FORWARD')
    elif key == ord(LEFT_KEY):
        action = HabitatSimActions.turn_left
        print('action: LEFT')
    elif key == ord(RIGHT_KEY):
        action = HabitatSimActions.turn_right
        print('action: RIGHT')
    elif key == ord(FINISH):
        action = HabitatSimActions.stop
        print('action: FINISH')
    else:
        action = None
        print('INVALID KEY')
    return action


def get_reward(observations):
    return observations['pointgoal_with_gps_compass'][0] < 0.2


def print_state(observations):
    print("Destination distance: {:3f}, theta (radians): {:.2f}".format(
        *observations['pointgoal_with_gps_compass']))
