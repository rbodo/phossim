import numpy as np
# from stable_baselines.common.atari_wrappers import make_atari, wrap_deepmind
from phossim.utils.atari_wrappers import make_atari, wrap_deepmind

from phossim.interface.environment import AbstractEnvironment


class Environment(AbstractEnvironment):
    def __init__(self, **kwargs: dict):
        super().__init__(**kwargs)
        self._environment = None

    def setup(self, **kwargs):
        seed = 42
        # Use the Baseline Atari environment because of Deepmind helper
        # functions
        env = make_atari("BreakoutNoFrameskip-v4")
        # Warp the frames, grey scale, stake four frame and scale to smaller
        # ratio
        env = wrap_deepmind(env, frame_stack=True, scale=True)
        env.seed(seed)
        self._environment = env

    def reset_state(self):
        self._state = np.array(self._environment.reset())
        return self._state

    def step(self, action):
        state, reward, done, info = self._environment.step(action)
        self._is_done = done
        self._reward = reward
        self._state = np.array(state)
        return self._state

    def visualize(self):
        self._environment.render()
