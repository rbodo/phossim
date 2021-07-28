import abc


class AbstractEnvironment(abc.ABC):
    def __init__(self, **kwargs):
        self._state = None
        self._reward = None
        self._is_done = False

    def setup(self, **kwargs):
        pass

    def reset(self):
        self._state = self.reset_state()
        self.reset_history()
        return self._state

    def reset_state(self):
        pass

    def reset_history(self):
        self._reward = None
        self._is_done = False

    @abc.abstractmethod
    def step(self, action):
        pass

    def visualize(self):
        pass

    @property
    def is_done(self):
        return self._is_done

    def get_current_state(self):
        return self._state

    def get_current_reward(self):
        return self._reward
