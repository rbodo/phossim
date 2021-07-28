import abc

import numpy as np


class AbstractAgent(abc.ABC):
    def __init__(self, **kwargs):
        self._action = None

    @abc.abstractmethod
    def step(self, state: np.ndarray, **kwargs):
        pass

    def get_current_action(self):
        return self._action

    def get_model(self):
        pass

    def train(self):
        pass
