import abc

import numpy as np


class AbstractReconstructionDecoder(abc.ABC):
    def __init__(self, **kwargs):
        self._model = None
        self._loss = None

    @abc.abstractmethod
    def step(self, state: np.ndarray, **kwargs):
        pass

    @abc.abstractmethod
    def get_model(self):
        pass

    @abc.abstractmethod
    def train(self):
        pass
