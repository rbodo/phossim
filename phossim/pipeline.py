from __future__ import annotations

from itertools import count
from typing import Union, TYPE_CHECKING

import gym
from stable_baselines3.common.base_class import BaseAlgorithm

if TYPE_CHECKING:
    from phossim.agent.human import HumanAgent
    from phossim.rendering import ViewerList


class BasePipeline:
    environment: gym.Env = None
    agent: Union[BaseAlgorithm, HumanAgent] = None
    renderer: ViewerList = None

    def __init__(self, *args, **kwargs):
        self.max_num_episodes = kwargs.get('max_num_episodes', float('inf'))

    def run_episode(self):

        observation = self.environment.reset()

        done = False
        while not done:

            action, _ = self.agent.predict(observation)

            observation, reward, done, info = self.environment.step(action)

            self.renderer.render(info)

            if self._is_pipeline_done():
                self.close()
                return

    def run(self):

        for i_episode in count():

            self.run_episode()

            if self._is_run_done(i_episode):
                return

    def _is_run_done(self, i_episode: int) -> bool:
        return self._is_pipeline_done() or i_episode > self.max_num_episodes

    def _is_pipeline_done(self) -> bool:
        return self.renderer.has_exit

    def close(self):
        self.environment.close()
        self.renderer.stop()
