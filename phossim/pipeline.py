from __future__ import annotations

from itertools import count
from typing import Union, TYPE_CHECKING

import gym
from stable_baselines3.common.base_class import BaseAlgorithm

if TYPE_CHECKING:
    from phossim.agent.human import HumanAgent
    from phossim.rendering import ViewerList, ViewerListBlocking


class BasePipeline:
    environment: gym.Env = None
    agent: Union[BaseAlgorithm, HumanAgent] = None
    renderer: Union[ViewerList, ViewerListBlocking] = None

    def __init__(self, *args, **kwargs):
        self.max_num_episodes = kwargs.get('max_num_episodes', float('inf'))

    def run(self):

        self.renderer.start()

        for i_episode in count():

            key = self.run_episode()

            if self._is_run_done(key, i_episode):
                break

        self.close()

    def run_episode(self):

        observation = self.environment.reset()

        while True:

            action, _ = self.agent.predict(observation)

            observation, reward, done, info = self.environment.step(action)

            key = self.renderer.render(info)

            if done or self._is_pipeline_done(key):
                return key

    def _is_run_done(self, key: str, i_episode: int) -> bool:
        return self.is_quit_key(key) or i_episode > self.max_num_episodes

    def _is_pipeline_done(self, key: str) -> bool:
        return self.is_quit_key(key)

    @staticmethod
    def is_quit_key(key: str) -> bool:
        return key == 'q'

    def close(self):
        self.environment.close()
        self.renderer.stop()


class InteractivePipeline(BasePipeline):
    def run_episode(self):

        self.environment.reset()
        observation = None
        while True:

            action, _ = self.agent.predict(key=observation)

            _, reward, done, info = self.environment.step(action)

            observation = self.renderer.render(info)

            if done or self._is_pipeline_done(observation):
                if reward:
                    print("You successfully navigated to destination point.")
                else:
                    print("Your navigation was unsuccessful.")
                return observation
