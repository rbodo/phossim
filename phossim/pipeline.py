from itertools import count
from typing import Union

import gym
from stable_baselines3.common.base_class import BaseAlgorithm

from phossim.agent.human import HumanAgent
from phossim.rendering import DisplayList

QUIT_KEY = 'q'


class BasePipeline:
    environment: gym.Env = None
    agent: Union[BaseAlgorithm, HumanAgent] = None
    renderer: DisplayList = None

    def __init__(self, *args, **kwargs):
        self.max_num_episodes = kwargs.get('max_num_episodes', float('inf'))
        self._quit_key = kwargs.get('quit_key', QUIT_KEY)

    def run_episode(self):

        observation = self.environment.reset()

        while True:

            action, _ = self.agent.predict(observation)

            observation, reward, done, info = self.environment.step(action)

            key = self.renderer.render(info)

            if self._is_episode_done(key, done):
                if self._is_pipeline_done(key):
                    self.close()
                return key

    def run(self):

        for i_episode in count():

            key = self.run_episode()

            if self._is_run_done(key, i_episode):
                return key

    def _is_episode_done(self, key: int, done: bool) -> bool:
        return self._is_quit(key) or done

    def _is_run_done(self, key: int, i_episode: int) -> bool:
        return self._is_quit(key) or i_episode > self.max_num_episodes

    def _is_pipeline_done(self, key: int) -> bool:
        return self._is_quit(key)

    def _is_quit(self, key: int) -> bool:
        return key == ord(self._quit_key)

    def close(self):
        self.environment.close()
        self.renderer.stop()
