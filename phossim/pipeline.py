from itertools import count

from phossim.config import QUIT_KEY, Config
from phossim.interface import get_agent, wrap_transforms, get_environment
from phossim.utils import wrap_common
from phossim.rendering import get_renderer


class Pipeline:
    def __init__(self, config: Config):
        self.config = config
        self.is_alive = False
        self.environment = None
        self.filter = None
        self.stimulus_generator = None
        self.phosphene_simulator = None
        self.agent = None
        self.renderer = None
        self.recorder = None

    def setup(self):
        self.renderer = get_renderer(self.config)
        self.environment = get_environment(self.config)
        self.environment = wrap_transforms(self.environment, self.config)
        self.environment = wrap_common(self.environment, self.config)
        self.agent = get_agent(self.environment, self.config)
        self.is_alive = True

    def update(self, key):
        self.config.apply_key(key)
        self.setup()

    def run(self):
        assert self.is_alive, "Call pipeline.setup() before running."

        # self.agent.learn(int(1e8))

        key = None
        for i_episode in count():

            observation = self.environment.reset()

            while True:

                action, state = self.agent.predict(observation)

                observation, reward, done, info = self.environment.step(action)

                key = self.renderer(info)

                if self._is_episode_done(key, done):
                    break

            if self._is_run_done(key, i_episode):
                break

        return key

    @staticmethod
    def _is_episode_done(key, done):
        return key or done

    def _is_run_done(self, key, i_episode):
        return key or i_episode > self.config.max_num_episodes

    @staticmethod
    def _is_pipeline_done(key):
        return key == ord(QUIT_KEY)

    def stop(self, key):
        self.environment.close()
        self.recorder.close()
        if self._is_pipeline_done(key):
            self.renderer.stop()
            self.is_alive = False


def main(config: Config):

    pipeline = Pipeline(config)

    pipeline.setup()

    while pipeline.is_alive:

        # Run full pipeline until user initiates a change in config.
        key = pipeline.run()

        # Close pipeline before applying any changes.
        pipeline.stop(key)

        # Apply modification to pipeline configuration.
        pipeline.update(key)
