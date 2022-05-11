from itertools import count

from phossim.config import QUIT_KEY, Config
from phossim.interface.agent import get_agent
from phossim.interface.environment import get_environment
from phossim.interface.filtering import wrap_filtering
from phossim.interface.phosphene_simulation import wrap_phosphene_simulation
from phossim.interface.stimulus_generation import wrap_stimulus_generation
from phossim.utils import wrap_common


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
        environment = get_environment(self.config)
        environment = wrap_filtering(environment, self.config)
        environment = wrap_stimulus_generation(environment, self.config)
        environment = wrap_phosphene_simulation(environment, self.config)
        environment = wrap_common(environment, self.config)
        self.environment = environment
        self.agent = get_agent(environment, self.config)
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

                # self.environment.render()

                action, state = self.agent.predict(observation)

                observation, reward, done, info = self.environment.step(action)

                # key = info['key']
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
        self.environment.stop()
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
