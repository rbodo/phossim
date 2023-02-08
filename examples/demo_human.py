import os
import sys
from dataclasses import dataclass
from itertools import count
from pathlib import Path
from typing import List, Tuple, Type, Optional, Union, Dict

import gym

from phossim.pipeline import BasePipeline
from phossim.environment.camera import CameraConfig, CameraStream
from phossim.transforms.common import (
    Transform, TransformConfig, wrap_transforms, ResizeConfig, RecordingConfig,
    RecordingTransform, GrayscaleTransform, GrayscaleConfig, ResizeTransform,
    VrDisplayConfig, VrDisplayTransform)
from phossim.transforms.edge import CannyFilter, CannyConfig
from phossim.agent.human import HumanAgentConfig, HumanAgent
from phossim.rendering import Viewer, ViewerConfig, ViewerList
from phossim.transforms import phosphenes


@dataclass
class SwappableTransformConfig(TransformConfig):
    key: str


class SwappableTransform(Transform):
    def __init__(self, env, config: SwappableTransformConfig):
        super().__init__(env, config)
        self.key = config.key


@dataclass
class Config:
    environments: Dict[str, List[gym.Env]]
    transforms: List[Union[
        Tuple[Type[Transform], TransformConfig],
        Dict[str, List[Tuple[Type[Transform], TransformConfig]]]]]
    agent_config: HumanAgentConfig
    viewers: List[Viewer]
    device: Optional[str] = 'cpu'


class Pipeline(BasePipeline):
    def __init__(self, config: Config):
        super().__init__()
        self.renderer = ViewerList(config.viewers)
        self.agent = HumanAgent(self.renderer, config.agent_config)
        self._environment_selector = EnvironmentSelector(config.environments)
        self._transform_selector = TransformSelector(config.transforms)
        self.apply_config()

    def apply_config(self, key: Optional[str] = None):
        self.environment = self._environment_selector.toggle_environment(key)
        transforms = self._transform_selector.toggle_transforms(key)
        self.environment = wrap_transforms(self.environment, transforms)

    def run(self):

        self.renderer.start()

        for i_episode in count():

            key = self.run_episode()

            if self._is_run_done(key, i_episode):
                break

            self.apply_config(key)

        self.renderer.stop()

    @property
    def _setup_keys(self):
        transform_keys = set(self._transform_selector.counters.keys())
        return transform_keys.union({self._environment_selector.key})

    def _is_pipeline_done(self, key: str) -> bool:
        return self.is_quit_key(key) or key in self._setup_keys


class EnvironmentSelector:
    def __init__(self, environments: dict):
        self.key, self.environments = environments.popitem()
        self.counter = None
        self.limit = None
        self.reset()

    def reset(self):
        self.counter = 0
        self.limit = len(self.environments)

    def _advance_counter(self) -> int:
        self.counter = (self.counter + 1) % self.limit
        return self.counter

    def toggle_environment(self, key: Optional[str] = None) -> gym.Env:
        i = self._advance_counter() if key == self.key else self.counter
        return self.environments[i]


class TransformSelector:
    def __init__(self, transforms: list):
        self.transforms_all = transforms
        self.counters = {}
        self.limits = {}
        self.reset()

    def reset(self):
        for transform in self.transforms_all:
            if isinstance(transform, dict):
                key, values = self._get_item(transform)
                self.counters[key] = 0
                self.limits[key] = len(values)

    @staticmethod
    def _get_item(singleton_dict: dict) -> Tuple[str, List]:
        return list(singleton_dict.items())[0]

    def _advance_counter(self, key: str) -> int:
        self.counters[key] = (self.counters[key] + 1) % self.limits[key]
        return self.counters[key]

    def toggle_transforms(self, key: Optional[str] = None) -> list:
        transforms = []
        for transform in self.transforms_all:
            if isinstance(transform, dict):
                key_, values = self._get_item(transform)
                i = (self._advance_counter(key) if key == key_ else
                     self.counters[key_])
                transform = values[i]
            transforms.append(transform)
        return transforms


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '4'
    device = 'cuda:0'
    input_key = 'input'
    filter_key = 'filtered_observation'
    phosphene_key = 'phosphenes'
    vr_key = 'vr_display'
    path_base = Path('~/Data/phosphenes/dvs_human').expanduser()
    path_recording = path_base.joinpath('recording')
    video_length = 300
    def recording_trigger(episode): return episode % 10000 == 0

    shape_in = (128, 128, 3)
    shape_resized = (512, 512, 3)
    shape_gray = (512, 512, 1)
    shape_vr = (1600, 2880, 1)
    num_phosphenes = 256

    environments = {'c': [
        CameraStream(CameraConfig(shape_in, 0)),
        CameraStream(CameraConfig(shape_in, 0))  # could be DVS or VR camera.
    ]}

    transforms = [
        (Transform, TransformConfig(input_key)),
        (ResizeTransform, ResizeConfig(None, shape_resized)),
        (RecordingTransform, RecordingConfig(path_recording,
                                             episode_trigger=recording_trigger,
                                             video_length=video_length,
                                             name_prefix='input')),
        (GrayscaleTransform, GrayscaleConfig(None, shape_gray)),
        {'f': [
            (Transform, TransformConfig(filter_key)),  # Identity (no filter).
            (CannyFilter, CannyConfig(filter_key, sigma=1))  # Edge filter.
        ]},
        (RecordingTransform, RecordingConfig(path_recording,
                                             episode_trigger=recording_trigger,
                                             video_length=video_length,
                                             name_prefix='filtered')),
        {'p': [
            (phosphenes.basic.PhospheneSimulation,
             phosphenes.basic.PhospheneSimulationConfig(phosphene_key)),
            (phosphenes.realistic.PhospheneSimulation,
             phosphenes.realistic.PhospheneSimulationConfig(
                 phosphene_key, shape_gray, num_phosphenes))
        ]},
        (RecordingTransform, RecordingConfig(path_recording,
                                             episode_trigger=recording_trigger,
                                             video_length=video_length,
                                             name_prefix='phosphenes')),
        (VrDisplayTransform, VrDisplayConfig(vr_key, shape_vr))
    ]

    agent_config = HumanAgentConfig({})

    displays = [
        Viewer(ViewerConfig(input_key, 'webcam')),
        Viewer(ViewerConfig(filter_key, 'filter')),
        Viewer(ViewerConfig(phosphene_key, 'phosphenes')),
        Viewer(ViewerConfig(vr_key, 'vr_display'))
    ]

    config = Config(environments,
                    transforms,
                    agent_config,
                    displays,
                    device)

    pipeline = Pipeline(config)

    pipeline.run()


if __name__ == '__main__':
    main()
    sys.exit()
