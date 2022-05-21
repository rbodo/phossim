from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict

import cv2
import gym
import numpy as np

from phossim.config import Config, QUIT_KEY, AbstractConfig

CAMERA_KEY = 'c'
FILTER_KEY = 'f'
STIMULUS_KEY = 's'
PHOSPHENE_KEY = 'p'
RECORD_KEY = 'r'

LABEL_MAP = {'Sensor signal': CAMERA_KEY,
             'Filtered signal': FILTER_KEY,
             'Stimulus': STIMULUS_KEY,
             'Phosphenes': PHOSPHENE_KEY}


class CameraModes(Enum):
    """Possible camera input choices.

    Toggle with key 'c'.
    """

    laptop = 0
    webcam = 1
    dvs_aps = 2
    dvs_aer = 3


class FilterModes(Enum):
    """Image processing modes.

    Toggle with key 'f'.
    """

    identity = 0
    edge = 1


class StimulusModes(Enum):
    """Possible stimulus generation modes.

    Toggle with key 's'.
    """

    identity = 0
    realistic = 1


class PhospheneModes(Enum):
    """Possible phosphene simulation methods.

    Toggle with key 'p'.
    """

    identity = 0
    basic = 1
    realistic = 2


class ShortcutModes(Enum):
    """Keyboard shortcuts to toggle between common pipelines quickly."""

    aps = {
        CAMERA_KEY: CameraModes.dvs_aps,
        FILTER_KEY: FilterModes.identity,
        STIMULUS_KEY: StimulusModes.identity,
        PHOSPHENE_KEY: PhospheneModes.identity}

    aps_edge = {
        CAMERA_KEY: CameraModes.dvs_aps,
        FILTER_KEY: FilterModes.edge,
        STIMULUS_KEY: StimulusModes.identity,
        PHOSPHENE_KEY: PhospheneModes.identity}

    aps_edge_phos = {
        CAMERA_KEY: CameraModes.dvs_aps,
        FILTER_KEY: FilterModes.edge,
        STIMULUS_KEY: StimulusModes.identity,
        PHOSPHENE_KEY: PhospheneModes.basic}

    dvs = {
        CAMERA_KEY: CameraModes.dvs_aer,
        FILTER_KEY: FilterModes.identity,
        STIMULUS_KEY: StimulusModes.identity,
        PHOSPHENE_KEY: PhospheneModes.identity}

    dvs_phos = {
        CAMERA_KEY: CameraModes.dvs_aer,
        FILTER_KEY: FilterModes.identity,
        STIMULUS_KEY: StimulusModes.identity,
        PHOSPHENE_KEY: PhospheneModes.basic}


keyboard_shortcuts = OrderedDict([(str(i + 1), v.value)
                                  for i, v in enumerate(ShortcutModes)])


KNOWN_KEYS = [ord(QUIT_KEY),
              ord(RECORD_KEY),
              ord(CAMERA_KEY),
              ord(FILTER_KEY),
              # ord(STIMULUS_KEY),  Switching modes not safely implemented yet.
              ord(PHOSPHENE_KEY)] + \
             [ord(k) for k in keyboard_shortcuts.keys()]


@dataclass
class HumanAgentConfig(Config):
    modes: Dict[str, Enum] = field(default_factory=dict)

    def pipeline_string(self) -> str:
        return "".join([str(m) + ' > ' for m in self.modes.values()])[:-3]

    def apply_key(self, key: int):
        if key is None:
            return

        key = chr(key)
        if key in keyboard_shortcuts.keys():
            self.modes = keyboard_shortcuts[key].copy()
        elif key in LABEL_MAP.values():
            mode = self.modes[key]
            cls = mode.__class__
            new_mode = cls((mode.value + 1) % len(cls))
            self.modes[key] = new_mode
            self.validate_config()

        print(self.pipeline_string())

    def validate_config(self):

        # Check that stimulus generator is compatible with phosphene simulator.
        phosphene_mode = self.modes[PHOSPHENE_KEY]
        if phosphene_mode in {PhospheneModes.identity, PhospheneModes.basic}:
            self.modes[STIMULUS_KEY] = StimulusModes.identity
        elif phosphene_mode == PhospheneModes.realistic:
            self.modes[STIMULUS_KEY] = StimulusModes.realistic

        # Check that phosphene simulator is compatible with stimulus generator.
        stimulus_mode = self.modes[STIMULUS_KEY]
        if stimulus_mode == StimulusModes.identity:
            assert phosphene_mode != PhospheneModes.realistic
        elif stimulus_mode == StimulusModes.realistic:
            self.modes[PHOSPHENE_KEY] = PhospheneModes.realistic


@dataclass
class HumanAgentConfig(AbstractConfig):
    action_map: dict
    default_action: int = 0


class HumanAgent:
    def __init__(self, environment: gym.Env, config: HumanAgentConfig):
        self.environment = environment
        self.action_map = config.action_map
        self.default_action = config.default_action
        assert len(self.action_map) <= self.environment.action_space.n

    # noinspection PyUnusedLocal
    def predict(self, observation: np.ndarray) -> int:
        key = cv2.waitKey(0)
        if key != -1:
            return self.action_map.get(chr(key), self.default_action)
        return self.default_action
