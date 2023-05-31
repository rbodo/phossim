from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from habitat.config.default_structured_configs import SimulatorSensorConfig

cs = ConfigStore.instance()

@dataclass
class HabitatBaselinesBaseConfig:
    pass

@dataclass
class ObsTransformConfig(HabitatBaselinesBaseConfig):
    type: str = MISSING

@dataclass
class GrayScaleTransform(ObsTransformConfig):
    type: str = "GrayScaleTransform"
    size: int = 256
    channels_last: bool = True
    trans_keys: Tuple[str, ...] = (
        "rgb",
        "depth",
        "semantic",
    )
    semantic_key: str = "semantic"


cs.store(
    package="habitat_baselines.rl.policy.obs_transforms.GrayScale",
    group="habitat_baselines/rl/policy/obs_transforms",
    name="GrayScale",
    node=GrayScaleTransform,
)