from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Callable

import gym


@dataclass
class RecordingConfig:
    video_folder: Path
    episode_trigger: Optional[Callable[[int], bool]] = None
    step_trigger: Optional[Callable[[int], bool]] = None
    video_length: int = 0
    name_prefix: str = None


class RecordingTransform(gym.wrappers.RecordVideo):
    def __init__(self, env: gym.Env, config: RecordingConfig):
        if 'render_modes' in env.metadata:  # Fixing gym bug
            env.metadata['render.modes'] = env.metadata['render_modes']
        super().__init__(env, **asdict(config))
