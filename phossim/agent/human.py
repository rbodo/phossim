from dataclasses import dataclass
from typing import Tuple

from phossim.rendering import ViewerList


@dataclass
class HumanAgentConfig:
    action_map: dict
    default_action: int = 0


class HumanAgent:
    def __init__(self, renderer: ViewerList, config: HumanAgentConfig):
        self.renderer = renderer
        self.action_map = config.action_map
        self.default_action = config.default_action

    def predict(self, *args, **kwargs) -> Tuple[int, None]:
        key = self.renderer.get_key(kind='action')
        action = self.action_map.get(key, self.default_action)
        return action, None
