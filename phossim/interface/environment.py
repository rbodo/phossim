from gym.utils.env_checker import check_env

from phossim.config import Config


def get_environment(config: Config):
    environment = config.environment_getter(config.environment)
    check_env(environment, skip_render_check=True)
    return environment
