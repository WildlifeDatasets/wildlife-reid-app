import logging
import logging.config
import os

import yaml

_module_dir = os.path.abspath(os.path.dirname(__file__))
LOGGER_CONFIG = os.path.join(_module_dir, "../logging.yaml")


def setup_logging():
    """Setup logging configuration from a file.

    :param cfg_file: Logging configuration yaml file.
    """
    if not logging.getLogger("app").hasHandlers():
        with open(LOGGER_CONFIG, "r") as f:
            config = yaml.safe_load(f.read())
            logging.config.dictConfig(config)
