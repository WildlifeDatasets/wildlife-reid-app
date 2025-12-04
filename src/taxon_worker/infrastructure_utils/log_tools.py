import logging


class TempLogContext:
    def __init__(self, logger_names: list, levels: list):

        self.logger_names = logger_names
        self.levels = levels
        self.loggers = [logging.getLogger(logger_name) for logger_name in logger_names]

    def __enter__(self):
        self.old_levels = [logger.level for logger in self.loggers]
        for logger, level in zip(self.loggers, self.levels):
            logger.setLevel(level)

    def __exit__(self, exc_type, exc_value, traceback):
        for logger, level in zip(self.loggers, self.old_levels):
            logger.setLevel(level)
