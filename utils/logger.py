import logging
from logging import Logger


class LyLMLogger(Logger):
    def __init__(
            self,
            name=__name__,
            level=logging.DEBUG,
            fmt="%(levelname)s %(asctime)s [%(filename)s:%(lineno)d] %(message)s",
            datefmt="%m-%d %H:%M:%S"
    ):
        super().__init__(name, level)

        if not self.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)
            console_handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
            self.addHandler(console_handler)


def get_logger(
        name=__name__,
        level=logging.DEBUG,
        fmt="%(levelname)s %(asctime)s [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%m-%d %H:%M:%S"
):
    return LyLMLogger(name, level, fmt, datefmt)
