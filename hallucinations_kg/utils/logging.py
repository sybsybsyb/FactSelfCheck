import sys

from loguru import logger


def setup_logger() -> None:
    logger.level("INFO")
    logger.add(
        sys.stdout,
        colorize=True,
        format="<green>{time}</green> <level>{message}</level>",
        level="INFO",
    )


def set_error_level() -> None:
    logger.remove()
    logger.add(sys.stderr, level="ERROR")
