"""Configure file logging for MNIST training."""

import logging
from pathlib import Path


def configure_file_logging(logger: logging.Logger) -> Path:
    """Ensure training INFO messages are written to the repository log file."""
    log_directory: Path = Path(__file__).resolve().parents[2] / 'log'
    log_directory.mkdir(parents=True, exist_ok=True)
    log_path: Path = log_directory / 'mnist-training.log'
    resolved_log_path: str = str(log_path.resolve())

    if not any(
        isinstance(handler, logging.FileHandler)
        and handler.baseFilename == resolved_log_path
        for handler in logger.handlers
    ):
        file_handler: logging.FileHandler = logging.FileHandler(
            log_path, encoding='utf-8'
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter(
                '%(asctime)s %(levelname)s %(name)s: %(message)s'
            )
        )
        logger.addHandler(file_handler)

    logger.setLevel(logging.INFO)
    return log_path
