import logging
import sys
from typing import Optional


def setup_logger(
    name: str = "BrainSpikeFormer", level: int = logging.INFO, log_file: Optional[str] = None, quiet: bool = False
) -> logging.Logger:
    """
    Setup logger with clean console output.

    Parameters
    ----------
    name : str
        Logger name
    level : int
        Logging level (INFO, WARNING, ERROR, DEBUG)
    log_file : str, optional
        Path to save log file
    quiet : bool
        Suppress warnings if True
    """
    if quiet:
        import warnings

        warnings.filterwarnings("ignore")
        logging.getLogger("matplotlib").setLevel(logging.WARNING)
        logging.getLogger("braindecode").setLevel(logging.WARNING)
        logging.getLogger("torch").setLevel(logging.WARNING)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)

    return logger


default_logger = setup_logger(quiet=True)
