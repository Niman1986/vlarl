# Copied and Adapted from https://github.com/volcengine/verl
# ---------------------------------------------------------------------
# Adapted from
# https://github.com/skypilot-org/skypilot/blob/86dc0f6283a335e4aa37b3c10716f90999f48ab6/sky/sky_logging.py
"""Logging configuration for vLLM."""
import logging
import sys
from typing import Dict, Optional

_FORMAT = "%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s"
_DATE_FORMAT = "%m-%d %H:%M:%S"


class NewLineFormatter(logging.Formatter):
    """Adds logging prefix to newlines to align multi-line messages."""

    def __init__(self, fmt, datefmt=None):
        logging.Formatter.__init__(self, fmt, datefmt)

    def format(self, record):
        msg = logging.Formatter.format(self, record)
        if record.message != "":
            parts = msg.split(record.message)
            msg = msg.replace("\n", "\r\n" + parts[0])
        return msg


class ColorFormatter(NewLineFormatter):
    """Adds colors to logging messages while maintaining newline alignment."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',     # cyan
        'INFO': '\033[32m',      # green
        'WARNING': '\033[33m',   # yellow
        'ERROR': '\033[31m',     # red
        'CRITICAL': '\033[41m',  # red background
        'RESET': '\033[0m'       # reset
    }

    def format(self, record):
        # Add colors to levelname
        if hasattr(record, 'levelname'):
            color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
            record.levelname = f"{color}{record.levelname}{self.COLORS['RESET']}"
        
        return super().format(record)


class RankFormatter(ColorFormatter):
    """Adds rank prefix and colors to logging messages while maintaining newline alignment."""
    
    def format(self, record):
        # Add rank prefix if rank attribute exists
        if hasattr(record, 'rank'):
            record.msg = f"[Rank {record.rank}] {record.msg}"
        
        return super().format(record)


_root_logger = logging.getLogger("openrlhf")
_default_handler = None


def _setup_logger():
    _root_logger.setLevel(logging.DEBUG)
    global _default_handler
    if _default_handler is None:
        _default_handler = logging.StreamHandler(sys.stdout)
        _default_handler.flush = sys.stdout.flush  # type: ignore
        _default_handler.setLevel(logging.INFO)
        _root_logger.addHandler(_default_handler)
    fmt = RankFormatter(_FORMAT, datefmt=_DATE_FORMAT)
    _default_handler.setFormatter(fmt)
    # Setting this will avoid the message
    # being propagated to the parent logger.
    _root_logger.propagate = False


# The logger is initialized when the module is imported.
# This is thread-safe as the module is only imported once,
# guaranteed by the Python GIL.
_setup_logger()


def init_logger(name: str, rank: Optional[int] = None):
    """Initialize a logger with optional rank information.
    
    Args:
        name: The name of the logger
        rank: The rank of the process (optional)
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Create a filter to add rank to all records
    class RankFilter(logging.Filter):
        def __init__(self, rank):
            self.rank = rank
            
        def filter(self, record):
            record.rank = self.rank
            return True
    
    if rank is not None:
        logger.addFilter(RankFilter(rank))
        
    logger.addHandler(_default_handler)
    logger.propagate = False
    return logger
