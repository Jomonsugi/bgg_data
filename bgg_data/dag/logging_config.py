"""
Centralized logging configuration for the BGG Data package.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from .config import LOGS_DIR


def setup_logging(log_file: str = "bgg_data.log", level: int = logging.INFO) -> None:
    """
    Set up centralized logging for the BGG Data package.
    
    Args:
        log_file: Name of the log file
        level: Logging level
    """
    # Avoid duplicate handlers if already configured
    if logging.getLogger().handlers:
        return
    
    # Ensure logs directory exists for per-run files when a path inside LOGS_DIR is provided
    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    
    # Default to logs dir if bare filename provided
    log_path = Path(log_file)
    if not log_path.is_absolute():
        log_path = LOGS_DIR / log_path.name
    # File handler
    file_handler = logging.FileHandler(str(log_path))
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    # Reduce noise from external libraries
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('selenium').setLevel(logging.WARNING)
