"""
Centralized logging configuration for the BGG Data package.
"""

import logging
import sys
from pathlib import Path


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
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
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
