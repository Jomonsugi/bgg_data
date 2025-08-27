"""
Common error handling utilities for the BGG Data package.
"""

import logging
from typing import Optional, Any, Callable
from functools import wraps

logger = logging.getLogger(__name__)


def handle_errors(default_return: Any = None, log_error: bool = True):
    """
    Decorator to handle common exceptions and provide consistent error logging.
    
    Args:
        default_return: Value to return on error
        log_error: Whether to log the error
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_error:
                    logger.error(f"Error in {func.__name__}: {e}")
                return default_return
        return wrapper
    return decorator


def safe_execute(func: Callable, *args, default_return: Any = None, 
                 error_msg: Optional[str] = None, **kwargs) -> Any:
    """
    Safely execute a function with error handling.
    
    Args:
        func: Function to execute
        *args: Arguments for the function
        default_return: Value to return on error
        error_msg: Custom error message
        **kwargs: Keyword arguments for the function
        
    Returns:
        Function result or default_return on error
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if error_msg:
            logger.error(f"{error_msg}: {e}")
        else:
            logger.error(f"Error in {func.__name__}: {e}")
        return default_return
