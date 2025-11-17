"""Error handling utilities with retry logic and custom exceptions."""

import functools
import logging
import time
from typing import Any, Callable, Optional, Type, Tuple

logger = logging.getLogger(__name__)


class LoFiException(Exception):
    """Base exception for lo-fi generator errors."""

    pass


class ModelLoadError(LoFiException):
    """Raised when model loading fails."""

    pass


class TokenizationError(LoFiException):
    """Raised when MIDI tokenization fails."""

    pass


class GenerationError(LoFiException):
    """Raised when music generation fails."""

    pass


class AudioProcessingError(LoFiException):
    """Raised when audio processing fails."""

    pass


class ConfigurationError(LoFiException):
    """Raised when configuration is invalid."""

    pass


def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable[[Exception, int], None]] = None,
) -> Callable:
    """Decorator to retry a function with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds before first retry
        backoff_factor: Multiplicative factor for delay between retries
        exceptions: Tuple of exception types to catch and retry
        on_retry: Optional callback function called on each retry with (exception, attempt_number)

    Returns:
        Decorated function with retry logic

    Example:
        >>> @retry_with_backoff(max_retries=3, exceptions=(ConnectionError,))
        ... def fetch_data():
        ...     # API call that might fail
        ...     pass
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            delay = initial_delay

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries:
                        logger.error(
                            f"Function {func.__name__} failed after {max_retries} retries: {e}"
                        )
                        raise

                    if on_retry:
                        on_retry(e, attempt + 1)

                    logger.warning(
                        f"Function {func.__name__} failed (attempt {attempt + 1}/{max_retries}): {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )

                    time.sleep(delay)
                    delay *= backoff_factor

            return None  # Should never reach here

        return wrapper

    return decorator


class ErrorHandler:
    """Context manager for handling errors with logging and optional fallback."""

    def __init__(
        self,
        operation_name: str,
        raise_on_error: bool = True,
        fallback_value: Any = None,
        log_level: int = logging.ERROR,
    ):
        """Initialize error handler.

        Args:
            operation_name: Name of the operation for logging
            raise_on_error: Whether to re-raise exceptions
            fallback_value: Value to return if error occurs and raise_on_error=False
            log_level: Logging level for errors
        """
        self.operation_name = operation_name
        self.raise_on_error = raise_on_error
        self.fallback_value = fallback_value
        self.log_level = log_level
        self.logger = logging.getLogger(__name__)

    def __enter__(self):
        """Enter context."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and handle exceptions.

        Returns:
            True to suppress exception, False to propagate
        """
        if exc_type is None:
            return False

        self.logger.log(
            self.log_level,
            f"Error in {self.operation_name}: {exc_type.__name__}: {exc_val}",
            exc_info=True,
        )

        if self.raise_on_error:
            return False  # Propagate exception

        return True  # Suppress exception


def validate_not_none(value: Any, name: str, error_type: Type[LoFiException] = LoFiException) -> Any:
    """Validate that a value is not None.

    Args:
        value: Value to check
        name: Name of the value for error message
        error_type: Exception type to raise

    Returns:
        The value if not None

    Raises:
        error_type: If value is None
    """
    if value is None:
        raise error_type(f"{name} cannot be None")
    return value


def validate_file_exists(file_path: str, error_type: Type[LoFiException] = LoFiException) -> str:
    """Validate that a file exists.

    Args:
        file_path: Path to file
        error_type: Exception type to raise

    Returns:
        The file path if it exists

    Raises:
        error_type: If file doesn't exist
    """
    from pathlib import Path

    path = Path(file_path)
    if not path.exists():
        raise error_type(f"File not found: {file_path}")
    if not path.is_file():
        raise error_type(f"Path is not a file: {file_path}")
    return file_path


def validate_directory_exists(
    dir_path: str, create: bool = False, error_type: Type[LoFiException] = LoFiException
) -> str:
    """Validate that a directory exists.

    Args:
        dir_path: Path to directory
        create: Whether to create directory if it doesn't exist
        error_type: Exception type to raise

    Returns:
        The directory path

    Raises:
        error_type: If directory doesn't exist and create=False
    """
    from pathlib import Path

    path = Path(dir_path)

    if not path.exists():
        if create:
            path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {dir_path}")
        else:
            raise error_type(f"Directory not found: {dir_path}")

    if not path.is_dir():
        raise error_type(f"Path is not a directory: {dir_path}")

    return dir_path


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default on division by zero.

    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value to return on division by zero

    Returns:
        Result of division or default
    """
    try:
        return numerator / denominator if denominator != 0 else default
    except (TypeError, ValueError):
        return default
