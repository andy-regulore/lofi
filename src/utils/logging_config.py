"""Structured logging configuration with context support."""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import json_log_formatter


class ContextFilter(logging.Filter):
    """Filter to add context information to log records."""

    def __init__(self, context: Optional[Dict[str, Any]] = None):
        """Initialize context filter.

        Args:
            context: Dictionary of context information to add to logs
        """
        super().__init__()
        self.context = context or {}

    def filter(self, record: logging.LogRecord) -> bool:
        """Add context to log record.

        Args:
            record: Log record

        Returns:
            True (always pass through)
        """
        for key, value in self.context.items():
            setattr(record, key, value)
        return True


class JSONFormatter(json_log_formatter.JSONFormatter):
    """Custom JSON formatter for structured logging."""

    def json_record(
        self, message: str, extra: Dict[str, Any], record: logging.LogRecord
    ) -> Dict[str, Any]:
        """Create JSON log record.

        Args:
            message: Log message
            extra: Extra fields
            record: Log record

        Returns:
            Dictionary to be JSON-serialized
        """
        extra["message"] = message
        extra["level"] = record.levelname
        extra["logger"] = record.name
        extra["timestamp"] = self.formatTime(record)

        # Add exception info if present
        if record.exc_info:
            extra["exception"] = self.formatException(record.exc_info)

        # Add custom context fields
        for key in dir(record):
            if key.startswith("_") or key in ["msg", "args", "levelname", "levelno", "name"]:
                continue
            value = getattr(record, key, None)
            if value and not callable(value):
                extra[key] = value

        return extra


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    json_format: bool = False,
    context: Optional[Dict[str, Any]] = None,
) -> None:
    """Set up logging configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for logging
        json_format: Whether to use JSON formatting
        context: Optional context dictionary to add to all logs
    """
    # Convert string to logging level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)

    if json_format:
        console_formatter = JSONFormatter()
    else:
        console_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    console_handler.setFormatter(console_formatter)

    # Add context filter if provided
    if context:
        context_filter = ContextFilter(context)
        console_handler.addFilter(context_filter)

    root_logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Rotating file handler (10MB max, 5 backups)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10 * 1024 * 1024, backupCount=5
        )
        file_handler.setLevel(numeric_level)

        if json_format:
            file_formatter = JSONFormatter()
        else:
            file_formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )

        file_handler.setFormatter(file_formatter)

        if context:
            file_handler.addFilter(context_filter)

        root_logger.addHandler(file_handler)

    # Set specific logger levels
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def get_logger(name: str, context: Optional[Dict[str, Any]] = None) -> logging.Logger:
    """Get a logger with optional context.

    Args:
        name: Logger name (usually __name__)
        context: Optional context dictionary

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    if context:
        context_filter = ContextFilter(context)
        logger.addFilter(context_filter)

    return logger


class LogContext:
    """Context manager for adding temporary context to logs."""

    def __init__(self, logger: logging.Logger, **context: Any):
        """Initialize log context.

        Args:
            logger: Logger to add context to
            **context: Context key-value pairs
        """
        self.logger = logger
        self.context = context
        self.filter: Optional[ContextFilter] = None

    def __enter__(self):
        """Enter context and add filter."""
        self.filter = ContextFilter(self.context)
        self.logger.addFilter(self.filter)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and remove filter."""
        if self.filter:
            self.logger.removeFilter(self.filter)
        return False
