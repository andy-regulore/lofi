"""Security utilities for safe file operations and input validation."""

import logging
import os
import re
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


class SecurePathHandler:
    """Handler for secure file path operations."""

    def __init__(self, allowed_base_paths: Optional[List[str]] = None):
        """Initialize secure path handler.

        Args:
            allowed_base_paths: List of allowed base directory paths
        """
        self.allowed_base_paths = [Path(p).resolve() for p in (allowed_base_paths or [])]

    def validate_path(self, file_path: str, must_exist: bool = False) -> Path:
        """Validate and sanitize a file path.

        Args:
            file_path: Path to validate
            must_exist: Whether the path must exist

        Returns:
            Validated Path object

        Raises:
            ValueError: If path is invalid or unsafe
        """
        # Resolve to absolute path
        path = Path(file_path).resolve()

        # Check for path traversal attempts
        if self._is_path_traversal(str(file_path)):
            raise ValueError(f"Path traversal detected: {file_path}")

        # Check against allowed base paths
        if self.allowed_base_paths:
            if not any(self._is_subpath(path, base) for base in self.allowed_base_paths):
                raise ValueError(f"Path outside allowed directories: {path}")

        # Check existence if required
        if must_exist and not path.exists():
            raise ValueError(f"Path does not exist: {path}")

        return path

    @staticmethod
    def _is_path_traversal(path_str: str) -> bool:
        """Check if path contains traversal attempts.

        Args:
            path_str: Path string to check

        Returns:
            True if path traversal detected
        """
        dangerous_patterns = [
            '..',  # Parent directory
            '~',  # Home directory
            '\\',  # Windows path separators (on Unix)
        ]

        # Normalize path separators
        normalized = path_str.replace('\\', '/')

        # Check for dangerous patterns
        for pattern in dangerous_patterns:
            if pattern in normalized:
                return True

        return False

    @staticmethod
    def _is_subpath(path: Path, base: Path) -> bool:
        """Check if path is under base directory.

        Args:
            path: Path to check
            base: Base directory

        Returns:
            True if path is under base
        """
        try:
            path.resolve().relative_to(base.resolve())
            return True
        except ValueError:
            return False

    def safe_join(self, base: str, *parts: str) -> Path:
        """Safely join path components.

        Args:
            base: Base directory
            *parts: Path components to join

        Returns:
            Validated Path object

        Raises:
            ValueError: If resulting path is unsafe
        """
        # Join components
        path = Path(base).joinpath(*parts)

        # Validate
        return self.validate_path(str(path))

    @staticmethod
    def sanitize_filename(filename: str, max_length: int = 255) -> str:
        """Sanitize a filename to be safe for file system.

        Args:
            filename: Original filename
            max_length: Maximum filename length

        Returns:
            Sanitized filename
        """
        # Remove or replace dangerous characters
        # Keep only alphanumeric, dash, underscore, dot
        sanitized = re.sub(r'[^\w\s\-\.]', '_', filename)

        # Replace multiple spaces/underscores with single
        sanitized = re.sub(r'[\s_]+', '_', sanitized)

        # Remove leading/trailing dots and spaces
        sanitized = sanitized.strip('. ')

        # Truncate to max length
        if len(sanitized) > max_length:
            name, ext = os.path.splitext(sanitized)
            max_name_length = max_length - len(ext)
            sanitized = name[:max_name_length] + ext

        # Ensure not empty
        if not sanitized:
            sanitized = 'unnamed_file'

        return sanitized


def validate_config_value(
    value: any,
    value_type: type,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    allowed_values: Optional[List[any]] = None,
) -> any:
    """Validate a configuration value.

    Args:
        value: Value to validate
        value_type: Expected type
        min_val: Minimum value (for numeric types)
        max_val: Maximum value (for numeric types)
        allowed_values: List of allowed values

    Returns:
        Validated value

    Raises:
        ValueError: If validation fails
    """
    # Type check
    if not isinstance(value, value_type):
        raise ValueError(f"Expected type {value_type.__name__}, got {type(value).__name__}")

    # Range check for numeric values
    if min_val is not None and value < min_val:
        raise ValueError(f"Value {value} below minimum {min_val}")

    if max_val is not None and value > max_val:
        raise ValueError(f"Value {value} above maximum {max_val}")

    # Allowed values check
    if allowed_values is not None and value not in allowed_values:
        raise ValueError(f"Value {value} not in allowed values: {allowed_values}")

    return value


def sanitize_shell_command(command: str) -> str:
    """Sanitize a shell command to prevent injection.

    Args:
        command: Command to sanitize

    Returns:
        Sanitized command

    Raises:
        ValueError: If command contains dangerous patterns
    """
    # Check for dangerous patterns
    dangerous_patterns = [
        ';',  # Command separator
        '|',  # Pipe
        '&',  # Background/AND
        '$',  # Variable expansion
        '`',  # Command substitution
        '>',  # Redirect
        '<',  # Redirect
        '\n',  # Newline
        '\r',  # Carriage return
    ]

    for pattern in dangerous_patterns:
        if pattern in command:
            raise ValueError(f"Dangerous pattern '{pattern}' found in command")

    return command


class InputValidator:
    """Validator for user inputs."""

    @staticmethod
    def validate_tempo(tempo: float) -> float:
        """Validate tempo value.

        Args:
            tempo: Tempo in BPM

        Returns:
            Validated tempo

        Raises:
            ValueError: If tempo is invalid
        """
        if not 20 <= tempo <= 300:
            raise ValueError(f"Tempo {tempo} outside valid range [20, 300] BPM")
        return tempo

    @staticmethod
    def validate_key(key: str) -> str:
        """Validate musical key.

        Args:
            key: Musical key (e.g., 'C', 'Am', 'F#')

        Returns:
            Validated key

        Raises:
            ValueError: If key is invalid
        """
        valid_keys = [
            'C',
            'C#',
            'D',
            'D#',
            'E',
            'F',
            'F#',
            'G',
            'G#',
            'A',
            'A#',
            'B',
            'Cm',
            'C#m',
            'Dm',
            'D#m',
            'Em',
            'Fm',
            'F#m',
            'Gm',
            'G#m',
            'Am',
            'A#m',
            'Bm',
        ]

        if key not in valid_keys:
            raise ValueError(f"Invalid key: {key}. Must be one of {valid_keys}")

        return key

    @staticmethod
    def validate_temperature(temperature: float) -> float:
        """Validate sampling temperature.

        Args:
            temperature: Temperature value

        Returns:
            Validated temperature

        Raises:
            ValueError: If temperature is invalid
        """
        if not 0.1 <= temperature <= 2.0:
            raise ValueError(f"Temperature {temperature} outside valid range [0.1, 2.0]")
        return temperature

    @staticmethod
    def validate_file_extension(file_path: str, allowed_extensions: List[str]) -> str:
        """Validate file extension.

        Args:
            file_path: File path
            allowed_extensions: List of allowed extensions (with dots)

        Returns:
            Validated file path

        Raises:
            ValueError: If extension is not allowed
        """
        path = Path(file_path)
        ext = path.suffix.lower()

        if ext not in allowed_extensions:
            raise ValueError(f"File extension {ext} not allowed. Allowed: {allowed_extensions}")

        return file_path
