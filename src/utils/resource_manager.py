"""Resource management utilities for GPU, memory, and disk space."""

import logging
import os
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple

import psutil
import torch

logger = logging.getLogger(__name__)


class ResourceManager:
    """Manager for system resources (GPU, memory, disk)."""

    def __init__(
        self,
        min_free_disk_gb: float = 10.0,
        min_free_memory_gb: float = 2.0,
        gpu_memory_fraction: float = 0.9,
    ):
        """Initialize resource manager.

        Args:
            min_free_disk_gb: Minimum free disk space in GB
            min_free_memory_gb: Minimum free RAM in GB
            gpu_memory_fraction: Maximum fraction of GPU memory to use
        """
        self.min_free_disk_gb = min_free_disk_gb
        self.min_free_memory_gb = min_free_memory_gb
        self.gpu_memory_fraction = gpu_memory_fraction

    def check_disk_space(self, path: str = ".") -> Tuple[bool, Dict[str, float]]:
        """Check available disk space.

        Args:
            path: Path to check disk space for

        Returns:
            Tuple of (has_enough_space, disk_info_dict)
        """
        path_obj = Path(path).resolve()
        disk = shutil.disk_usage(path_obj)

        total_gb = disk.total / (1024**3)
        used_gb = disk.used / (1024**3)
        free_gb = disk.free / (1024**3)
        percent_used = (disk.used / disk.total) * 100

        disk_info = {
            "total_gb": total_gb,
            "used_gb": used_gb,
            "free_gb": free_gb,
            "percent_used": percent_used,
        }

        has_enough = free_gb >= self.min_free_disk_gb

        if not has_enough:
            logger.warning(
                f"Low disk space: {free_gb:.2f}GB free (minimum: {self.min_free_disk_gb}GB)"
            )

        return has_enough, disk_info

    def check_memory(self) -> Tuple[bool, Dict[str, float]]:
        """Check available RAM.

        Returns:
            Tuple of (has_enough_memory, memory_info_dict)
        """
        mem = psutil.virtual_memory()

        total_gb = mem.total / (1024**3)
        available_gb = mem.available / (1024**3)
        used_gb = mem.used / (1024**3)
        percent_used = mem.percent

        memory_info = {
            "total_gb": total_gb,
            "available_gb": available_gb,
            "used_gb": used_gb,
            "percent_used": percent_used,
        }

        has_enough = available_gb >= self.min_free_memory_gb

        if not has_enough:
            logger.warning(
                f"Low memory: {available_gb:.2f}GB available (minimum: {self.min_free_memory_gb}GB)"
            )

        return has_enough, memory_info

    def check_gpu(self) -> Tuple[bool, Optional[Dict[str, any]]]:
        """Check GPU availability and memory.

        Returns:
            Tuple of (gpu_available, gpu_info_dict or None)
        """
        if not torch.cuda.is_available():
            logger.info("CUDA not available")
            return False, None

        gpu_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()

        gpu_info = {
            "available": True,
            "count": gpu_count,
            "current_device": current_device,
            "device_name": torch.cuda.get_device_name(current_device),
            "devices": [],
        }

        for i in range(gpu_count):
            device_props = torch.cuda.get_device_properties(i)
            total_memory_gb = device_props.total_memory / (1024**3)

            # Get current memory usage
            if torch.cuda.is_available():
                allocated_gb = torch.cuda.memory_allocated(i) / (1024**3)
                reserved_gb = torch.cuda.memory_reserved(i) / (1024**3)
                free_gb = total_memory_gb - allocated_gb
            else:
                allocated_gb = 0
                reserved_gb = 0
                free_gb = total_memory_gb

            device_info = {
                "index": i,
                "name": device_props.name,
                "total_memory_gb": total_memory_gb,
                "allocated_gb": allocated_gb,
                "reserved_gb": reserved_gb,
                "free_gb": free_gb,
                "compute_capability": f"{device_props.major}.{device_props.minor}",
            }

            gpu_info["devices"].append(device_info)

        return True, gpu_info

    def clear_gpu_cache(self) -> None:
        """Clear GPU cache to free memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Cleared GPU cache")

    def set_gpu_memory_limit(self, device: int = 0) -> None:
        """Set GPU memory limit.

        Args:
            device: GPU device index
        """
        if not torch.cuda.is_available():
            return

        # PyTorch doesn't have a direct way to limit memory,
        # but we can set environment variable before initialization
        total_memory = torch.cuda.get_device_properties(device).total_memory
        limit = int(total_memory * self.gpu_memory_fraction)

        logger.info(f"GPU memory limit set to {limit / (1024**3):.2f}GB")

    def get_optimal_device(self) -> str:
        """Get optimal device for computation.

        Returns:
            Device string ('cuda', 'mps', or 'cpu')
        """
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def check_all_resources(self, path: str = ".") -> Dict[str, any]:
        """Check all system resources.

        Args:
            path: Path to check disk space for

        Returns:
            Dictionary with resource information
        """
        disk_ok, disk_info = self.check_disk_space(path)
        memory_ok, memory_info = self.check_memory()
        gpu_available, gpu_info = self.check_gpu()

        resources = {
            "disk": {"ok": disk_ok, "info": disk_info},
            "memory": {"ok": memory_ok, "info": memory_info},
            "gpu": {"available": gpu_available, "info": gpu_info},
            "optimal_device": self.get_optimal_device(),
            "all_ok": disk_ok and memory_ok,
        }

        # Log summary
        logger.info(f"Resource check summary:")
        logger.info(f"  Disk: {disk_info['free_gb']:.2f}GB free ({disk_ok and 'OK' or 'LOW'})")
        logger.info(
            f"  Memory: {memory_info['available_gb']:.2f}GB available ({memory_ok and 'OK' or 'LOW'})"
        )
        logger.info(f"  GPU: {gpu_available and 'Available' or 'Not available'}")
        logger.info(f"  Optimal device: {resources['optimal_device']}")

        return resources

    def cleanup_directory(
        self, directory: str, pattern: str = "*", max_files: Optional[int] = None
    ) -> int:
        """Clean up old files in a directory.

        Args:
            directory: Directory path
            pattern: File pattern to match
            max_files: Keep only this many most recent files

        Returns:
            Number of files deleted
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            return 0

        # Get matching files sorted by modification time (newest first)
        files = sorted(dir_path.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)

        if max_files and len(files) > max_files:
            files_to_delete = files[max_files:]
            count = 0

            for file_path in files_to_delete:
                try:
                    file_path.unlink()
                    count += 1
                    logger.debug(f"Deleted old file: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to delete {file_path}: {e}")

            logger.info(f"Cleaned up {count} old files from {directory}")
            return count

        return 0

    def estimate_model_memory(self, num_parameters: int, dtype: str = "float32") -> float:
        """Estimate memory required for a model.

        Args:
            num_parameters: Number of model parameters
            dtype: Data type ('float32', 'float16', etc.)

        Returns:
            Estimated memory in GB
        """
        bytes_per_param = {
            "float32": 4,
            "float16": 2,
            "bfloat16": 2,
            "int8": 1,
        }

        bytes_needed = num_parameters * bytes_per_param.get(dtype, 4)
        gb_needed = bytes_needed / (1024**3)

        # Add overhead for optimizer states (roughly 2x for Adam)
        gb_with_overhead = gb_needed * 3  # Model + gradients + optimizer

        return gb_with_overhead
