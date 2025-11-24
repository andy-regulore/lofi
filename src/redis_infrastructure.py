"""
Redis Queue and Caching Infrastructure

Provides distributed job queue management and caching layer for improved
performance and scalability.

Features:
- Redis-based job queue (alternative to in-memory)
- Priority queue system
- Result caching
- Session caching
- API response caching
- Cache invalidation
- Queue monitoring and statistics

Author: Claude
License: MIT
"""

import hashlib
import json
import logging
import pickle
import time
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional

import redis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Priority(Enum):
    """Job priority levels."""

    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


class RedisQueue:
    """
    Redis-based distributed job queue with priority support.
    """

    def __init__(
        self, host: str = "localhost", port: int = 6379, db: int = 0, password: Optional[str] = None
    ):
        """
        Initialize Redis queue.

        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            password: Redis password (if required)
        """
        try:
            self.redis_client = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=False,  # We'll handle encoding
            )
            # Test connection
            self.redis_client.ping()
            logger.info(f"Connected to Redis at {host}:{port}")
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            logger.warning("Falling back to mock Redis for testing")
            self.redis_client = None

        # Queue keys by priority
        self.queue_keys = {
            Priority.URGENT: "queue:urgent",
            Priority.HIGH: "queue:high",
            Priority.NORMAL: "queue:normal",
            Priority.LOW: "queue:low",
        }

        # Processing and completed queues
        self.processing_key = "queue:processing"
        self.completed_key = "queue:completed"
        self.failed_key = "queue:failed"

    def enqueue(
        self, task_data: Dict, priority: Priority = Priority.NORMAL, task_id: Optional[str] = None
    ) -> str:
        """
        Add a task to the queue.

        Args:
            task_data: Task data dictionary
            priority: Task priority
            task_id: Optional task ID (generated if not provided)

        Returns:
            Task ID
        """
        if not self.redis_client:
            logger.warning("Redis not available, task not queued")
            return ""

        # Generate task ID if not provided
        if not task_id:
            task_id = f"task_{int(time.time() * 1000)}"

        # Prepare task
        task = {
            "task_id": task_id,
            "data": task_data,
            "priority": priority.value,
            "enqueued_at": datetime.now().isoformat(),
            "status": "queued",
        }

        # Serialize task
        serialized_task = pickle.dumps(task)

        # Add to appropriate priority queue
        queue_key = self.queue_keys[priority]
        self.redis_client.rpush(queue_key, serialized_task)

        # Store task metadata
        self.redis_client.hset(
            f"task:{task_id}",
            mapping={
                "status": "queued",
                "priority": priority.value,
                "enqueued_at": task["enqueued_at"],
            },
        )

        logger.info(f"Enqueued task {task_id} with priority {priority.name}")
        return task_id

    def dequeue(self, block: bool = True, timeout: int = 0) -> Optional[Dict]:
        """
        Get next task from queue (highest priority first).

        Args:
            block: Whether to block waiting for task
            timeout: Timeout in seconds (0 = wait forever)

        Returns:
            Task dictionary or None
        """
        if not self.redis_client:
            return None

        # Try queues in priority order
        for priority in [Priority.URGENT, Priority.HIGH, Priority.NORMAL, Priority.LOW]:
            queue_key = self.queue_keys[priority]

            if block:
                # Blocking pop with timeout
                result = self.redis_client.blpop(queue_key, timeout=timeout)
                if result:
                    _, serialized_task = result
                    task = pickle.loads(serialized_task)
                    self._mark_processing(task)
                    return task
            else:
                # Non-blocking pop
                serialized_task = self.redis_client.lpop(queue_key)
                if serialized_task:
                    task = pickle.loads(serialized_task)
                    self._mark_processing(task)
                    return task

        return None

    def _mark_processing(self, task: Dict):
        """Mark task as being processed."""
        task_id = task["task_id"]
        task["status"] = "processing"
        task["started_at"] = datetime.now().isoformat()

        # Update task metadata
        self.redis_client.hset(
            f"task:{task_id}", mapping={"status": "processing", "started_at": task["started_at"]}
        )

        # Add to processing queue
        self.redis_client.rpush(self.processing_key, pickle.dumps(task))

        logger.info(f"Started processing task {task_id}")

    def complete_task(self, task_id: str, result: Any = None):
        """
        Mark task as completed.

        Args:
            task_id: Task ID
            result: Optional task result
        """
        if not self.redis_client:
            return

        completed_at = datetime.now().isoformat()

        # Update task metadata
        self.redis_client.hset(
            f"task:{task_id}", mapping={"status": "completed", "completed_at": completed_at}
        )

        # Store result if provided
        if result is not None:
            self.redis_client.set(f"result:{task_id}", pickle.dumps(result), ex=86400)  # 24h TTL

        # Remove from processing queue
        # (Note: This is simplified - production would need proper cleanup)

        logger.info(f"Completed task {task_id}")

    def fail_task(self, task_id: str, error: str):
        """
        Mark task as failed.

        Args:
            task_id: Task ID
            error: Error message
        """
        if not self.redis_client:
            return

        failed_at = datetime.now().isoformat()

        # Update task metadata
        self.redis_client.hset(
            f"task:{task_id}", mapping={"status": "failed", "failed_at": failed_at, "error": error}
        )

        logger.error(f"Task {task_id} failed: {error}")

    def get_task_status(self, task_id: str) -> Optional[Dict]:
        """Get task status and metadata."""
        if not self.redis_client:
            return None

        task_data = self.redis_client.hgetall(f"task:{task_id}")

        if not task_data:
            return None

        # Decode bytes to strings
        return {k.decode(): v.decode() for k, v in task_data.items()}

    def get_task_result(self, task_id: str) -> Any:
        """Get task result if available."""
        if not self.redis_client:
            return None

        result_data = self.redis_client.get(f"result:{task_id}")

        if result_data:
            return pickle.loads(result_data)

        return None

    def get_queue_stats(self) -> Dict:
        """Get queue statistics."""
        if not self.redis_client:
            return {}

        stats = {"queued": {}, "processing": 0, "total_queued": 0}

        # Count tasks in each priority queue
        for priority, queue_key in self.queue_keys.items():
            count = self.redis_client.llen(queue_key)
            stats["queued"][priority.name] = count
            stats["total_queued"] += count

        # Count processing tasks
        stats["processing"] = self.redis_client.llen(self.processing_key)

        return stats

    def clear_queue(self, priority: Optional[Priority] = None):
        """
        Clear queue(s).

        Args:
            priority: Specific priority to clear, or None for all
        """
        if not self.redis_client:
            return

        if priority:
            queue_key = self.queue_keys[priority]
            self.redis_client.delete(queue_key)
            logger.info(f"Cleared {priority.name} queue")
        else:
            for queue_key in self.queue_keys.values():
                self.redis_client.delete(queue_key)
            logger.info("Cleared all queues")


class RedisCache:
    """
    Redis-based caching layer with automatic expiration.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 1,
        password: Optional[str] = None,
        default_ttl: int = 3600,
    ):
        """
        Initialize Redis cache.

        Args:
            host: Redis host
            port: Redis port
            db: Redis database number (use different DB than queue)
            password: Redis password
            default_ttl: Default TTL in seconds (1 hour)
        """
        try:
            self.redis_client = redis.Redis(
                host=host, port=port, db=db, password=password, decode_responses=False
            )
            self.redis_client.ping()
            logger.info(f"Connected to Redis cache at {host}:{port}/{db}")
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis cache: {e}")
            self.redis_client = None

        self.default_ttl = default_ttl

    def _generate_key(self, key: str, namespace: str = "cache") -> str:
        """Generate namespaced cache key."""
        return f"{namespace}:{key}"

    def set(self, key: str, value: Any, ttl: Optional[int] = None, namespace: str = "cache"):
        """
        Set cache value.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (uses default if None)
            namespace: Cache namespace
        """
        if not self.redis_client:
            return

        cache_key = self._generate_key(key, namespace)
        serialized_value = pickle.dumps(value)

        ttl = ttl or self.default_ttl
        self.redis_client.setex(cache_key, ttl, serialized_value)

        logger.debug(f"Cached {cache_key} (TTL: {ttl}s)")

    def get(self, key: str, namespace: str = "cache") -> Optional[Any]:
        """
        Get cache value.

        Args:
            key: Cache key
            namespace: Cache namespace

        Returns:
            Cached value or None
        """
        if not self.redis_client:
            return None

        cache_key = self._generate_key(key, namespace)
        cached_data = self.redis_client.get(cache_key)

        if cached_data:
            logger.debug(f"Cache hit: {cache_key}")
            return pickle.loads(cached_data)

        logger.debug(f"Cache miss: {cache_key}")
        return None

    def delete(self, key: str, namespace: str = "cache"):
        """Delete cache entry."""
        if not self.redis_client:
            return

        cache_key = self._generate_key(key, namespace)
        self.redis_client.delete(cache_key)
        logger.debug(f"Deleted cache: {cache_key}")

    def clear_namespace(self, namespace: str):
        """Clear all keys in a namespace."""
        if not self.redis_client:
            return

        pattern = f"{namespace}:*"
        keys = self.redis_client.keys(pattern)

        if keys:
            self.redis_client.delete(*keys)
            logger.info(f"Cleared {len(keys)} keys from namespace '{namespace}'")

    def exists(self, key: str, namespace: str = "cache") -> bool:
        """Check if key exists in cache."""
        if not self.redis_client:
            return False

        cache_key = self._generate_key(key, namespace)
        return bool(self.redis_client.exists(cache_key))

    def get_ttl(self, key: str, namespace: str = "cache") -> int:
        """Get remaining TTL for key."""
        if not self.redis_client:
            return -1

        cache_key = self._generate_key(key, namespace)
        return self.redis_client.ttl(cache_key)


def cached(ttl: int = 3600, namespace: str = "function", key_prefix: str = ""):
    """
    Decorator for caching function results in Redis.

    Args:
        ttl: Cache TTL in seconds
        namespace: Cache namespace
        key_prefix: Optional key prefix

    Usage:
        @cached(ttl=300, namespace="api", key_prefix="youtube")
        def get_video_stats(video_id):
            # Expensive API call
            return stats
    """

    def decorator(func: Callable) -> Callable:
        cache = RedisCache(default_ttl=ttl)

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key from function name and arguments
            key_parts = [key_prefix, func.__name__] if key_prefix else [func.__name__]

            # Add args to key
            if args:
                args_str = "_".join(str(arg) for arg in args)
                key_parts.append(args_str)

            # Add kwargs to key
            if kwargs:
                kwargs_str = "_".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
                key_parts.append(kwargs_str)

            cache_key = ":".join(key_parts)

            # Try to get from cache
            cached_result = cache.get(cache_key, namespace)

            if cached_result is not None:
                logger.info(f"Returning cached result for {func.__name__}")
                return cached_result

            # Execute function
            result = func(*args, **kwargs)

            # Cache result
            cache.set(cache_key, result, ttl, namespace)

            return result

        # Add cache control methods
        wrapper.clear_cache = lambda: cache.clear_namespace(namespace)
        wrapper.cache = cache

        return wrapper

    return decorator


class CacheManager:
    """
    High-level cache management with common patterns.
    """

    def __init__(self, cache: Optional[RedisCache] = None):
        """
        Initialize cache manager.

        Args:
            cache: RedisCache instance (creates new if None)
        """
        self.cache = cache or RedisCache()

    def cache_api_response(self, api_name: str, endpoint: str, response: Any, ttl: int = 300):
        """
        Cache API response.

        Args:
            api_name: API name (e.g., "youtube", "spotify")
            endpoint: Endpoint identifier
            response: Response data
            ttl: Cache TTL (default 5 minutes)
        """
        key = f"{api_name}:{endpoint}"
        self.cache.set(key, response, ttl, namespace="api")

    def get_api_response(self, api_name: str, endpoint: str) -> Optional[Any]:
        """Get cached API response."""
        key = f"{api_name}:{endpoint}"
        return self.cache.get(key, namespace="api")

    def cache_generated_content(
        self, content_type: str, content_id: str, content: Any, ttl: int = 86400
    ):
        """
        Cache generated content (music, videos, etc.).

        Args:
            content_type: Content type (e.g., "music", "video", "thumbnail")
            content_id: Content identifier
            content: Content data
            ttl: Cache TTL (default 24 hours)
        """
        key = f"{content_type}:{content_id}"
        self.cache.set(key, content, ttl, namespace="content")

    def get_generated_content(self, content_type: str, content_id: str) -> Optional[Any]:
        """Get cached generated content."""
        key = f"{content_type}:{content_id}"
        return self.cache.get(key, namespace="content")

    def cache_analytics(self, metric_name: str, data: Any, ttl: int = 3600):
        """
        Cache analytics data.

        Args:
            metric_name: Metric identifier
            data: Analytics data
            ttl: Cache TTL (default 1 hour)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H")
        key = f"{metric_name}:{timestamp}"
        self.cache.set(key, data, ttl, namespace="analytics")

    def invalidate_api_cache(self, api_name: str):
        """Invalidate all cached responses for an API."""
        self.cache.clear_namespace(f"api:{api_name}")

    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        if not self.cache.redis_client:
            return {}

        info = self.cache.redis_client.info("stats")

        return {
            "hits": info.get("keyspace_hits", 0),
            "misses": info.get("keyspace_misses", 0),
            "hit_rate": self._calculate_hit_rate(info),
            "total_keys": self.cache.redis_client.dbsize(),
        }

    def _calculate_hit_rate(self, info: Dict) -> float:
        """Calculate cache hit rate."""
        hits = info.get("keyspace_hits", 0)
        misses = info.get("keyspace_misses", 0)
        total = hits + misses

        if total == 0:
            return 0.0

        return (hits / total) * 100


# Example usage
if __name__ == "__main__":
    print("=== Redis Queue Example ===")

    # Initialize queue
    queue = RedisQueue()

    if queue.redis_client:
        # Enqueue some tasks
        task_ids = []
        task_ids.append(queue.enqueue({"type": "generate_music", "count": 10}, Priority.HIGH))
        task_ids.append(queue.enqueue({"type": "create_video", "track_id": "123"}, Priority.NORMAL))
        task_ids.append(queue.enqueue({"type": "upload_youtube", "video_id": "456"}, Priority.LOW))

        # Check queue stats
        stats = queue.get_queue_stats()
        print(f"\nQueue Stats: {stats}")

        # Dequeue and process
        task = queue.dequeue(block=False)
        if task:
            print(f"\nProcessing: {task['data']}")
            # Simulate processing
            time.sleep(0.1)
            queue.complete_task(task["task_id"], result={"status": "success"})

            # Check status
            status = queue.get_task_status(task["task_id"])
            print(f"Task Status: {status}")

    print("\n=== Redis Cache Example ===")

    # Initialize cache
    cache = RedisCache()

    if cache.redis_client:
        # Cache some data
        cache.set("video:123", {"views": 1000, "likes": 50}, ttl=60)
        cache.set("channel:abc", {"subscribers": 10000}, ttl=300)

        # Retrieve from cache
        video_data = cache.get("video:123")
        print(f"\nCached video data: {video_data}")

        # Check TTL
        ttl = cache.get_ttl("video:123")
        print(f"TTL remaining: {ttl}s")

    print("\n=== Cached Decorator Example ===")

    @cached(ttl=30, namespace="example")
    def expensive_computation(x, y):
        """Simulate expensive computation."""
        print(f"Computing {x} + {y}...")
        time.sleep(0.5)  # Simulate slow operation
        return x + y

    # First call - executes function
    result1 = expensive_computation(5, 3)
    print(f"Result 1: {result1}")

    # Second call - returns cached result (much faster)
    result2 = expensive_computation(5, 3)
    print(f"Result 2: {result2}")

    print("\n=== Cache Manager Example ===")

    manager = CacheManager()

    if manager.cache.redis_client:
        # Cache API response
        manager.cache_api_response("youtube", "videos/list?id=123", {"title": "My Video"})

        # Retrieve API response
        response = manager.get_api_response("youtube", "videos/list?id=123")
        print(f"\nCached API response: {response}")

        # Cache statistics
        stats = manager.get_cache_stats()
        print(f"\nCache Stats: {stats}")
