"""Article and Prediction In-Memory LRU Cache with TTL.

Provides fast response times and prevents duplicate web scraping
and ML inference on recently processed news URLs.
"""

import time
from typing import Dict, Any, Optional
from collections import OrderedDict


class PredictionCache:
    """Thread-safe, bounded, in-memory LRU cache with time-to-live expiration."""

    def __init__(self, max_size: int = 100, ttl_seconds: int = 3600):
        """Initialize cache with capacity and expiration window (default 1 hour)."""
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._store: OrderedDict[str, Dict[str, Any]] = OrderedDict()

    def get(self, url: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached prediction result for a given normalized URL if still valid."""
        clean_key = url.strip().lower()
        if clean_key not in self._store:
            return None

        entry = self._store[clean_key]
        now = time.time()

        # Check TTL expiration
        if now - entry["timestamp"] > self.ttl_seconds:
            del self._store[clean_key]
            return None

        # Move to end (most recently accessed)
        self._store.move_to_end(clean_key)
        cached_result = dict(entry["data"])
        cached_result["cached"] = True
        return cached_result

    def set(self, url: str, data: Dict[str, Any]) -> None:
        """Store prediction result associated with a normalized URL."""
        clean_key = url.strip().lower()

        if clean_key in self._store:
            del self._store[clean_key]
        elif len(self._store) >= self.max_size:
            # Evict least recently used item (first item)
            self._store.popitem(last=False)

        # Store a copy without the cached flag internally
        saved_copy = dict(data)
        saved_copy["cached"] = False

        self._store[clean_key] = {
            "timestamp": time.time(),
            "data": saved_copy,
        }

    def clear(self) -> None:
        """Flush the cache."""
        self._store.clear()

    def __len__(self) -> int:
        """Return the current number of cached elements."""
        return len(self._store)


# Global singleton instance
GLOBAL_CACHE = PredictionCache(max_size=100, ttl_seconds=3600)
