"""Embedding cache with LRU eviction for performance optimization.

Provides fast retrieval of previously generated embeddings to avoid
redundant Ollama calls and achieve sub-50ms latencies for routing operations.
"""

import hashlib
from typing import Dict, Optional, Tuple
from collections import OrderedDict
import time
import logging
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """LRU cache for embedding vectors with automatic eviction.

    Stores text-to-embedding mappings to accelerate routing operations
    where the same content may be embedded multiple times.
    """

    def __init__(self, max_size: int = 10000, ttl_seconds: Optional[float] = None):
        """Initialize embedding cache.

        Args:
            max_size: Maximum number of embeddings to cache
            ttl_seconds: Optional time-to-live for cache entries
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds

        # Core cache storage: text_hash -> (embedding, timestamp)
        self._cache: OrderedDict[str, Tuple[np.ndarray, float]] = OrderedDict()

        # Performance tracking
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.expirations = 0

    def _hash_text(self, text: str) -> str:
        """Generate hash for text content."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def put(self, text: str, embedding: np.ndarray) -> None:
        """Store embedding in cache.

        Args:
            text: Original text that was embedded
            embedding: Embedding vector to store
        """
        text_hash = self._hash_text(text)

        # Remove existing entry if present (will be re-added at end)
        if text_hash in self._cache:
            del self._cache[text_hash]

        # Evict if at capacity
        if len(self._cache) >= self.max_size:
            _, _ = self._cache.popitem(last=False)  # Remove oldest
            self.evictions += 1
            logger.debug(f"Cache eviction: size limit exceeded")

        # Store new entry with timestamp
        self._cache[text_hash] = (embedding.copy(), time.time())

        logger.debug(f"Cached embedding for text hash {text_hash[:8]}...")

    def get(self, text: str) -> Optional[np.ndarray]:
        """Retrieve embedding from cache.

        Args:
            text: Original text to look up

        Returns:
            Cached embedding vector, or None if not found or expired
        """
        text_hash = self._hash_text(text)

        if text_hash not in self._cache:
            self.misses += 1
            return None

        embedding, timestamp = self._cache[text_hash]

        # Check TTL expiration
        if self.ttl_seconds is not None:
            age = time.time() - timestamp
            if age > self.ttl_seconds:
                del self._cache[text_hash]
                self.expirations += 1
                logger.debug(f"Cache expiration: entry aged {age:.1f}s > {self.ttl_seconds}s")
                return None

        # Move to end (most recently used)
        self._cache.move_to_end(text_hash)
        self.hits += 1

        logger.debug(f"Cache hit for text hash {text_hash[:8]}...")
        return embedding.copy()

    def contains(self, text: str) -> bool:
        """Check if text has cached embedding."""
        text_hash = self._hash_text(text)

        if text_hash not in self._cache:
            return False

        # Check TTL if enabled
        if self.ttl_seconds is not None:
            _, timestamp = self._cache[text_hash]
            age = time.time() - timestamp
            if age > self.ttl_seconds:
                return False

        return True

    def clear(self) -> None:
        """Clear all cached embeddings."""
        self._cache.clear()
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.expirations = 0

    def cleanup_expired(self) -> int:
        """Remove expired entries from cache.

        Returns:
            Number of entries removed
        """
        if self.ttl_seconds is None:
            return 0

        current_time = time.time()
        expired_keys = []

        for text_hash, (_, timestamp) in self._cache.items():
            age = current_time - timestamp
            if age > self.ttl_seconds:
                expired_keys.append(text_hash)

        for key in expired_keys:
            del self._cache[key]

        self.expirations += len(expired_keys)
        logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")

        return len(expired_keys)

    @property
    def size(self) -> int:
        """Current cache size."""
        return len(self._cache)

    @property
    def capacity(self) -> float:
        """Cache utilization (0.0 to 1.0)."""
        if self.max_size == 0:
            return 0.0
        return self.size / self.max_size

    @property
    def hit_rate(self) -> float:
        """Cache hit rate (0.0 to 1.0)."""
        total_accesses = self.hits + self.misses
        if total_accesses == 0:
            return 0.0
        return self.hits / total_accesses

    def get_stats(self) -> Dict[str, float]:
        """Get cache statistics."""
        return {
            'size': self.size,
            'capacity': self.capacity,
            'hit_rate': self.hit_rate,
            'hits': self.hits,
            'misses': self.misses,
            'evictions': self.evictions,
            'expirations': self.expirations
        }

    def __repr__(self) -> str:
        return (".02f")


# Global cache instance for default usage
_default_cache = EmbeddingCache(max_size=5000, ttl_seconds=3600)  # 1 hour TTL


def get_default_cache() -> EmbeddingCache:
    """Get the default global embedding cache."""
    return _default_cache


def cached_generate(generator_func, text: str, cache: Optional[EmbeddingCache] = None) -> np.ndarray:
    """Generate embedding with caching.

    Args:
        generator_func: Function that generates embedding from text
        cache: Cache instance to use (default global cache if None)
        text: Text to embed (if not provided, generator_func should handle it)

    Returns:
        Embedding vector (from cache if available, generated otherwise)
    """
    if cache is None:
        cache = _default_cache

    if text is None:
        raise ValueError("text parameter required for caching")

    # Check cache first
    cached_embedding = cache.get(text)
    if cached_embedding is not None:
        return cached_embedding

    # Generate and cache
    embedding = generator_func(text)
    cache.put(text, embedding)

    return embedding


# Convenience functions for common operations
def warm_cache(texts: list, generator) -> int:
    """Pre-populate cache with embeddings for common texts.

    Args:
        texts: List of texts to pre-embed
        generator: Embedding generator function

    Returns:
        Number of embeddings cached
    """
    cache = get_default_cache()
    cached_count = 0

    for text in texts:
        if not cache.contains(text):
            embedding = generator(text)
            cache.put(text, embedding)
            cached_count += 1

    logger.info(f"Warmed cache with {cached_count} new embeddings")
    return cached_count
