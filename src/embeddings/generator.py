"""Embedding generation using Ollama integration.

Provides interface to generate vector embeddings using Ollama-hosted
nomic-embed-text model for CyberMesh vector shard creation.
"""

import requests
import json
import time
from typing import List, Optional, Dict, Any
import os
import logging
from functools import lru_cache
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Ollama-based embedding generator for vector shards.

    Handles generation of embeddings using nomic-embed-text model
    with caching and performance optimization for routing operations.
    """

    def __init__(self, model_name: str = "nomic-embed-text:137m-v1.5-fp16",
                 host: str = "http://localhost:11434",
                 timeout: float = 30.0,
                 max_retries: int = 3):
        """Initialize embedding generator.

        Args:
            model_name: Ollama model name to use
            host: Ollama server host URL
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts on failure
        """
        self.model_name = model_name
        self.host = host.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries

        # Performance tracking
        self.request_count = 0
        self.total_request_time = 0.0
        self.cache_hits = 0
        self.last_warmup = None

        self._test_connection()

    def _test_connection(self) -> None:
        """Test connection to Ollama server."""
        try:
            response = requests.get(f"{self.host}/api/version", timeout=5.0)
            response.raise_for_status()
            logger.info(f"Connected to Ollama at {self.host}")
        except requests.RequestException as e:
            logger.warning(f"Could not connect to Ollama: {e}. Embeddings will fail until server is available.")

    def generate(self, text: str) -> np.ndarray:
        """Generate embedding vector for input text.

        Args:
            text: Input text to embed

        Returns:
            Normalized embedding vector as numpy array

        Raises:
            RuntimeError: If embedding generation fails
        """
        if not text or not text.strip():
            raise ValueError("Text input cannot be empty")

        start_time = time.time()
        embedding: Optional[np.ndarray] = None

        for attempt in range(self.max_retries):
            try:
                payload = {
                    "model": self.model_name,
                    "prompt": text.strip()
                }

                response = requests.post(
                    f"{self.host}/api/embeddings",
                    json=payload,
                    timeout=self.timeout
                )
                response.raise_for_status()

                result = response.json()

                if 'embedding' not in result:
                    raise RuntimeError(f"No embedding in response: {result}")

                embedding = np.array(result['embedding'], dtype=np.float32)

                # Validate embedding
                if embedding.size == 0:
                    raise RuntimeError("Generated embedding is empty")

                if not np.isfinite(embedding).all():
                    raise RuntimeError("Embedding contains invalid values")

                # Normalize to unit length for cosine similarity
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm

                # Success - break out of retry loop
                break

            except requests.RequestException as e:
                if attempt == self.max_retries - 1:
                    raise RuntimeError(f"Embedding generation failed after {self.max_retries} attempts: {e}")
                logger.warning(f"Embedding attempt {attempt + 1} failed: {e}, retrying...")
                time.sleep(0.5 * (attempt + 1))  # Exponential backoff

        # At this point, embedding is guaranteed to be set if we didn't raise
        assert embedding is not None, "Embedding should be set after successful generation"

        # Update performance stats and return
        self.request_count += 1
        request_time = time.time() - start_time
        self.total_request_time += request_time

        logger.debug(".2f")
        return embedding

    def generate_batch(self, texts: List[str], batch_size: int = 5) -> List[np.ndarray]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of input texts
            batch_size: Number of concurrent requests (limited to avoid overload)

        Returns:
            List of embedding vectors
        """
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            # Process batch sequentially (Ollama doesn't support batching yet)
            for text in batch:
                try:
                    embedding = self.generate(text)
                    embeddings.append(embedding)
                except Exception as e:
                    logger.error(f"Failed to generate embedding for text: {text[:50]}...: {e}")
                    # Add zero vector as fallback
                    embeddings.append(np.zeros(768, dtype=np.float32))

            # Small delay between batches to be gentle on server
            if i + batch_size < len(texts):
                time.sleep(0.1)

        return embeddings

    def warmup(self) -> bool:
        """Warm up the model with a test embedding.

        Returns:
            True if warmup successful
        """
        try:
            self.last_warmup = time.time()
            test_embedding = self.generate("test input for model warmup")
            logger.info(".2f")
            return True
        except Exception as e:
            logger.warning(f"Warmup failed: {e}")
            return False

    @property
    def average_latency(self) -> float:
        """Average embedding generation latency in seconds."""
        if self.request_count == 0:
            return 0.0
        return self.total_request_time / self.request_count

    @property
    def throughput(self) -> float:
        """Embeddings per second."""
        if self.total_request_time == 0:
            return 0.0
        return self.request_count / self.total_request_time

    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            'request_count': self.request_count,
            'total_request_time': self.total_request_time,
            'average_latency': self.average_latency,
            'throughput': self.throughput,
            'cache_hits': self.cache_hits,
            'last_warmup': self.last_warmup
        }

    def __repr__(self) -> str:
        return (f"EmbeddingGenerator(model={self.model_name}, "
                ".2f")

# Global generator instance (can be configured via environment)
def create_generator() -> EmbeddingGenerator:
    """Create embedding generator from environment configuration."""
    model_name = os.getenv('OLLAMA_MODEL', 'nomic-embed-text:137m-v1.5-fp16')
    host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')

    return EmbeddingGenerator(model_name=model_name, host=host)


# Default instance
default_generator = create_generator()
