"""Vector similarity and distance metrics for embedding comparison.

Provides optimized functions for cosine similarity, distance calculations,
and similarity-based ranking used in shard routing decisions.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Callable
import numba
from numba import jit
import logging

logger = logging.getLogger(__name__)


@jit(nopython=True, cache=True)
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        a: First vector (assumed normalized)
        b: Second vector (assumed normalized)

    Returns:
        Similarity score between -1 and 1 (1 = identical, -1 = opposite)
    """
    dot_product = np.dot(a, b)

    # Clamp to [-1, 1] due to floating point precision
    return max(-1.0, min(1.0, dot_product))


@jit(nopython=True, cache=True)
def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine distance (1 - similarity) between vectors.

    Args:
        a: First vector (assumed normalized)
        b: Second vector (assumed normalized)

    Returns:
        Distance score between 0 and 2 (0 = identical, 2 = opposite)
    """
    sim = cosine_similarity(a, b)
    return 1.0 - sim


@jit(nopython=True, cache=True)
def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Euclidean distance between vectors.

    Args:
        a: First vector
        b: Second vector

    Returns:
        Euclidean distance
    """
    return np.linalg.norm(a - b)


@jit(nopython=True, cache=True)
def manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Manhattan (L1) distance between vectors.

    Args:
        a: First vector
        b: Second vector

    Returns:
        Manhattan distance
    """
    return np.sum(np.abs(a - b))


def normalize_vector(v: np.ndarray) -> np.ndarray:
    """Normalize vector to unit length.

    Args:
        v: Input vector

    Returns:
        Normalized vector
    """
    norm = np.linalg.norm(v)
    if norm > 0:
        return v / norm
    return v


def batch_cosine_similarity(query: np.ndarray, candidates: List[np.ndarray]) -> np.ndarray:
    """Compute cosine similarities between query and multiple candidates.

    Args:
        query: Query vector (will be normalized)
        candidates: List of candidate vectors (will be normalized)

    Returns:
        Array of similarity scores
    """
    # Normalize query
    query_norm = normalize_vector(query)

    # Normalize candidates and compute similarities
    similarities = np.zeros(len(candidates))
    for i, candidate in enumerate(candidates):
        candidate_norm = normalize_vector(candidate)
        similarities[i] = cosine_similarity(query_norm, candidate_norm)

    return similarities


def find_most_similar(query: np.ndarray, candidates: List[np.ndarray],
                     top_k: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Find most similar vectors to query.

    Args:
        query: Query vector
        candidates: List of candidate vectors
        top_k: Number of top matches to return

    Returns:
        Tuple of (similarities, indices) sorted by similarity descending
    """
    similarities = batch_cosine_similarity(query, candidates)

    # Get top-k indices sorted by similarity (highest first)
    if top_k >= len(similarities):
        top_indices = np.argsort(similarities)[::-1]
        top_similarities = similarities[top_indices]
    else:
        top_indices = np.argpartition(similarities, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]
        top_similarities = similarities[top_indices]

    return top_similarities, top_indices


def compute_similarity_matrix(vectors: List[np.ndarray],
                             metric: str = 'cosine') -> np.ndarray:
    """Compute pairwise similarity matrix for a set of vectors.

    Args:
        vectors: List of vectors to compare
        metric: Similarity metric ('cosine', 'euclidean', 'manhattan')

    Returns:
        Symmetric similarity matrix
    """
    n = len(vectors)
    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            if metric == 'cosine':
                sim = cosine_similarity(vectors[i], vectors[j])
            elif metric == 'euclidean':
                sim = euclidean_distance(vectors[i], vectors[j])
                # Convert distance to similarity (higher values = more similar)
                sim = 1.0 / (1.0 + sim)
            elif metric == 'manhattan':
                sim = manhattan_distance(vectors[i], vectors[j])
                sim = 1.0 / (1.0 + sim)
            else:
                raise ValueError(f"Unknown metric: {metric}")

            matrix[i, j] = sim
            matrix[j, i] = sim

    # Self-similarity is 1.0 (perfect similarity)
    np.fill_diagonal(matrix, 1.0)

    return matrix


class SimilarityIndex:
    """Efficient similarity search index for large vector sets.

    Uses clustering and approximate nearest neighbor techniques
    for fast similarity queries in routing operations.
    """

    def __init__(self, vectors: Optional[List[np.ndarray]] = None,
                 metric: str = 'cosine'):
        """Initialize similarity index.

        Args:
            vectors: Initial set of vectors to index
            metric: Similarity metric to use
        """
        self.vectors: List[np.ndarray] = []
        self.metric = metric
        self.similarity_matrix: Optional[np.ndarray] = None

        if vectors:
            self.add_vectors(vectors)

    def add_vectors(self, vectors: List[np.ndarray]) -> None:
        """Add vectors to the index.

        Args:
            vectors: List of vectors to add
        """
        self.vectors.extend(vectors)
        # Recompute similarity matrix (can be optimized for incremental updates)
        if len(self.vectors) > 0:
            self.similarity_matrix = compute_similarity_matrix(self.vectors, self.metric)

    def clear(self) -> None:
        """Clear all vectors from index."""
        self.vectors.clear()
        self.similarity_matrix = None

    def search(self, query: np.ndarray, top_k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Find most similar vectors to query.

        Args:
            query: Query vector
            top_k: Number of results to return

        Returns:
            Tuple of (similarities, indices)
        """
        if not self.vectors:
            return np.array([]), np.array([])

        # Compute similarities to all indexed vectors
        similarities = batch_cosine_similarity(query, self.vectors)

        # Get top-k results
        top_similarities, top_indices = find_most_similar(query, self.vectors, top_k)

        return top_similarities, top_indices

    def get_similar_to_index(self, index: int, top_k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Find vectors most similar to vector at given index.

        Args:
            index: Index of reference vector
            top_k: Number of similar vectors to return

        Returns:
            Tuple of (similarities, indices) excluding the reference vector
        """
        if self.similarity_matrix is None or index >= len(self.vectors):
            return np.array([]), np.array([])

        # Get similarity scores for this vector
        similarities = self.similarity_matrix[index].copy()

        # Don't return self-similarity
        similarities[index] = -np.inf

        # Get top-k similar vectors
        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_similarities = similarities[top_indices]

        return top_similarities, top_indices

    def cluster_vectors(self, n_clusters: int = 10) -> Dict[int, List[int]]:
        """Cluster vectors into similarity groups.

        Args:
            n_clusters: Number of clusters to create

        Returns:
            Dictionary mapping cluster_id to list of vector indices
        """
        if not self.vectors:
            return {}

        # Simple clustering based on similarity matrix
        clusters = {i: [] for i in range(n_clusters)}

        for i in range(len(self.vectors)):
            # Find most similar existing cluster
            best_cluster = None
            best_similarity = -1

            for cluster_id in range(n_clusters):
                if not clusters[cluster_id]:
                    # Empty cluster, can assign
                    best_cluster = cluster_id
                    break

                # Check similarity to cluster representative (first vector)
                rep_index = clusters[cluster_id][0]
                if self.similarity_matrix is not None:
                    similarity = self.similarity_matrix[i, rep_index]
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_cluster = cluster_id

            if best_cluster is not None:
                clusters[best_cluster].append(i)

        return clusters

    @property
    def size(self) -> int:
        """Number of vectors in index."""
        return len(self.vectors)


# Utility functions for routing applications

def route_similarity_score(source_embedding: np.ndarray,
                          destination_embedding: np.ndarray,
                          current_position: Tuple[int, int],
                          target_position: Tuple[int, int]) -> float:
    """Compute routing score combining embedding similarity and spatial distance.

    Args:
        source_embedding: Embedding of source shard
        destination_embedding: Embedding of destination region
        current_position: Current (x,y) position
        target_position: Target (x,y) position

    Returns:
        Combined routing score (higher = better route)
    """
    # Embedding similarity component (normalized to 0-1)
    embedding_sim = cosine_similarity(source_embedding, destination_embedding)
    embedding_score = (embedding_sim + 1.0) / 2.0  # Convert to 0-1 range

    # Spatial distance component (normalized, lower distance = higher score)
    spatial_distance = np.sqrt((current_position[0] - target_position[0])**2 +
                              (current_position[1] - target_position[1])**2)
    max_reasonable_distance = 100.0  # Configurable
    spatial_score = max(0, 1.0 - spatial_distance / max_reasonable_distance)

    # Combine scores with weighting
    combined_score = 0.7 * embedding_score + 0.3 * spatial_score

    return combined_score


def optimize_route_path(shard_embedding: np.ndarray,
                       start_pos: Tuple[int, int],
                       goal_pos: Tuple[int, int],
                       waypoints: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Optimize route path using embedding similarity.

    Args:
        shard_embedding: Embedding of shard being routed
        start_pos: Starting position
        goal_pos: Goal position
        waypoints: List of intermediate waypoints

    Returns:
        Optimized path as list of positions
    """
    if not waypoints:
        return [start_pos, goal_pos]

    # Create waypoint embeddings (simplified - in practice would use region embeddings)
    waypoint_embeddings = [shard_embedding] * len(waypoints)  # Placeholder

    # Find optimal ordering using embedding similarity
    path = [start_pos]
    remaining = list(zip(waypoints, waypoint_embeddings))

    current_pos = start_pos

    while remaining:
        # Find waypoint with highest routing score from current position
        best_score = -1
        best_index = -1

        for i, (waypoint, embedding) in enumerate(remaining):
            score = route_similarity_score(shard_embedding, embedding, current_pos, waypoint)
            if score > best_score:
                best_score = score
                best_index = i

        # Add best waypoint to path
        best_waypoint, _ = remaining.pop(best_index)
        path.append(best_waypoint)
        current_pos = best_waypoint

    path.append(goal_pos)
    return path


# Default similarity function for routing
default_similarity_metric = cosine_similarity

# Export common functions
__all__ = [
    'cosine_similarity',
    'cosine_distance',
    'euclidean_distance',
    'manhattan_distance',
    'batch_cosine_similarity',
    'find_most_similar',
    'compute_similarity_matrix',
    'SimilarityIndex',
    'route_similarity_score',
    'optimize_route_path',
    'default_similarity_metric'
]
