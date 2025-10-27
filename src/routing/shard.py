"""Vector shard data model for CyberMesh memory fragments.

Implements the core data structure representing memory shards that are
transported by gliders across the cellular automaton substrate. Each shard
contains an embedding vector, routing metadata, and state information.
"""

import uuid
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, Optional
import numpy as np
import hashlib
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class VectorShard:
    """Memory fragment with embedding vector and routing metadata.

    Represents a quantum of information that can be transported by glider
    patterns through the CA substrate. Shards are the fundamental unit
    of memory distribution in CyberMesh.
    """

    shard_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    embedding: np.ndarray = field(default_factory=lambda: np.array([]))
    destination: Tuple[int, int] = field(default=(0, 0))
    ttl: int = field(default=100)  # Time-to-live in simulation steps
    priority: float = field(default=1.0)  # Routing priority (0.0-1.0)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Runtime state (not serialized)
    position: Tuple[int, int] = field(default=(0, 0), init=False)
    attached_glider_id: Optional[str] = field(default=None, init=False)
    creation_time: datetime = field(default_factory=datetime.now, init=False)
    hop_count: int = field(default=0, init=False)
    last_movement: datetime = field(default_factory=datetime.now, init=False)

    def __post_init__(self):
        """Validate shard after construction."""
        self._validate()

        # Initialize runtime state
        if not self.position:
            self.position = self.destination  # Start at destination initially
        self.creation_time = datetime.now()
        self.last_movement = self.creation_time
        self.hop_count = 0

    def _validate(self):
        """Validate shard data integrity."""
        if not isinstance(self.shard_id, str) or len(self.shard_id.strip()) == 0:
            raise ValueError("shard_id must be non-empty string")

        if not isinstance(self.embedding, np.ndarray) or self.embedding.size == 0:
            raise ValueError("embedding must be non-empty numpy array")

        if not np.isfinite(self.embedding).all():
            raise ValueError("embedding contains invalid values (NaN or Inf)")

        # Normalize embedding vector (ensure unit length for cosine similarity)
        norm = np.linalg.norm(self.embedding)
        if norm > 0:
            self.embedding = self.embedding / norm

        destination_x, destination_y = self.destination
        if not (isinstance(destination_x, int) and isinstance(destination_y, int)):
            raise ValueError("destination must be tuple of integers")

        if not (0 <= self.ttl <= 10000):  # Reasonable TTL bounds
            raise ValueError("ttl must be between 0 and 10000")

        if not (0.0 <= self.priority <= 1.0):
            raise ValueError("priority must be between 0.0 and 1.0")

    @property
    def age(self) -> float:
        """Time elapsed since shard creation in seconds."""
        return (datetime.now() - self.creation_time).total_seconds()

    @property
    def distance_to_destination(self) -> float:
        """Euclidean distance to destination coordinates."""
        dx = self.position[0] - self.destination[0]
        dy = self.position[1] - self.destination[1]
        return np.sqrt(dx**2 + dy**2)

    @property
    def is_expired(self) -> bool:
        """Check if shard has exceeded its time-to-live."""
        return self.ttl <= 0

    @property
    def is_delivered(self) -> bool:
        """Check if shard has reached its destination."""
        return self.position == self.destination and not self.attached_glider_id

    @property
    def progress_ratio(self) -> float:
        """Progress toward destination (0.0 = at destination, higher = further)."""
        total_distance = np.sqrt(self.destination[0]**2 + self.destination[1]**2)
        if total_distance == 0:
            return 0.0  # Already at destination
        return self.distance_to_destination / total_distance

    def decrement_ttl(self) -> bool:
        """Decrement TTL by one step. Returns True if shard should be discarded."""
        self.ttl -= 1
        return self.is_expired

    def attach_to_glider(self, glider_id: str) -> None:
        """Attach shard to a glider for transport."""
        self.attached_glider_id = glider_id
        self.last_movement = datetime.now()
        logger.debug(f"Shard {self.shard_id} attached to glider {glider_id}")

    def detach_from_glider(self) -> None:
        """Detach shard from its glider."""
        if self.attached_glider_id:
            logger.debug(f"Shard {self.shard_id} detached from glider {self.attached_glider_id}")
        self.attached_glider_id = None

    def update_position(self, new_position: Tuple[int, int]) -> None:
        """Update shard position and increment hop counter."""
        old_position = self.position
        self.position = new_position
        self.last_movement = datetime.now()
        self.hop_count += 1

        logger.debug(f"Shard {self.shard_id} moved from {old_position} to {new_position} (hop {self.hop_count})")

    def get_routing_cost(self, candidate_position: Tuple[int, int]) -> float:
        """Calculate routing cost to move to candidate position.

        Cost combines distance to destination with priority weighting.
        Lower cost = better routing choice.
        """
        # Primary cost: distance to destination
        candidate_distance = np.sqrt(
            (candidate_position[0] - self.destination[0])**2 +
            (candidate_position[1] - self.destination[1])**2
        )

        # Secondary cost: penalize low priority
        priority_penalty = (1.0 - self.priority) * 10.0

        # Tertiary cost: discourage excessive hops (TTL awareness)
        ttl_penalty = max(0, (100 - self.ttl) * 0.1)

        return candidate_distance + priority_penalty + ttl_penalty

    def to_dict(self) -> Dict[str, Any]:
        """Serialize shard to dictionary for logging/storage."""
        return {
            'shard_id': self.shard_id,
            'embedding_shape': self.embedding.shape,
            'embedding_hash': hashlib.sha256(self.embedding.tobytes()).hexdigest()[:16],
            'destination': self.destination,
            'position': self.position,
            'ttl': self.ttl,
            'priority': self.priority,
            'hop_count': self.hop_count,
            'attached_glider_id': self.attached_glider_id,
            'creation_time': self.creation_time.isoformat(),
            'last_movement': self.last_movement.isoformat(),
            'distance_to_destination': self.distance_to_destination,
            'progress_ratio': self.progress_ratio,
            'metadata': self.metadata.copy()
        }

    @classmethod
    def from_text(cls, text: str, embedding_generator, destination: Tuple[int, int],
                  priority: float = 1.0, ttl: int = 100, **metadata) -> 'VectorShard':
        """Create shard from text content with generated embedding."""
        embedding = embedding_generator.generate(text)
        metadata.update({'source_text': text[:100] + '...' if len(text) > 100 else text})

        return cls(
            embedding=embedding,
            destination=destination,
            priority=priority,
            ttl=ttl,
            metadata=metadata
        )

    @classmethod
    def from_embedding(cls, embedding: np.ndarray, destination: Tuple[int, int],
                       priority: float = 1.0, ttl: int = 100, **metadata) -> 'VectorShard':
        """Create shard directly from embedding vector."""
        return cls(
            embedding=embedding.astype(np.float32),
            destination=destination,
            priority=priority,
            ttl=ttl,
            metadata=metadata
        )

    def __repr__(self) -> str:
        status = "DELIVERED" if self.is_delivered else ("ATTACHED" if self.attached_glider_id else "WAITING")
        return (f"VectorShard(id={self.shard_id[:6]}..., "
                f"pos={self.position}, dest={self.destination}, "
                f"ttl={self.ttl}, priority={self.priority:.2f}, "
                f"hops={self.hop_count}, status={status})")

    def __eq__(self, other: object) -> bool:
        """Equality based on shard_id only."""
        if not isinstance(other, VectorShard):
            return False
        return self.shard_id == other.shard_id

    def __hash__(self) -> int:
        """Hash based on shard_id for use in sets."""
        return hash(self.shard_id)


def create_test_shard(grid_size: int = 32, **kwargs) -> VectorShard:
    """Create a test shard with random embedding and destination."""
    # Random embedding vector (768 dims like nomic-embed)
    embedding = np.random.normal(0, 1, 768).astype(np.float32)
    embedding = embedding / np.linalg.norm(embedding)  # Normalize

    # Random destination within grid
    destination = (np.random.randint(0, grid_size), np.random.randint(0, grid_size))

    metadata = {'test_shard': True, 'creation_method': 'create_test_shard'}

    return VectorShard.from_embedding(embedding, destination, **kwargs, **metadata)
