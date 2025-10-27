"""Unit tests for VectorShard data model and validation.

Tests shard creation, validation, state management, and core functionality.
Ensures integrity of vector shards for routing operations.
"""

import pytest
import numpy as np
from datetime import datetime
from src.routing.shard import VectorShard, create_test_shard


class TestShardCreation:
    """Test shard creation and initialization."""

    def test_create_from_text_stub(self):
        """Test shard creation from text (stub implementation)."""
        # Create test embedding
        embedding = np.random.normal(0, 1, 768).astype(np.float32)

        # Mock generator function
        def mock_generator(text: str) -> np.ndarray:
            return embedding

        destination = (10, 20)
        shard = VectorShard.from_text("test content", mock_generator, destination)

        assert shard.destination == destination
        assert shard.metadata['source_text'] == "test content"
        assert shard.shard_id is not None
        assert len(shard.shard_id) == 8  # UUID prefix

    def test_create_from_embedding(self):
        """Test shard creation directly from embedding vector."""
        embedding = np.random.normal(0, 1, 768).astype(np.float32)
        destination = (15, 25)
        priority = 0.8
        ttl = 50

        shard = VectorShard.from_embedding(
            embedding, destination, priority=priority, ttl=ttl
        )

        assert shard.destination == destination
        assert shard.priority == priority
        assert shard.ttl == ttl
        assert shard.position == destination  # Should start at destination
        assert not shard.is_delivered()  # Not yet attached to glider

    def test_invalid_embedding(self):
        """Test validation of malformed embeddings."""
        # Empty embedding
        with pytest.raises(ValueError, match="must be non-empty"):
            VectorShard(embedding=np.array([]), destination=(0, 0))

        # Non-finite values
        bad_embedding = np.array([1.0, np.nan, 1.0])
        with pytest.raises(ValueError, match="contains invalid values"):
            VectorShard(embedding=bad_embedding, destination=(0, 0))

        # Wrong data type
        wrong_type = np.array([1, 2, 3], dtype=int)
        shard = VectorShard(embedding=wrong_type.astype(float), destination=(0, 0))
        # Should be converted and normalized
        assert shard.embedding.dtype == np.float32

    def test_invalid_destination(self):
        """Test validation of destination coordinates."""
        embedding = np.random.normal(0, 1, 768).astype(np.float32)

        # Non-integer coordinates
        with pytest.raises(ValueError, match="must be tuple of integers"):
            VectorShard(embedding=embedding, destination=(1.5, 2.0))

    def test_invalid_ttl_priority(self):
        """Test validation of TTL and priority ranges."""
        embedding = np.random.normal(0, 1, 768).astype(np.float32)

        # Invalid TTL
        with pytest.raises(ValueError, match="ttl must be between"):
            VectorShard(embedding=embedding, destination=(0, 0), ttl=-5)

        with pytest.raises(ValueError, match="ttl must be between"):
            VectorShard(embedding=embedding, destination=(0, 0), ttl=15000)

        # Invalid priority
        with pytest.raises(ValueError, match="priority must be between"):
            VectorShard(embedding=embedding, destination=(0, 0), priority=2.0)

        with pytest.raises(ValueError, match="priority must be between"):
            VectorShard(embedding=embedding, destination=(0, 0), priority=-1.0)

    def test_embedding_normalization(self):
        """Test that embeddings are automatically normalized."""
        # Non-normalized embedding
        raw_embedding = np.array([3.0, 4.0, 0.0, 0.0])  # Length = 5.0
        shard = VectorShard(embedding=raw_embedding, destination=(0, 0))

        # Should be normalized to unit length
        norm = np.linalg.norm(shard.embedding)
        assert abs(norm - 1.0) < 1e-6

        # Original direction should be preserved
        expected_normalized = raw_embedding / 5.0
        np.testing.assert_array_almost_equal(shard.embedding, expected_normalized)


class TestShardStateManagement:
    """Test shard state properties and lifecycle."""

    def setup_method(self):
        """Create test shard for each test."""
        embedding = np.random.normal(0, 1, 768).astype(np.float32)
        self.shard = VectorShard.from_embedding(embedding, (100, 200))

    def test_initial_state(self):
        """Test shard initial state."""
        assert self.shard.position == (100, 200)  # Starts at destination
        assert self.shard.ttl == 100
        assert self.shard.priority == 1.0
        assert self.shard.hop_count == 0
        assert self.shard.attached_glider_id is None
        assert not self.shard.is_delivered()
        assert not self.shard.is_expired()

    def test_distance_calculations(self):
        """Test distance and progress calculations."""
        # Distance to destination (same position)
        assert self.shard.distance_to_destination == 0.0
        assert self.shard.progress_ratio == 0.0  # At destination = 0 progress needed

        # Move away from destination
        self.shard.update_position((90, 190))  # 10 units away

        distance = self.shard.distance_to_destination
        assert distance == 10.0

        # Progress ratio should be distance / total_possible_distance
        total_distance = np.sqrt(100**2 + 200**2)  # Distance from (0,0) to destination
        expected_progress = distance / total_distance
        assert abs(self.shard.progress_ratio - expected_progress) < 0.01

    def test_ttl_management(self):
        """Test TTL decrement and expiration."""
        assert not self.shard.is_expired()

        # Decrement TTL
        expired = self.shard.decrement_ttl()
        assert not expired
        assert self.shard.ttl == 99

        # Decrement to zero
        self.shard.ttl = 1
        expired = self.shard.decrement_ttl()
        assert expired
        assert self.shard.is_expired()

    def test_attachment_lifecycle(self):
        """Test glider attachment and detachment."""
        glider_id = "glider_123"

        # Attach
        self.shard.attach_to_glider(glider_id)
        assert self.shard.attached_glider_id == glider_id
        assert self.shard.is_delivered() is False  # Not at destination while attached

        # Detach
        self.shard.detach_from_glider()
        assert self.shard.attached_glider_id is None

    def test_position_updates(self):
        """Test position updates and hop counting."""
        assert self.shard.hop_count == 0

        # Move to new position
        self.shard.update_position((50, 100))
        assert self.shard.position == (50, 100)
        assert self.shard.hop_count == 1

        # Another move
        self.shard.update_position((75, 150))
        assert self.shard.position == (75, 150)
        assert self.shard.hop_count == 2

    def test_delivery_conditions(self):
        """Test delivery status detection."""
        # Start at destination without glider attachment = delivered
        self.shard.position = self.shard.destination
        self.shard.attached_glider_id = None
        assert self.shard.is_delivered()

        # At destination but still attached = not delivered
        self.shard.attached_glider_id = "glider_abc"
        assert not self.shard.is_delivered()

        # Away from destination = not delivered
        self.shard.update_position((50, 50))
        assert not self.shard.is_delivered()


class TestShardRouting:
    """Test routing cost calculations."""

    def setup_method(self):
        """Create test shard."""
        embedding = np.random.normal(0, 1, 768).astype(np.float32)
        self.shard = VectorShard.from_embedding(embedding, (100, 100))
        self.shard.update_position((50, 50))  # Start away from destination

    def test_routing_cost(self):
        """Test routing cost calculation."""
        # Cost to destination should be low (direct path)
        dest_cost = self.shard.get_routing_cost((100, 100))
        assert dest_cost < 50  # Should be distance-based: ~50 units

        # Cost to further position should be higher
        far_cost = self.shard.get_routing_cost((200, 200))
        assert far_cost > dest_cost

        # Test priority effects (lower priority = higher cost)
        low_priority_shard = VectorShard.from_embedding(
            self.shard.embedding.copy(), (100, 100), priority=0.1
        )
        low_priority_shard.update_position((50, 50))

        low_cost = low_priority_shard.get_routing_cost((100, 100))
        assert low_cost > dest_cost  # Should be higher due to low priority

    def test_routing_cost_edge_cases(self):
        """Test routing cost edge cases."""
        # Same position (cost should still include priority penalty)
        same_cost = self.shard.get_routing_cost((50, 50))
        assert same_cost >= 0  # Non-negative

        # Very distant position
        distant_cost = self.shard.get_routing_cost((1000, 1000))
        assert distant_cost > 100


class TestShardSerialization:
    """Test shard serialization for logging and persistence."""

    def setup_method(self):
        """Create test shard."""
        embedding = np.random.normal(0, 1, 768).astype(np.float32)
        self.shard = VectorShard.from_embedding(embedding, (123, 456))
        self.shard.metadata = {
            'source': 'test',
            'importance': 'high',
            'tags': ['experimental', 'demo']
        }

    def test_to_dict_serialization(self):
        """Test serialization to dictionary."""
        data = self.shard.to_dict()

        # Check core fields
        assert data['shard_id'] == self.shard.shard_id
        assert data['destination'] == (123, 456)
        assert data['ttl'] == 100
        assert data['priority'] == 1.0
        assert data['hop_count'] == 0
        assert data['progress_ratio'] == 0.0

        # Check embedding hash (not full vector for privacy/performance)
        assert 'embedding_hash' in data
        assert len(data['embedding_hash']) == 16  # 32 hex chars / 2

        # Check timestamps
        assert 'creation_time' in data
        assert 'last_movement' in data

        # Check metadata preservation
        assert data['metadata']['source'] == 'test'
        assert data['metadata']['importance'] == 'high'

    def test_to_dict_after_operations(self):
        """Test serialization after various operations."""
        # Perform some operations
        self.shard.update_position((100, 100))
        self.shard.attach_to_glider("test_glider")
        self.shard.decrement_ttl()

        data = self.shard.to_dict()

        assert data['position'] == (100, 100)
        assert data['attached_glider_id'] == "test_glider"
        assert data['ttl'] == 99
        assert data['hop_count'] == 1


class TestUtilityFunctions:
    """Test utility functions for shard creation."""

    def test_create_test_shard(self):
        """Test test shard creation function."""
        shard = create_test_shard(grid_size=64)

        # Should have random but valid properties
        assert len(shard.shard_id) == 8
        assert shard.embedding.shape == (768,)
        assert np.isfinite(shard.embedding).all()
        assert 0 <= shard.destination[0] < 64
        assert 0 <= shard.destination[1] < 64
        assert shard.ttl == 100
        assert shard.priority == 1.0
        assert shard.metadata['test_shard'] is True

    def test_create_test_shard_custom_params(self):
        """Test test shard with custom parameters."""
        shard = create_test_shard(
            grid_size=128,
            ttl=200,
            priority=0.5,
            destination=(50, 50)  # Should override random
        )

        assert shard.ttl == 200
        assert shard.priority == 0.5
        assert shard.destination == (50, 50)  # Custom wins over random
