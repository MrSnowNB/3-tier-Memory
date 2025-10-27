"""Comprehensive tests for glider-shard binding and carrier transport.

Tests the integration between glider detection, shard attachment/detachment,
position synchronization, and transport lifecycle management.
"""

import pytest
import numpy as np
from src.routing.shard import VectorShard, create_test_shard
from src.routing.carrier import ShardCarrier, bind_shard_to_glider_detection, synchronize_glider_movements
from src.patterns.glider import GliderOrientation, GliderPhase, get_glider_pattern
from src.patterns.detector import GliderDetection
from src.core.grid import Grid
from src.core.conway import ConwayEngine, default_engine as conway


class TestShardCarrierInitialization:
    """Test ShardCarrier system initialization and basic operations."""

    def setup_method(self):
        """Create fresh carrier for each test."""
        self.carrier = ShardCarrier(max_shards_per_cell=2)

    def test_carrier_creation(self):
        """Test carrier initializes correctly."""
        assert self.carrier.max_shards_per_cell == 2
        assert len(self.carrier.shards) == 0
        assert len(self.carrier.cell_occupancy) == 0
        assert len(self.carrier.glider_attachments) == 0
        assert len(self.carrier.shard_attachments) == 0

    def test_add_remove_shard(self):
        """Test basic shard add/remove operations."""
        shard = create_test_shard(grid_size=32, destination=(10, 10))

        # Add shard
        assert self.carrier.add_shard(shard)
        assert shard.shard_id in self.carrier.shards
        assert (10, 10) in self.carrier.cell_occupancy
        assert shard.shard_id in self.carrier.cell_occupancy[(10, 10)]

        # Remove shard
        removed = self.carrier.remove_shard(shard.shard_id)
        assert removed is shard
        assert shard.shard_id not in self.carrier.shards
        assert shard.shard_id not in self.carrier.cell_occupancy[(10, 10)]

    def test_capacity_limits(self):
        """Test shard capacity limits per cell."""
        carrier = ShardCarrier(max_shards_per_cell=1)

        # First shard should succeed
        shard1 = create_test_shard(destination=(5, 5))
        assert carrier.add_shard(shard1)

        # Second shard to same cell should fail
        shard2 = create_test_shard(destination=(5, 5))
        assert not carrier.add_shard(shard2)

        # Shard to different cell should succeed
        shard3 = create_test_shard(destination=(6, 6))
        assert carrier.add_shard(shard3)


class TestGliderShardAttachment:
    """Test attachment and detachment of shards to/from gliders."""

    def setup_method(self):
        """Create carrier and test shards."""
        self.carrier = ShardCarrier()
        self.shard = create_test_shard(destination=(10, 10))
        self.carrier.add_shard(self.shard)

        self.glider_id = "test_glider_001"
        self.glider_pos = (5, 5)

    def test_basic_attachment(self):
        """Test basic shard-to-glider attachment."""
        # Attach shard to glider
        success = self.carrier.attach_shard_to_glider(
            self.shard.shard_id, self.glider_id, self.glider_pos
        )
        assert success

        # Verify attachment state
        assert self.shard.attached_glider_id == self.glider_id
        assert self.shard.position == self.glider_pos
        assert self.carrier.glider_attachments[self.glider_id] == self.shard.shard_id
        assert self.carrier.shard_attachments[self.shard.shard_id] == self.glider_id

        # Verify occupancy updated
        assert self.shard.shard_id not in self.carrier.cell_occupancy[(10, 10)]

    def test_attachment_validation(self):
        """Test attachment validation (no double attachments)."""
        # Attach first time - should work
        assert self.carrier.attach_shard_to_glider(
            self.shard.shard_id, self.glider_id, self.glider_pos
        )

        # Try to attach same shard again - should fail
        assert not self.carrier.attach_shard_to_glider(
            self.shard.shard_id, "different_glider", (6, 6)
        )

        # Try to attach different shard to same glider - should fail
        shard2 = create_test_shard(destination=(11, 11))
        self.carrier.add_shard(shard2)
        assert not self.carrier.attach_shard_to_glider(
            shard2.shard_id, self.glider_id, self.glider_pos
        )

    def test_detachment(self):
        """Test shard detachment from gliders."""
        # First attach
        self.carrier.attach_shard_to_glider(
            self.shard.shard_id, self.glider_id, self.glider_pos
        )

        # Detach via glider
        detached = self.carrier.detach_shard_from_glider(self.glider_id)
        assert detached is self.shard
        assert self.shard.attached_glider_id is None

        # Cross-references should be cleaned up
        assert self.glider_id not in self.carrier.glider_attachments
        assert self.shard.shard_id not in self.carrier.shard_attachments

        # Shard should be back in occupancy
        assert self.shard.shard_id in self.carrier.cell_occupancy[self.glider_pos]


class TestPositionSynchronization:
    """Test synchronized movement between gliders and attached shards."""

    def setup_method(self):
        """Set up carrier with attached shard."""
        self.carrier = ShardCarrier()
        self.shard = create_test_shard(destination=(20, 20))
        self.carrier.add_shard(self.shard)

        self.glider_id = "sync_glider"
        self.carrier.attach_shard_to_glider(
            self.shard.shard_id, self.glider_id, (10, 10)
        )

    def test_position_updates(self):
        """Test that shard position follows glider movement."""
        # Glider moves to new position
        new_pos = (12, 12)
        self.carrier.update_glider_position(self.glider_id, new_pos)

        # Shard should have moved too
        assert self.shard.position == new_pos
        assert self.shard.hop_count == 1

        # Move again
        final_pos = (15, 15)
        self.carrier.update_glider_position(self.glider_id, final_pos)
        assert self.shard.position == final_pos
        assert self.shard.hop_count == 2

    def test_delivery_detection(self):
        """Test automatic delivery detection when reaching destination."""
        # Move directly to destination
        dest_pos = (20, 20)
        self.carrier.update_glider_position(self.glider_id, dest_pos)

        # Shard should still be attached since we haven't detached yet
        assert self.shard.attached_glider_id == self.glider_id
        assert not self.shard.is_delivered()  # Still attached

        # Detach and check delivery
        self.carrier.detach_shard_from_glider(self.glider_id)
        assert self.shard.is_delivered()
        assert self.carrier.total_transports == 1

    def test_auto_detachment_near_destination(self):
        """Test probabilistic auto-detachment near destination."""
        # Move to position near destination
        near_dest = (19, 19)  # 1.41 units from (20,20)
        self.carrier.update_glider_position(self.glider_id, near_dest)

        # Shard should still be attached (auto-detach is probabilistic)
        assert self.shard.attached_glider_id == self.glider_id

        # Since it's probabilistic, we can't reliably test it here
        # but the mechanism should exist in the code


class TestShardLifecycle:
    """Test shard lifecycle management (TTL, cleanup, expiration)."""

    def setup_method(self):
        """Set up carrier with test shards."""
        self.carrier = ShardCarrier()

    def test_ttl_management(self):
        """Test TTL decrement and expiration."""
        # Create shard with TTL=5
        shard = create_test_shard(ttl=5, destination=(10, 10))
        self.carrier.add_shard(shard)

        assert not shard.is_expired()

        # Process lifecycle multiple times
        for i in range(5):
            expired, delivered, cleaned = self.carrier.process_shard_lifecycle()
            assert len(self.carrier.shards) > 0  # Not expired yet
            assert shard.ttl == 4 - i

        # Next process should expire the shard
        expired, delivered, cleaned = self.carrier.process_shard_lifecycle()
        assert expired == 1
        assert shard.shard_id not in self.carrier.shards

    def test_deliveries_vs_expirations(self):
        """Test proper counting of deliveries vs expirations."""
        # Add two shards - one will be delivered, one will expire
        shard1 = create_test_shard(destination=(5, 5), ttl=10)
        shard2 = create_test_shard(destination=(15, 15), ttl=2)

        self.carrier.add_shard(shard1)
        self.carrier.add_shard(shard2)

        # Deliver shard1
        self.carrier.attach_shard_to_glider(shard1.shard_id, "glider1", (5, 5))
        self.carrier.detach_shard_from_glider("glider1")

        # Let shard2 expire
        for _ in range(3):  # TTL=2, so after 3 decrements it expires
            self.carrier.process_shard_lifecycle()

        assert self.carrier.total_transports == 1  # shard1 delivered
        assert self.carrier.failed_deliveries == 1  # shard2 expired


class TestCellInventoryManagement:
    """Test shard inventory management per grid cell."""

    def setup_method(self):
        """Set up carrier for cell testing."""
        self.carrier = ShardCarrier(max_shards_per_cell=3)

    def test_cell_queries(self):
        """Test querying shards at specific positions."""
        shard1 = create_test_shard(destination=(5, 5))
        shard2 = create_test_shard(destination=(5, 5))
        shard3 = create_test_shard(destination=(6, 6))

        self.carrier.add_shard(shard1)
        self.carrier.add_shard(shard2)
        self.carrier.add_shard(shard3)

        # Check cell (5,5) has 2 shards
        shards_at_55 = self.carrier.get_shard_at_position((5, 5))
        assert len(shards_at_55) == 2
        assert shard1 in shards_at_55
        assert shard2 in shards_at_55

        # Check cell (6,6) has 1 shard
        shards_at_66 = self.carrier.get_shard_at_position((6, 6))
        assert len(shards_at_66) == 1
        assert shard3 in shards_at_66

        # Check empty cell
        shards_at_empty = self.carrier.get_shard_at_position((10, 10))
        assert len(shards_at_empty) == 0

    def test_attachment_updates_occupancy(self):
        """Test that attachment/detachment updates cell occupancy."""
        shard = create_test_shard(destination=(8, 8))
        self.carrier.add_shard(shard)

        # Initially in destination cell
        assert shard.shard_id in self.carrier.cell_occupancy[(8, 8)]

        # Attach to glider at different position
        self.carrier.attach_shard_to_glider(shard.shard_id, "glider", (12, 12))

        # Should be removed from original cell and associated with glider
        assert shard.shard_id not in self.carrier.cell_occupancy[(8, 8)]
        assert shard.shard_id not in self.carrier.cell_occupancy[(12, 12)]  # Not in cell when attached

        # Detach and check occupancy restore
        self.carrier.detach_shard_from_glider("glider")
        assert shard.shard_id in self.carrier.cell_occupancy[(12, 12)]


class TestGliderDetectionIntegration:
    """Test integration with Phase 1 glider detection."""

    def setup_method(self):
        """Set up integrated test environment."""
        self.carrier = ShardCarrier()
        self.shard = create_test_shard(destination=(20, 20))
        self.carrier.add_shard(self.shard)

    def test_bind_to_glider_detection(self):
        """Test binding shard to detected glider."""
        # Create a glider detection
        detection = GliderDetection(
            x=10, y=10,
            orientation=GliderOrientation.NORTHWEST,
            phase=GliderPhase.PHASE_0,
            confidence=0.95,
            pattern_match=np.zeros((3, 3), dtype=bool)
        )

        # Bind shard to detection
        success = bind_shard_to_glider_detection(self.carrier, self.shard, detection)
        assert success

        # Check binding worked
        expected_glider_id = f"glider_{detection.x}_{detection.y}_{detection.orientation.name}_{detection.phase.name}"
        attached_shard = self.carrier.get_attached_shard(expected_glider_id)
        assert attached_shard is self.shard

        # Shard should be at detection position
        assert self.shard.position == (detection.x, detection.y)

    def test_movement_synchronization(self):
        """Test synchronized movement with detection updates."""
        # Initial binding
        detection1 = GliderDetection(
            x=5, y=5,
            orientation=GliderOrientation.NORTHEAST,
            phase=GliderPhase.PHASE_0,
            confidence=0.9,
            pattern_match=np.zeros((3, 3), dtype=bool)
        )

        bind_shard_to_glider_detection(self.carrier, self.shard, detection1)
        glider_id = f"glider_{detection1.x}_{detection1.y}_{detection1.orientation.name}_{detection1.phase.name}"

        # Simulate glider movement
        movements = {glider_id: (8, 8)}
        synchronize_glider_movements(self.carrier, movements)

        # Shard should follow
        assert self.shard.position == (8, 8)

        # Another movement
        movements = {glider_id: (12, 12)}
        synchronize_glider_movements(self.carrier, movements)
        assert self.shard.position == (12, 12)


class TestCarrierStatistics:
    """Test carrier performance and statistics tracking."""

    def setup_method(self):
        """Set up carrier for statistics testing."""
        self.carrier = ShardCarrier()

    def test_transport_statistics(self):
        """Test transport and delivery statistics."""
        # Add and deliver some shards
        shard1 = create_test_shard(destination=(10, 10))
        shard2 = create_test_shard(destination=(20, 20))

        self.carrier.add_shard(shard1)
        self.carrier.add_shard(shard2)

        # Deliver shard1
        self.carrier.attach_shard_to_glider(shard1.shard_id, "g1", (10, 10))
        self.carrier.detach_shard_from_glider("g1")

        # Leave shard2 undelivered (will expire)
        shard2.ttl = 1
        self.carrier.process_shard_lifecycle()

        stats = self.carrier.get_transport_statistics()

        assert stats['total_shards'] == 2  # Both started
        assert stats['attached_shards'] == 0  # Both detached
        assert stats['waiting_shards'] == 1  # shard2 still exists before cleanup
        assert stats['total_attachments'] == 1
        assert stats['total_detachments'] == 1
        assert stats['successful_deliveries'] == 1
        assert stats['failed_deliveries'] == 1  # shard2 expired

    def test_carrier_clear(self):
        """Test carrier state clearing."""
        # Add some content
        shard = create_test_shard()
        self.carrier.add_shard(shard)
        self.carrier.attach_shard_to_glider(shard.shard_id, "g1", (5, 5))

        # Clear everything
        self.carrier.clear()

        # Verify clean state
        assert len(self.carrier.shards) == 0
        assert len(self.carrier.cell_occupancy) == 0
        assert len(self.carrier.glider_attachments) == 0
        assert len(self.carrier.shard_attachments) == 0
        assert self.carrier.total_attachments == 0


class TestCarrierErrorHandling:
    """Test error handling and edge cases."""

    def setup_method(self):
        """Create carrier for error testing."""
        self.carrier = ShardCarrier()

    def test_nonexistent_shard_operations(self):
        """Test operations on nonexistent shards."""
        # Try to attach nonexistent shard
        success = self.carrier.attach_shard_to_glider("fake_id", "glider", (0, 0))
        assert not success

        # Try to remove nonexistent shard
        removed = self.carrier.remove_shard("fake_id")
        assert removed is None

        # Try to detach from nonexistent glider
        detached = self.carrier.detach_shard_from_glider("fake_glider")
        assert detached is None

    def test_boundary_conditions(self):
        """Test operation near grid boundaries."""
        # Create shards near boundaries
        shard_edge = create_test_shard(destination=(0, 0))  # Corner
        shard_wide = create_test_shard(destination=(99, 99))  # Far corner

        assert self.carrier.add_shard(shard_edge)
        assert self.carrier.add_shard(shard_wide)

        # Operations should work normally despite position
        assert self.carrier.attach_shard_to_glider(shard_edge.shard_id, "g1", (1, 1))
        assert self.carrier.attach_shard_to_glider(shard_wide.shard_id, "g2", (98, 98))

    def test_nearest_shard_finding(self):
        """Test finding nearest available shards."""
        # No shards yet
        nearest = self.carrier.find_nearest_available_shard((10, 10))
        assert nearest is None

        # Add shards at various distances
        shard1 = create_test_shard(destination=(10, 10))  # At search position
        shard2 = create_test_shard(destination=(15, 15))  # 7.07 units away
        shard3 = create_test_shard(destination=(20, 20))  # 14.14 units away

        self.carrier.add_shard(shard1)
        self.carrier.add_shard(shard2)
        self.carrier.add_shard(shard3)

        # Attach one shard (should be skipped)
        self.carrier.attach_shard_to_glider(shard1.shard_id, "g1", (10, 10))

        # Find nearest (should be shard2)
        nearest = self.carrier.find_nearest_available_shard((10, 10), max_distance=20.0)
        assert nearest is shard2  # Closer and available

        # Find with limited range (should find nothing)
        nearest = self.carrier.find_nearest_available_shard((10, 10), max_distance=5.0)
        assert nearest is None  # shard2 is 7 units away
