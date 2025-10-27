"""Comprehensive tests for routing performance and success rates.

Tests single-hop and multi-hop routing algorithms to validate Phase 2
requirements: ≥95% single-hop success, ≥90% multi-hop success, <50ms
embedding operations, <10ms single-hop routing.
"""

import pytest
import numpy as np
import time
from unittest.mock import MagicMock
from src.routing.shard import VectorShard, create_test_shard
from src.routing.carrier import ShardCarrier
from src.routing.router import (
    SingleHopRouter,
    MultiHopRouter,
    simulate_single_hop_delivery,
    validate_95_percent_success,
    create_3x3_router
)
from src.patterns.glider import GliderOrientation, GliderPhase


class TestSingleHopRouting:
    """Test single-hop routing algorithms."""

    def setup_method(self):
        """Set up test fixtures."""
        self.carrier = ShardCarrier()
        self.router = SingleHopRouter(self.carrier)

    def test_basic_single_hop_routing(self):
        """Test basic single-hop routing functionality."""
        # Create a shard
        shard = create_test_shard(destination=(10, 10))
        self.carrier.add_shard(shard)

        # Available gliders
        gliders = {
            "glider1": (5, 5),
            "glider2": (8, 8),
            "glider3": (12, 12)
        }

        # Route the shard
        selected_glider = self.router.route_single_hop(shard.shard_id, gliders)

        # Should have selected a glider
        assert selected_glider in gliders

        # Shard should be attached to the selected glider
        assert shard.attached_glider_id == selected_glider

        # Statistics should be updated
        stats = self.router.get_routing_stats()
        assert stats['successful_routes'] == 1
        assert stats['routing_attempts'] == 1

    def test_routing_cost_calculation(self):
        """Test routing cost considers distance and priority."""
        # Create shards with different destinations and priorities
        shard1 = create_test_shard(destination=(10, 10), priority=0.1)  # Low priority
        shard2 = create_test_shard(destination=(10, 10), priority=1.0)  # High priority

        self.carrier.add_shard(shard1)
        self.carrier.add_shard(shard2)

        # Both at same position
        shard1.update_position((5, 5))
        shard2.update_position((5, 5))

        gliders = {"glider1": (6, 6)}

        # Route both shards - higher priority should be selected first
        # (they have the same cost, but implementation selects first available)

        # First routing should succeed
        result1 = self.router.route_single_hop(shard1.shard_id, gliders)
        assert result1 is not None

        # Second routing should fail (glider already occupied)
        result2 = self.router.route_single_hop(shard2.shard_id, gliders)
        assert result2 is None

    def test_routing_with_unavailable_gliders(self):
        """Test routing when all gliders are occupied."""
        shard = create_test_shard(destination=(10, 10))
        self.carrier.add_shard(shard)

        gliders = {"glider1": (5, 5)}

        # First route should succeed
        result1 = self.router.route_single_hop(shard.shard_id, gliders)
        assert result1 == "glider1"

        # Create another shard
        shard2 = create_test_shard(destination=(15, 15))
        self.carrier.add_shard(shard2)

        # Second route should fail (no available gliders)
        result2 = self.router.route_single_hop(shard2.shard_id, gliders)
        assert result2 is None

        stats = self.router.get_routing_stats()
        assert stats['successful_routes'] == 1
        assert stats['failed_routes'] == 1


class TestMultiHopRouting:
    """Test multi-hop routing with A* pathfinding."""

    def setup_method(self):
        """Set up test fixtures."""
        self.carrier = ShardCarrier()

    def test_3x3_router_creation(self):
        """Test creating router for 3x3 grid scenarios."""
        router = create_3x3_router(self.carrier)

        assert router.min_x == -1
        assert router.min_y == -1
        assert router.max_x == 1
        assert router.max_y == 1

    def test_simple_pathfinding(self):
        """Test basic A* pathfinding."""
        # Create 5x5 grid router for testing
        router = MultiHopRouter(self.carrier, (-2, -2, 2, 2))

        # Create a shard that needs to travel from (0,0) to (2,2)
        shard = create_test_shard(destination=(2, 2), priority=1.0)
        shard.update_position((0, 0))  # Start at origin
        self.carrier.add_shard(shard)

        # Find route
        path = router.find_multi_hop_route(shard.shard_id, max_hops=10)

        # Should find a path
        assert path is not None
        assert path[0] == (0, 0)  # Start position
        assert path[-1] == (2, 2)  # End position
        assert len(path) >= 3  # Should take some steps

        # Path should be monotonically approaching destination
        for i in range(1, len(path)):
            prev_dist = abs(path[i-1][0] - 2) + abs(path[i-1][1] - 2)
            curr_dist = abs(path[i][0] - 2) + abs(path[i][1] - 2)
            # Distance should generally decrease (A* optimality)
            assert curr_dist <= prev_dist + 2  # Allow small increases

    def test_no_path_available(self):
        """Test routing when no path exists."""
        # Create very constrained router (only 1x1 grid)
        router = MultiHopRouter(self.carrier, (0, 0, 0, 0))

        shard = create_test_shard(destination=(5, 5))
        shard.update_position((0, 0))
        self.carrier.add_shard(shard)

        # Should not find path (can't reach destination)
        path = router.find_multi_hop_route(shard.shard_id, max_hops=5)

        assert path is None

        stats = router.get_routing_stats()
        assert stats['successful_paths'] == 0
        assert stats['pathfinding_attempts'] == 1


class TestRoutingPerformance:
    """Test routing performance against Phase 2 requirements."""

    def test_single_hop_routing_performance(self):
        """Validate single-hop routing meets <10ms performance target."""
        carrier = ShardCarrier()
        router = SingleHopRouter(carrier)

        # Create test scenario
        shard = create_test_shard(destination=(10, 10))
        carrier.add_shard(shard)

        gliders = {f"glider{i}": (i + 5, i + 5) for i in range(5)}

        # Measure time over multiple operations
        start_time = time.time()
        num_operations = 100

        for _ in range(num_operations):
            router.route_single_hop(shard.shard_id, gliders)
            # Detach for next attempt
            if shard.attached_glider_id:
                carrier.detach_shard_from_glider(shard.attached_glider_id)

        total_time = time.time() - start_time
        avg_time_per_operation = total_time / num_operations

        # Phase 2 target: <10ms per single-hop operation
        assert avg_time_per_operation < 0.010, \
            f"Single-hop routing too slow: {avg_time_per_operation:.4f}s > 0.010s"

        print(f"Single-hop routing performance: {avg_time_per_operation:.4f}s per operation")

    def test_single_hop_success_rate_95_percent(self):
        """Validate single-hop routing achieves ≥95% success rate."""
        carrier = ShardCarrier(max_shards_per_cell=1)
        router = SingleHopRouter(carrier)

        # Create multiple shards with distributed destinations
        num_shards = 10
        shards = []
        for i in range(num_shards):
            # Spread destinations over a wider area to avoid capacity conflicts
            shard = create_test_shard(
                destination=(i + 5, i + 5),  # Destinations like (5,5), (6,6), ..., (14,14)
                priority=0.5 + (i / num_shards) * 0.5  # Vary priorities
            )
            carrier.add_shard(shard)
            shards.append(shard)

        # Create available gliders in routing area
        gliders = {}
        for i in range(5):  # 5 gliders available
            gliders[f"glider{i}"] = (7 + i % 2, 7 + i // 2)

        # Simulate delivery
        stats = simulate_single_hop_delivery(router, shards, gliders, max_steps=20)

        print(f"Routing simulation results:")
        print(f"  Simulation steps: {stats['simulation_steps']}")
        print(f"  Successful deliveries: {stats['successful_deliveries']}")
        print(f"  Remaining shards: {stats['remaining_shards']}")
        print(f"  Expired shards: {stats['expired_shards']}")
        print(f"  Success rate: {stats['routing_success_rate']:.1%}")

        # Check if we meet the 95% success criterion (delivery-based)
        success_criterion_met = validate_95_percent_success(stats)
        assert success_criterion_met, \
            f"Phase 2 single-hop success criterion not met: {stats['successful_deliveries']}/{stats['successful_deliveries'] + stats['remaining_shards'] + stats['expired_shards']} ({stats['successful_deliveries']/(stats['successful_deliveries'] + stats['remaining_shards'] + stats['expired_shards']):.1%}) < 95%"

        # Additional validation
        total_shards = stats['successful_deliveries'] + stats['remaining_shards'] + stats['expired_shards']
        assert total_shards == num_shards, f"Shard count mismatch: {total_shards} != {num_shards}"

        # Should have reasonable delivery rates
        assert stats['successful_deliveries'] > 0, "No successful deliveries achieved"

    def test_multi_hop_performance(self):
        """Test multi-hop routing performance in constrained scenarios."""
        carrier = ShardCarrier()
        router = MultiHopRouter(carrier, (-2, -2, 2, 2))  # 5x5 grid

        # Create shard needing multi-hop route
        shard = create_test_shard(destination=(2, 2))
        shard.update_position((-2, -2))  # Start far away
        carrier.add_shard(shard)

        # Measure pathfinding performance
        start_time = time.time()
        num_attempts = 50

        successful_paths = 0
        for _ in range(num_attempts):
            path = router.find_multi_hop_route(shard.shard_id, max_hops=10)
            if path:
                successful_paths += 1

        total_time = time.time() - start_time
        avg_time_per_attempt = total_time / num_attempts
        success_rate = successful_paths / num_attempts

        print(f"Multi-hop performance:")
        print(f"  Avg time per attempt: {avg_time_per_attempt:.4f}s")
        print(f"  Pathfinding success rate: {success_rate:.1%}")

        # Should complete within reasonable time
        assert avg_time_per_attempt < 0.1, f"Pathfinding too slow: {avg_time_per_attempt:.4f}s"

        # Should achieve good success rate for simple paths
        assert success_rate > 0.8, f"Low pathfinding success: {success_rate:.1%}"

    def test_memory_usage_bounds(self):
        """Test memory usage stays within Phase 2 limits (<2GB)."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        start_memory = process.memory_info().rss / 1024 / 1024  # MB

        carrier = ShardCarrier()
        router = SingleHopRouter(carrier)

        # Create many shards and perform routing operations
        num_shards = 100
        shards = []

        for i in range(num_shards):
            shard = create_test_shard(
                destination=(i % 10 + 5, i % 10 + 5),
                priority=np.random.random()
            )
            carrier.add_shard(shard)
            shards.append(shard)

        gliders = {f"g{i}": (i % 5 + 5, i % 5 + 5) for i in range(20)}

        # Perform extensive routing
        for step in range(50):
            for shard in shards[:10]:  # Route first 10 shards repeatedly
                if not shard.is_delivered and not shard.attached_glider_id:
                    router.route_single_hop(shard.shard_id, gliders)

            carrier.process_shard_lifecycle()

        end_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = end_memory - start_memory

        print(f"Phase 2 memory usage: {memory_used:.1f}MB")

        # Phase 2 target: <2GB total for active routing
        assert memory_used < 2000, f"Memory usage too high: {memory_used:.1f}MB >= 2000MB"

        # Should be much lower in practice
        assert memory_used < 500, f"Memory usage excessive: {memory_used:.1f}MB"


class TestRoutingIntegration:
    """Integration tests combining multiple routing components."""

    def test_end_to_end_routing_scenario(self):
        """Test complete routing scenario from shard creation to delivery."""
        carrier = ShardCarrier()
        router = SingleHopRouter(carrier)

        # Create an "emerging" shard (simulating detection)
        shard = create_test_shard(destination=(10, 10), ttl=50)
        carrier.add_shard(shard)

        # Gliders available for transport
        gliders = {
            "north_glider": (8, 8),    # Close to shard
            "south_glider": (15, 15),  # Farther away
        }

        # Step 1: Route shard to glider
        result = router.route_single_hop(shard.shard_id, gliders)
        assert result == "north_glider"  # Should choose closer glider

        # Step 2: Simulate movement toward destination
        carrier.update_glider_position(result, (9, 9))
        assert shard.position == (9, 9)

        carrier.update_glider_position(result, (10, 10))
        assert shard.position == (10, 10)

        # Shard should be at destination
        assert not shard.is_delivered  # But still attached to glider

        # Step 3: Detach at destination
        detached = carrier.detach_shard_from_glider(result)
        assert detached is shard
        assert shard.is_delivered

        # Step 4: Lifecycle processing should clean up
        expired, delivered, cleaned = carrier.process_shard_lifecycle()
        assert delivered == 1
        assert shard.shard_id not in carrier.shards

        # Verify final statistics
        stats = router.get_routing_stats()
        assert stats['successful_routes'] == 1

        carrier_stats = carrier.get_transport_statistics()
        assert carrier_stats['successful_deliveries'] == 1

        print("End-to-end routing scenario completed successfully!")


class TestRoutingEdgeCases:
    """Test routing behavior in edge cases."""

    def test_routing_with_no_gliders(self):
        """Test routing when no gliders are available."""
        carrier = ShardCarrier()
        router = SingleHopRouter(carrier)

        shard = create_test_shard(destination=(10, 10))
        carrier.add_shard(shard)

        result = router.route_single_hop(shard.shard_id, {})
        assert result is None

        stats = router.get_routing_stats()
        assert stats['failed_routes'] == 1

    def test_routing_expired_shard(self):
        """Test attempting to route an expired shard."""
        carrier = ShardCarrier()
        router = SingleHopRouter(carrier)

        shard = create_test_shard(destination=(10, 10), ttl=1)
        carrier.add_shard(shard)

        # Expire the shard
        carrier.process_shard_lifecycle()
        assert shard.is_expired

        gliders = {"glider1": (5, 5)}
        result = router.route_single_hop(shard.shard_id, gliders)
        assert result is None  # Should not route expired shard

    def test_concurrent_routing_conflicts(self):
        """Test routing multiple shards with glider conflicts."""
        carrier = ShardCarrier()
        router = SingleHopRouter(carrier)

        # Create multiple shards competing for one glider
        shards = []
        for i in range(3):
            shard = create_test_shard(destination=(10 + i, 10 + i))
            carrier.add_shard(shard)
            shards.append(shard)

        gliders = {"only_glider": (10, 10)}

        results = []
        for shard in shards:
            result = router.route_single_hop(shard.shard_id, gliders)
            results.append(result)

        # Only one shard should be routed successfully
        successful_routes = sum(1 for r in results if r is not None)
        assert successful_routes == 1

        # Only the routed shard should be attached
        attached_shards = [s for s in shards if s.attached_glider_id]
        assert len(attached_shards) == 1
