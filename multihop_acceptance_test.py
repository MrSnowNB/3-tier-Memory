#!/usr/bin/env python3
"""Standalone multi-hop routing acceptance test."""

from src.routing.shard import create_test_shard
from src.routing.carrier import ShardCarrier
from src.routing.router import MultiHopRouter
import time

def test_multi_hop_acceptance():
    print("Running Phase 2 acceptance test: ≥90% multi-hop success...")

    # Test multi-hop routing success rate on 3x3 grid (as per Phase 2 spec)
    carrier = ShardCarrier(max_shards_per_cell=5)  # Allow multiple shards per cell for testing
    router = MultiHopRouter(carrier, (-1, -1, 1, 1))  # 3x3 Phase 2 target grid

    # Create 10 shards with different destinations
    success_count = 0
    total_count = 10

    # Destinations in 3x3 space: avoid clustering
    destinations = [(i//3-1, i%3-1) for i in range(9)]  # 9 unique positions in 3x3 grid

    for i in range(total_count):
        # Use start position far from destination to require multi-hop route
        start_x, start_y = -2, -2  # Corner opposite to destination region

        # Choose destination, or repeat if we have more shards than positions
        dest_idx = min(i, len(destinations) - 1)
        destination = destinations[dest_idx]

        shard = create_test_shard(destination=destination)
        shard.update_position((start_x, start_y))
        carrier.add_shard(shard)

        # Try to find multi-hop route
        start_time = time.time()
        path = router.find_multi_hop_route(shard.shard_id, max_hops=20)
        duration = time.time() - start_time

        if path:
            success_count += 1
            print(f"Shard {i+1}: Route found in {len(path)} hops from {start_x,start_y} to {destination} ({duration:.4f}s)")
        else:
            print(f"Shard {i+1}: No route found from {start_x,start_y} to {destination} ({duration:.4f}s)")

        # Clear the shard to prevent interference (not realistic but validates routing algorithm)
        carrier.remove_shard(shard.shard_id)

    success_rate = success_count / total_count * 100
    print(f"\nMulti-hop success rate: {success_rate:.1f}% (target: ≥90%)")

    # Check if meets Phase 2 criteria
    if success_rate >= 90:
        print("✅ ACCEPTANCE TEST PASSED")
        return True
    else:
        print("❌ ACCEPTANCE TEST FAILED")
        return False

if __name__ == "__main__":
    test_multi_hop_acceptance()
