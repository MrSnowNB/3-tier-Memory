#!/usr/bin/env python3
"""
Detailed memory profiling for Phase 2 routing operations.

Tracks memory usage over multiple cycles to validate <2GB budget
and detect potential leaks in CyberMesh routing system.
"""

import psutil
import os
import time
from typing import List, Dict
import json
import sys

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from routing.shard import create_test_shard
from routing.carrier import ShardCarrier
from routing.router import SingleHopRouter
import gc

def measure_memory_mb() -> float:
    """Get current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def create_test_scenario(num_shards: int = 50, grid_size: int = 32) -> tuple[ShardCarrier, SingleHopRouter, List[str], Dict[str, tuple[int, int]]]:
    """Create a realistic test scenario with shards, carrier, and router."""
    carrier = ShardCarrier(max_shards_per_cell=3)  # Realistic capacity limits
    router = SingleHopRouter(carrier)

    shard_ids = []
    # Create shards with distributed destinations
    for i in range(num_shards):
        shard = create_test_shard(destination=(i % grid_size, (i // grid_size) % grid_size))
        shard.update_position((grid_size//2, grid_size//2))  # Start clustered
        if carrier.add_shard(shard):
            shard_ids.append(shard.shard_id)

    # Create gliders matching shard destinations
    gliders = {f"glider{j}": ((j % grid_size), ((j // grid_size) % grid_size))
              for j in range(min(10, len(shard_ids)))}

    return carrier, router, shard_ids, gliders

def run_routing_cycle(carrier: ShardCarrier, router: SingleHopRouter,
                      shard_ids: List[str], gliders: Dict[str, tuple]) -> Dict[str, int]:
    """Run one complete routing cycle and return statistics."""
    routed = 0

    # Try to route available shards
    for shard_id in shard_ids:
        if router.route_single_hop(shard_id, gliders):
            routed += 1

    # Simulate glider movement (simplified)
    for glider_id in carrier.glider_attachments.copy():
        attached_shard = carrier.get_attached_shard(glider_id)
        if attached_shard:
            # Move toward shard destination
            dx = attached_shard.destination[0] - attached_shard.position[0]
            dy = attached_shard.destination[1] - attached_shard.position[1]
            step_x = 1 if dx > 0 else -1 if dx < 0 else 0
            step_y = 1 if dy > 0 else -1 if dy < 0 else 0

            new_pos = (attached_shard.position[0] + step_x * min(2, abs(dx)),
                      attached_shard.position[1] + step_y * min(2, abs(dy)))

            carrier.update_glider_position(glider_id, new_pos)

            # Auto-detach when close to destination
            if attached_shard.distance_to_destination < 1.5:
                carrier.detach_shard_from_glider(glider_id)

    # Run lifecycle processing
    expired, delivered, cleaned = carrier.process_shard_lifecycle()

    return {
        'routed': routed,
        'expired': expired,
        'delivered': delivered,
        'cleanup': cleaned,
        'total_shards': carrier.get_transport_statistics()['total_shards']
    }

def profile_memory_usage(cycles: int = 20, num_shards: int = 50) -> Dict:
    """Profile memory usage over multiple routing cycles."""
    print(f"🔍 Profiling memory usage over {cycles} cycles with {num_shards} concurrent shards...")
    print("=" * 60)

    # Baseline memory before any operations
    gc.collect()  # Clean up any existing garbage
    time.sleep(0.1)  # Stabilize
    baseline_memory = measure_memory_mb()
    print(f"Baseline Memory:              {baseline_memory:6.1f} MB")
    start_times = []
    end_times = []
    memory_measurements = []
    cycle_stats = []

    # Create test scenario
    carrier, router, shard_ids, gliders = create_test_scenario(num_shards=num_shards)
    scenario_creation_memory = measure_memory_mb()

    print(f"Scenario Creation Memory:     {scenario_creation_memory:6.1f} MB")
    print(f"Initial Memory After Setup:   {scenario_creation_memory:6.1f} MB")
    # Run profiling cycles
    for cycle in range(cycles):
        start_time = time.time()
        start_memory = measure_memory_mb()

        # Run routing cycle
        stats = run_routing_cycle(carrier, router, shard_ids, gliders)

        end_memory = measure_memory_mb()
        end_time = time.time()

        memory_delta = end_memory - start_memory
        time_taken = end_time - start_time

        memory_measurements.append({
            'cycle': cycle + 1,
            'start_memory_mb': start_memory,
            'end_memory_mb': end_memory,
            'delta_mb': memory_delta,
            'time_seconds': time_taken,
            'stats': stats
        })

        start_times.append(start_time)
        end_times.append(end_time)
        cycle_stats.append(stats)

        print(f"Cycle {cycle+1:2d}: {stats['delivered']:2d} delivered, {stats['routed']:2d} routed | "
              f"Memory: {start_memory:6.1f}MB -> {end_memory:6.1f}MB ({memory_delta:+5.1f}MB) | "
              f"Time: {time_taken:.4f}s")

    # Final cleanup
    final_memory = measure_memory_mb()
    gc.collect()

    # Analysis
    memory_deltas = [m['delta_mb'] for m in memory_measurements]
    peak_memory = max([m['end_memory_mb'] for m in memory_measurements])

    # Check for memory leaks (steady upward trend in peak memory)
    memory_trend = []
    for i in range(5, cycles):
        # Look at sliding window of recent cycles
        window = memory_measurements[max(0, i-5):i+1]
        avg_delta = sum(m['delta_mb'] for m in window) / len(window)
        memory_trend.append(avg_delta)

    avg_memory_trend = sum(memory_trend) / len(memory_trend) if memory_trend else 0
    leak_detected = avg_memory_trend > 0.5  # More than 0.5MB average growth per cycle

    # Peaks per operation
    total_operations = sum(stats['routed'] + stats['delivered'] for stats in cycle_stats)
    memory_per_operation = peak_memory / total_operations if total_operations > 0 else 0

    results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'cycles': cycles,
        'num_shards': num_shards,
        'memory_baseline_mb': baseline_memory,
        'memory_scenario_creation_mb': scenario_creation_memory,
        'memory_final_mb': final_memory,
        'peak_memory_mb': peak_memory,
        'avg_memory_delta_mb': sum(memory_deltas) / len(memory_deltas),
        'memory_per_operation_mb': memory_per_operation,
        'total_operations': total_operations,
        'leak_detected': leak_detected,
        'memory_trend_mb_per_cycle': avg_memory_trend,
        'all_measurements': memory_measurements
    }

    print("\n" + "=" * 60)
    print("📊 MEMORY PROFILING SUMMARY")
    print("=" * 60)
    print(f"Baseline Memory:             {results['memory_baseline_mb']:6.1f} MB")
    print(f"Scenario Creation Memory:    {results['memory_scenario_creation_mb']:6.1f} MB")
    print(f"Final Memory After Cleanup:  {results['memory_final_mb']:6.1f} MB")
    print(f"Peak Memory Usage:           {results['peak_memory_mb']:6.1f} MB")
    print(f"Avg Memory Delta per Cycle:  {results['avg_memory_delta_mb']:+6.2f} MB")
    print(f"Memory Trend:                {results['memory_trend_mb_per_cycle']:+6.2f} MB/cycle")
    print(f"Memory per Operation:        {results['memory_per_operation_mb']:.4f} MB/op")
    print(f"Total Operations:            {total_operations:,}")
    print(f"Phase 2 Budget (<2GB):       {'✅ PASSED' if peak_memory < 2000 else '❌ FAILED'}")
    print(f"Leak Detection:               {'❌ LEAK SUSPECTED' if leak_detected else '✅ NO LEAK'}")

    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Memory profiling for Phase 2 routing")
    parser.add_argument("--cycles", type=int, default=20, help="Number of profiling cycles")
    parser.add_argument("--shards", type=int, default=50, help="Number of concurrent shards")
    parser.add_argument("--output", type=str, default="logs/memory_profile.log", help="Output log file")
    args = parser.parse_args()

    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)

    # Run profiling
    results = profile_memory_usage(cycles=args.cycles, num_shards=args.shards)

    # Save detailed results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n📄 Detailed results saved to: {args.output}")
    print("Phase 2 memory profiling complete.")
