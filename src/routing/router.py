"""Shard routing algorithms for hop-by-hop delivery.

Implements path finding and forwarding logic for vector shard transport
across the cellular automaton grid. Uses similarity-based routing with
capacity constraints and congestion avoidance.
"""

import heapq
from typing import Dict, List, Tuple, Set, Optional, NamedTuple
from collections import defaultdict
import logging
from .shard import VectorShard
from .carrier import ShardCarrier
from ..embeddings.similarity import cosine_similarity
from ..core.grid import Grid

logger = logging.getLogger(__name__)


class RouteNode(NamedTuple):
    """Node in routing graph with position and cost information."""
    position: Tuple[int, int]
    total_cost: float
    shard_id: str
    path: List[Tuple[int, int]]


class SingleHopRouter:
    """Single-hop routing with immediate neighbor forwarding.

    Implements simple but effective routing by evaluating immediate
    neighbors and selecting lowest-cost forwarding option.
    """

    def __init__(self, carrier: ShardCarrier,
                 similarity_threshold: float = 0.1,
                 max_search_distance: int = 5):
        """Initialize single-hop router.

        Args:
            carrier: Shard carrier system for transport operations
            similarity_threshold: Minimum similarity improvement required
            max_search_distance: Maximum distance to search for forward options
        """
        self.carrier = carrier
        self.similarity_threshold = similarity_threshold
        self.max_search_distance = max_search_distance

        # Performance tracking
        self.routing_attempts = 0
        self.successful_routes = 0
        self.failed_routes = 0
        self.average_hops = 0.0

    def route_single_hop(self, shard_id: str,
                        available_gliders: Dict[str, Tuple[int, int]]) -> Optional[str]:
        """Route shard one hop toward destination using available gliders.

        Args:
            shard_id: ID of shard to route
            available_gliders: Dict of glider_id -> position

        Returns:
            Selected glider_id if routing successful, None otherwise
        """
        self.routing_attempts += 1

        shard = self.carrier.shards.get(shard_id)
        if not shard:
            logger.warning(f"Shard {shard_id} not found for routing")
            return None

        if shard.is_delivered:
            logger.info(f"Shard {shard_id} already delivered, skipping routing")
            return None

        # Find best available glider for transport
        best_glider = None
        best_cost = float('inf')

        for glider_id, glider_pos in available_gliders.items():
            # Skip gliders already carrying shards
            if self.carrier.get_attached_shard(glider_id):
                continue

            # Calculate routing cost
            cost = shard.get_routing_cost(glider_pos)

            # Add distance penalty (prefer closer gliders)
            distance_penalty = self._calculate_distance_penalty(shard.position, glider_pos)
            total_cost = cost + distance_penalty

            if total_cost < best_cost:
                best_cost = total_cost
                best_glider = glider_id

        if best_glider:
            # Perform attachment
            success = self.carrier.attach_shard_to_glider(shard_id, best_glider, available_gliders[best_glider])
            if success:
                self.successful_routes += 1
                logger.info(f"Successfully routed shard {shard_id} via glider {best_glider}")
                return best_glider
            else:
                self.failed_routes += 1
                logger.warning(f"Failed to attach shard {shard_id} to glider {best_glider}")

        self.failed_routes += 1
        return None

    def _calculate_distance_penalty(self, shard_pos: Tuple[int, int],
                                  glider_pos: Tuple[int, int]) -> float:
        """Calculate distance penalty for routing cost.

        Args:
            shard_pos: Current shard position
            glider_pos: Potential glider position

        Returns:
            Distance-based penalty (higher for farther gliders)
        """
        dx = shard_pos[0] - glider_pos[0]
        dy = shard_pos[1] - glider_pos[1]
        distance = (dx**2 + dy**2)**0.5

        # Exponential penalty for distance
        return distance * 2.0

    def get_routing_stats(self) -> Dict[str, float]:
        """Get routing performance statistics."""
        success_rate = 0.0
        if self.routing_attempts > 0:
            success_rate = self.successful_routes / self.routing_attempts

        return {
            'routing_attempts': self.routing_attempts,
            'successful_routes': self.successful_routes,
            'failed_routes': self.failed_routes,
            'success_rate': success_rate,
            'average_hops': self.average_hops
        }


class MultiHopRouter:
    """Multi-hop routing with path finding across grid.

    Uses A* search with embedding similarity heuristics to find
    optimal paths through 3x3 and larger grids.
    """

    def __init__(self, carrier: ShardCarrier,
                 grid_bounds: Tuple[int, int, int, int],
                 similarity_weight: float = 0.7,
                 distance_weight: float = 0.3):
        """Initialize multi-hop router.

        Args:
            carrier: Shard carrier system
            grid_bounds: (min_x, min_y, max_x, max_y) of routing area
            similarity_weight: Weight for similarity in cost function (0.0-1.0)
            distance_weight: Weight for distance in cost function (0.0-1.0)
        """
        self.carrier = carrier
        self.min_x, self.min_y, self.max_x, self.max_y = grid_bounds
        self.similarity_weight = similarity_weight
        self.distance_weight = distance_weight

        # Ensure weights sum to 1.0
        total_weight = similarity_weight + distance_weight
        if total_weight > 0:
            self.similarity_weight /= total_weight
            self.distance_weight /= total_weight

        # Performance tracking
        self.pathfinding_attempts = 0
        self.successful_paths = 0
        self.average_path_length = 0.0
        self.total_nodes_explored = 0

    def find_multi_hop_route(self, shard_id: str, max_hops: int = 10) -> Optional[List[Tuple[int, int]]]:
        """Find multi-hop route for shard delivery using A* search.

        Args:
            shard_id: ID of shard to route
            max_hops: Maximum hops to consider

        Returns:
            List of positions forming optimal path, or None if no path found
        """
        self.pathfinding_attempts += 1

        shard = self.carrier.shards.get(shard_id)
        if not shard or shard.is_delivered:
            return None

        # A* search for optimal path
        start_pos = shard.position
        goal_pos = shard.destination

        # Priority queue for A* search: (f_score, g_score, position, path)
        frontier = []
        heapq.heappush(frontier, (0.0, 0.0, start_pos, [start_pos]))

        # Cost tracking
        g_score = {start_pos: 0.0}  # Cost from start to node
        f_score = {start_pos: self._heuristic(start_pos, goal_pos, shard)}

        # Explored nodes
        explored = set()

        nodes_explored = 0

        while frontier:
            current_f, current_g, current_pos, current_path = heapq.heappop(frontier)

            if current_pos in explored:
                continue

            explored.add(current_pos)
            nodes_explored += 1

            # Check if we've reached the goal
            if self._positions_equal(current_pos, goal_pos):
                # Update statistics
                self.successful_paths += 1
                self.average_path_length = (
                    (self.average_path_length * (self.successful_paths - 1)) +
                    len(current_path)
                ) / self.successful_paths
                self.total_nodes_explored += nodes_explored

                logger.info(f"Found route for shard {shard_id}: {len(current_path)} hops")
                return current_path

            # Don't exceed max hops
            if len(current_path) >= max_hops:
                continue

            # Explore neighbors
            for neighbor in self._get_neighbors(current_pos):
                tentative_g = current_g + self._movement_cost(current_pos, neighbor, shard)

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(neighbor, goal_pos, shard)

                    new_path = current_path + [neighbor]
                    heapq.heappush(frontier, (f_score[neighbor], tentative_g, neighbor, new_path))

        # No path found
        self.total_nodes_explored += nodes_explored
        logger.warning(f"No route found for shard {shard_id} within {max_hops} hops")
        return None

    def _heuristic(self, position: Tuple[int, int],
                  goal: Tuple[int, int], shard: VectorShard) -> float:
        """A* heuristic estimate of distance to goal.

        Args:
            position: Current position
            goal: Goal position
            shard: Shard being routed (for similarity adjustments)

        Returns:
            Estimated cost to goal
        """
        # Euclidean distance with similarity weighting
        dx = position[0] - goal[0]
        dy = position[1] - goal[1]
        distance = (dx**2 + dy**2)**0.5

        # Basic similarity bonus (closer positions are more similar)
        # This is a simplification - real routing would use embedding similarities
        position_similarity = max(0, 1.0 - distance / 50.0)  # Assume max distance ~50

        return (self.distance_weight * distance +
                self.similarity_weight * (1.0 - position_similarity))

    def _movement_cost(self, from_pos: Tuple[int, int],
                      to_pos: Tuple[int, int], shard: VectorShard) -> float:
        """Cost of moving from one position to another.

        Args:
            from_pos: Starting position
            to_pos: Ending position
            shard: Shard being moved

        Returns:
            Movement cost
        """
        # Base distance cost
        dx = from_pos[0] - to_pos[0]
        dy = from_pos[1] - to_pos[1]
        distance = (dx**2 + dy**2)**0.5

        # Check for capacity constraints (simplified)
        if self.carrier.get_shard_at_position(to_pos):
            # Small penalty for positions with existing shards
            capacity_penalty = 1.0
        else:
            capacity_penalty = 0.0

        # Shard-specific routing cost
        routing_cost = shard.get_routing_cost(to_pos)

        return distance + capacity_penalty + routing_cost * 0.1

    def _get_neighbors(self, position: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get valid neighbor positions for routing.

        Args:
            position: Current position

        Returns:
            List of valid neighbor positions
        """
        x, y = position
        neighbors = []

        # Check all 8 directions
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:  # Skip center
                    continue

                nx, ny = x + dx, y + dy

                # Check bounds
                if (self.min_x <= nx <= self.max_x and
                    self.min_y <= ny <= self.max_y):
                    neighbors.append((nx, ny))

        return neighbors

    def _positions_equal(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> bool:
        """Check if two positions are equal."""
        return pos1[0] == pos2[0] and pos1[1] == pos2[1]

    def get_routing_stats(self) -> Dict[str, float]:
        """Get pathfinding performance statistics."""
        success_rate = 0.0
        if self.pathfinding_attempts > 0:
            success_rate = self.successful_paths / self.pathfinding_attempts

        return {
            'pathfinding_attempts': self.pathfinding_attempts,
            'successful_paths': self.successful_paths,
            'success_rate': success_rate,
            'average_path_length': self.average_path_length,
            'total_nodes_explored': self.total_nodes_explored
        }


def create_3x3_router(carrier: ShardCarrier) -> MultiHopRouter:
    """Create router optimized for 3x3 grid scenarios."""
    # Standard 3x3 bounds centered around origin
    return MultiHopRouter(carrier, (-1, -1, 1, 1))


def simulate_single_hop_delivery(router: SingleHopRouter,
                                shards: List[VectorShard],
                                available_gliders: Dict[str, Tuple[int, int]],
                                max_steps: int = 10) -> Dict[str, float]:
    """Simulate single-hop delivery scenario.

    Args:
        router: Single-hop router instance
        shards: List of shards to route
        available_gliders: Dict of available gliders
        max_steps: Maximum simulation steps

    Returns:
        Delivery statistics
    """
    step = 0
    for step in range(max_steps):
        # Try to route each unrouted shard
        for shard in shards:
            if not shard.is_delivered and not shard.attached_glider_id:
                router.route_single_hop(shard.shard_id, available_gliders)

        # Move gliders to shard destinations to complete delivery
        for glider_id, glider_pos in available_gliders.items():
            attached_shard = router.carrier.get_attached_shard(glider_id)
            if attached_shard:
                # Move glider directly to shard destination
                router.carrier.update_glider_position(glider_id, attached_shard.destination)

                # Check if delivery is complete and detach if so
                if attached_shard.position == attached_shard.destination:
                    router.carrier.detach_shard_from_glider(glider_id)

        # Process lifecycle (TTL, delivery detection)
        expired, delivered, cleaned = router.carrier.process_shard_lifecycle()

        if expired + delivered + cleaned == 0:
            # No more activity
            break

    # Return final statistics (note: start_shards should be computed from input if needed)
    stats = router.get_routing_stats()
    carrier_stats = router.carrier.get_transport_statistics()

    # Total shards starts with original count, minus what's left in carrier
    # We track this externally for now
    remaining_shards = carrier_stats['total_shards']  # Current shards still in carrier

    return {
        'simulation_steps': step + 1,
        'successful_deliveries': carrier_stats['successful_deliveries'],
        'remaining_shards': remaining_shards,
        'routing_success_rate': stats['success_rate'],
        'expired_shards': carrier_stats['failed_deliveries']
    }


def validate_95_percent_success(stats: Dict[str, float]) -> bool:
    """Validate that routing meets 95% Phase 2 success criterion."""
    total_shards = stats['successful_deliveries'] + stats['remaining_shards'] + stats['expired_shards']
    if total_shards == 0:
        return False

    success_rate = stats['successful_deliveries'] / total_shards
    return success_rate >= 0.95
