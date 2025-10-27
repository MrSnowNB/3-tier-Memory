"""Glider-shard binding and carrier transport mechanics.

Implements the binding between vector shards and glider patterns,
providing synchronized movement, position tracking, and transport lifecycle
management for the CyberMesh routing substrate.
"""

import numpy as np
from typing import Dict, List, Tuple, Set, Optional, Iterator
from collections import defaultdict
from .shard import VectorShard
from ..patterns.glider import GliderOrientation, GliderPhase, get_glider_pattern
from ..patterns.detector import GliderDetection
from ..core.grid import Grid
import logging

logger = logging.getLogger(__name__)


class ShardCarrier:
    """Manages binding between gliders and transported shards.

    Provides the transport mechanism for vector shards, synchronizing
    their movement with glider patterns while maintaining routing state.
    """

    def __init__(self, max_shards_per_cell: int = 1):
        """Initialize shard carrier system.

        Args:
            max_shards_per_cell: Maximum shards that can occupy same grid cell
        """
        self.max_shards_per_cell = max_shards_per_cell

        # Core data structures
        self.shards: Dict[str, VectorShard] = {}  # shard_id -> VectorShard
        self.cell_occupancy: Dict[Tuple[int, int], Set[str]] = defaultdict(set)  # (x,y) -> set of shard_ids
        self.glider_attachments: Dict[str, str] = {}  # glider_id -> shard_id
        self.shard_attachments: Dict[str, str] = {}  # shard_id -> glider_id

        # Performance tracking
        self.total_attachments = 0
        self.total_detachments = 0
        self.total_transports = 0
        self.failed_deliveries = 0

    def add_shard(self, shard: VectorShard) -> bool:
        """Add a shard to the carrier system.

        Args:
            shard: The shard to add

        Returns:
            True if added successfully, False if rejected (e.g., capacity exceeded)
        """
        # Check if destination cell has capacity
        dest_cell = shard.destination
        if len(self.cell_occupancy[dest_cell]) >= self.max_shards_per_cell:
            logger.warning(f"Cell {dest_cell} at capacity ({self.max_shards_per_cell} shards)")
            return False

        # Add shard
        self.shards[shard.shard_id] = shard
        self.cell_occupancy[dest_cell].add(shard.shard_id)

        logger.debug(f"Added shard {shard.shard_id} targeting {dest_cell}")
        return True

    def remove_shard(self, shard_id: str) -> Optional[VectorShard]:
        """Remove a shard from the carrier system.

        Args:
            shard_id: ID of shard to remove

        Returns:
            The removed shard, or None if not found
        """
        if shard_id not in self.shards:
            return None

        shard = self.shards[shard_id]

        # Clean up attachments and occupancy
        if shard.attached_glider_id:
            self._detach_from_glider(shard_id)

        cell = shard.position
        self.cell_occupancy[cell].discard(shard_id)
        if not self.cell_occupancy[cell]:
            del self.cell_occupancy[cell]

        del self.shards[shard_id]

        logger.debug(f"Removed shard {shard_id}")
        return shard

    def attach_shard_to_glider(self, shard_id: str, glider_id: str,
                               glider_position: Tuple[int, int]) -> bool:
        """Attach a shard to a glider for transport.

        Args:
            shard_id: ID of shard to attach
            glider_id: ID of glider doing transport
            glider_position: Current glider position

        Returns:
            True if attachment successful
        """
        if shard_id not in self.shards:
            logger.warning(f"Shard {shard_id} not found in carrier system")
            return False

        shard = self.shards[shard_id]

        # Check if shard is already attached
        if shard.attached_glider_id:
            logger.warning(f"Shard {shard_id} already attached to glider {shard.attached_glider_id}")
            return False

        # Check if glider is already carrying a shard
        if glider_id in self.glider_attachments:
            logger.warning(f"Glider {glider_id} already carrying shard {self.glider_attachments[glider_id]}")
            return False

        # Perform attachment
        shard.attach_to_glider(glider_id)
        shard.update_position(glider_position)

        self.glider_attachments[glider_id] = shard_id
        self.shard_attachments[shard_id] = glider_id

        # Remove from cell occupancy (now attached to glider)
        old_cell = shard.position
        if old_cell in self.cell_occupancy and shard_id in self.cell_occupancy[old_cell]:
            self.cell_occupancy[old_cell].remove(shard_id)
            if not self.cell_occupancy[old_cell]:
                del self.cell_occupancy[old_cell]

        self.total_attachments += 1

        logger.info(f"Attached shard {shard_id} to glider {glider_id} at {glider_position}")
        return True

    def detach_shard_from_glider(self, glider_id: str) -> Optional[VectorShard]:
        """Detach shard from a glider.

        Args:
            glider_id: ID of glider to detach from

        Returns:
            The detached shard, or None if no attachment
        """
        if glider_id not in self.glider_attachments:
            return None

        shard_id = self.glider_attachments[glider_id]
        shard = self.shards.get(shard_id)

        if shard:
            self._detach_from_glider(shard_id)
            logger.info(f"Detached shard {shard_id} from glider {glider_id}")
            return shard

        return None

    def _detach_from_glider(self, shard_id: str) -> None:
        """Internal detachment logic."""
        if shard_id in self.shard_attachments:
            glider_id = self.shard_attachments[shard_id]

            # Clean up cross-references
            del self.glider_attachments[glider_id]
            del self.shard_attachments[shard_id]

            # Update shard state
            if shard_id in self.shards:
                shard = self.shards[shard_id]
                shard.detach_from_glider()

                # Add back to cell occupancy
                cell = shard.position
                self.cell_occupancy[cell].add(shard_id)

                self.total_detachments += 1

    def update_glider_position(self, glider_id: str, new_position: Tuple[int, int]) -> None:
        """Update position of a glider and its attached shard.

        Args:
            glider_id: ID of glider that moved
            new_position: New glider position
        """
        if glider_id in self.glider_attachments:
            shard_id = self.glider_attachments[glider_id]
            shard = self.shards.get(shard_id)

            if shard:
                old_position = shard.position
                shard.update_position(new_position)

                # Check for delivery
                if shard.position == shard.destination and not shard.attached_glider_id:
                    self.total_transports += 1
                    logger.info(f"Shard {shard_id} delivered to destination {shard.destination}")

                # Check for automatic detachment near destination
                distance = shard.distance_to_destination
                if distance <= 2.0 and np.random.random() < 0.1:  # 10% chance within 2 cells
                    self._detach_from_glider(shard_id)
                    logger.debug(f"Auto-detached shard {shard_id} near destination (distance: {distance:.1f})")

    def process_shard_lifecycle(self) -> Tuple[int, int, int]:
        """Process shard lifecycle events (TTL, delivery, cleanup).

        Returns:
            Tuple of (expired_count, delivered_count, cleaned_count)
        """
        expired = []
        delivered = []
        to_clean = []

        for shard_id, shard in self.shards.items():
            # Decrement TTL
            if shard.decrement_ttl():
                expired.append(shard_id)
                logger.debug(f"Shard {shard_id} expired (TTL reached 0)")
                continue

            # Check for delivery
            if shard.is_delivered():
                delivered.append(shard_id)
                logger.debug(f"Shard {shard_id} delivered successfully")
                continue

            # Mark for cleanup if severely overdue (TTL < -10)
            if shard.ttl < -10:
                to_clean.append(shard_id)
                logger.warning(f"Shard {shard_id} severely overdue, marking for cleanup")

        # Clean up expired and delivered shards
        for shard_id in expired + delivered:
            self.remove_shard(shard_id)

        for shard_id in to_clean:
            shard = self.remove_shard(shard_id)
            if shard:
                self.failed_deliveries += 1

        return len(expired), len(delivered), len(to_clean)

    def get_shard_at_position(self, position: Tuple[int, int]) -> List[VectorShard]:
        """Get all shards at a specific grid position.

        Args:
            position: (x, y) position to query

        Returns:
            List of shards at that position
        """
        shard_ids = self.cell_occupancy.get(position, set())
        return [self.shards[sid] for sid in shard_ids if sid in self.shards]

    def get_attached_shard(self, glider_id: str) -> Optional[VectorShard]:
        """Get shard attached to a specific glider.

        Args:
            glider_id: ID of glider

        Returns:
            Attached shard, or None
        """
        shard_id = self.glider_attachments.get(glider_id)
        return self.shards.get(shard_id) if shard_id else None

    def get_glider_for_shard(self, shard_id: str) -> Optional[str]:
        """Get glider ID carrying a specific shard.

        Args:
            shard_id: ID of shard

        Returns:
            Glider ID, or None if not attached
        """
        return self.shard_attachments.get(shard_id)

    def find_nearest_available_shard(self, position: Tuple[int, int],
                                     max_distance: float = 10.0) -> Optional[VectorShard]:
        """Find nearest shard available for transport.

        Args:
            position: Search origin position
            max_distance: Maximum search distance

        Returns:
            Nearest shard not currently attached, or None
        """
        nearest_shard = None
        min_distance = float('inf')

        for shard_id, shard in self.shards.items():
            if shard.attached_glider_id:  # Skip attached shards
                continue

            distance = shard.distance_to_destination
            if distance <= max_distance and distance < min_distance:
                min_distance = distance
                nearest_shard = shard

        return nearest_shard

    def get_transport_statistics(self) -> Dict[str, int]:
        """Get transport and delivery statistics."""
        attached_shards = len(self.shard_attachments)
        waiting_shards = len(self.shards) - attached_shards

        return {
            'total_shards': len(self.shards),
            'attached_shards': attached_shards,
            'waiting_shards': waiting_shards,
            'total_attachments': self.total_attachments,
            'total_detachments': self.total_detachments,
            'successful_deliveries': self.total_transports,
            'failed_deliveries': self.failed_deliveries
        }

    def clear(self) -> None:
        """Clear all shards and reset carrier state."""
        self.shards.clear()
        self.cell_occupancy.clear()
        self.glider_attachments.clear()
        self.shard_attachments.clear()

        self.total_attachments = 0
        self.total_detachments = 0
        self.total_transports = 0
        self.failed_deliveries = 0

        logger.info("Carrier system cleared")


def bind_shard_to_glider_detection(carrier: ShardCarrier, shard: VectorShard,
                                  detection: GliderDetection) -> bool:
    """Bind a shard to a detected glider.

    Args:
        carrier: Shard carrier system
        shard: Shard to bind
        detection: Glider detection result

    Returns:
        True if binding successful
    """
    glider_id = f"glider_{detection.x}_{detection.y}_{detection.orientation.name}_{detection.phase.name}"
    return carrier.attach_shard_to_glider(shard.shard_id, glider_id, (detection.x, detection.y))


def synchronize_glider_movements(carrier: ShardCarrier, glider_movements: Dict[str, Tuple[int, int]]) -> None:
    """Synchronize shard positions with glider movements.

    Args:
        carrier: Shard carrier system
        glider_movements: Dict of glider_id -> new_position
    """
    for glider_id, new_position in glider_movements.items():
        carrier.update_glider_position(glider_id, new_position)


# Default carrier instance
default_carrier = ShardCarrier()
