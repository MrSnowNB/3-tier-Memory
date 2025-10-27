"""
Pure Conway's Game of Life Rules Implementation

This module provides the fundamental Conway rules for cellular automaton evolution.
No energy fields, routing, or agent intelligence - just pure Conway physics.
"""

from typing import Set, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


# Standard Conway rules - unmodified
SURVIVAL_SET: Set[int] = {2, 3}  # Live cells survive with 2-3 neighbors
BIRTH_SET: Set[int] = {3}        # Dead cells born with exactly 3 neighbors


def update_cell(alive: bool, live_neighbors: int) -> bool:
    """Apply Conway's rules to determine next cell state.

    Args:
        alive: Current cell state (True=alive, False=dead)
        live_neighbors: Number of live neighbors (0-8)

    Returns:
        Next cell state (True=alive, False=dead)
    """
    if alive:
        # Survival rule
        return live_neighbors in SURVIVAL_SET
    else:
        # Birth rule
        return live_neighbors in BIRTH_SET


def count_live_neighbors(grid: 'np.ndarray', x: int, y: int) -> int:
    """Count live neighbors of cell at (x,y) using Moore neighborhood.

    Args:
        grid: 2D boolean numpy array
        x: Cell x-coordinate
        y: Cell y-coordinate

    Returns:
        Number of live neighbors (0-8)
    """
    height, width = grid.shape
    count = 0

    # Check all 8 neighbors
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue  # Skip center cell

            nx, ny = x + dx, y + dy

            # Handle boundaries (wrap around for seamless grid)
            nx = nx % width
            ny = ny % height

            if grid[ny, nx]:
                count += 1

    return count


class ConwayRuleParams:
    """Parameters for Conway rule sets.

    Can be modified by environmental factors (like energy) but defaults
    to standard Conway rules.
    """

    def __init__(self,
                 survival_set: Optional[Set[int]] = None,
                 birth_set: Optional[Set[int]] = None):
        """Initialize rule parameters.

        Args:
            survival_set: Neighbor counts for live cell survival (default {2,3})
            birth_set: Neighbor counts for dead cell birth (default {3})
        """
        self.survival_set: Set[int] = survival_set if survival_set is not None else SURVIVAL_SET.copy()
        self.birth_set: Set[int] = birth_set if birth_set is not None else BIRTH_SET.copy()

    @classmethod
    def standard(cls) -> 'ConwayRuleParams':
        """Create standard Conway rules."""
        return cls(SURVIVAL_SET.copy(), BIRTH_SET.copy())

    def update_cell(self, alive: bool, live_neighbors: int) -> bool:
        """Apply these rule parameters to a cell.

        Args:
            alive: Current cell state
            live_neighbors: Number of live neighbors

        Returns:
            Next cell state
        """
        if alive:
            return live_neighbors in self.survival_set
        else:
            return live_neighbors in self.birth_set

    def __repr__(self) -> str:
        return f"ConwayRuleParams(survival={self.survival_set}, birth={self.birth_set})"
