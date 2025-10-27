"""Conway's Game of Life rules engine.

Implements the cellular automaton rules that govern cell state transitions
based on neighbor counts. This module provides the core algorithm for
pattern evolution in the CyberMesh substrate.
"""

import numpy as np
from typing import Tuple, Optional
from .grid import Grid
import logging

logger = logging.getLogger(__name__)


class ConwayEngine:
    """Conway's Game of Life rules engine.

    Implements the classic cellular automaton rules:
    - Live cell survives with 2-3 neighbors
    - Dead cell becomes alive with exactly 3 neighbors
    - All other cells die/become dead
    """

    def __init__(self):
        """Initialize the Conway rules engine."""
        # No configuration needed for classic rules
        pass

    def count_neighbors(self, grid: Grid, x: int, y: int) -> int:
        """Count living neighbors of a cell using Moore neighborhood.

        Args:
            grid: The grid containing the cell
            x: X coordinate of cell (column)
            y: Y coordinate of cell (row)

        Returns:
            Number of living neighbors (0-8)
        """
        count = 0

        # Check all 8 neighboring positions
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                # Skip center cell
                if dx == 0 and dy == 0:
                    continue

                nx, ny = x + dx, y + dy

                # Boundary check - cells outside grid are considered dead
                if 0 <= nx < grid.width and 0 <= ny < grid.height:
                    if grid[ny, nx]:  # Note: grid[y,x] where y is row, x is col
                        count += 1

        return count

    def update_cell(self, grid: Grid, x: int, y: int) -> bool:
        """Apply Conway's rules to determine next state of a cell.

        Args:
            grid: Current grid state
            x: X coordinate of cell (column)
            y: Y coordinate of cell (row)

        Returns:
            Next state of the cell (True=alive, False=dead)
        """
        current_state = grid[y, x]  # grid[y,x] where y=row, x=col
        neighbors = self.count_neighbors(grid, x, y)

        # Conway's Game of Life rules
        if current_state:  # Cell is currently alive
            if neighbors in [2, 3]:
                return True  # Survives
            else:
                return False  # Dies (underpopulation or overpopulation)
        else:  # Cell is currently dead
            if neighbors == 3:
                return True  # Becomes alive (reproduction)
            else:
                return False  # Stays dead

    def update_grid(self, grid: Grid) -> Grid:
        """Apply one generation of Conway's rules to entire grid.

        Args:
            grid: Current grid state

        Returns:
            New grid with next generation state
        """
        # Create new grid with same dimensions
        new_grid = Grid(grid.width, grid.height)

        # Update each cell
        for y in range(grid.height):
            for x in range(grid.width):
                new_state = self.update_cell(grid, x, y)
                new_grid[y, x] = new_state

        return new_grid

    def step(self, grid: Grid) -> None:
        """Update grid in-place with next generation.

        Args:
            grid: Grid to update (modified in-place)
        """
        # Create temporary grid to store new state
        temp_grid = self.update_grid(grid)

        # Copy new state back to original grid
        grid.state[:] = temp_grid.state[:]

    def get_rule_table(self) -> dict:
        """Get the complete rule table for all 512 possible neighborhoods.

        Returns:
            Dictionary mapping (current_state, neighbor_count) to next_state
        """
        rules = {}

        for current_state in [False, True]:
            for neighbors in range(9):
                # Apply Conway rules
                if current_state:  # Alive
                    next_state = neighbors in [2, 3]
                else:  # Dead
                    next_state = neighbors == 3

                rules[(current_state, neighbors)] = next_state

        return rules


def generate_neighborhood_pattern(center_alive: bool, neighbors_mask: int) -> np.ndarray:
    """Generate a 3x3 boolean pattern showing a specific neighborhood configuration.

    Args:
        center_alive: Whether center cell is alive
        neighbors_mask: 8-bit integer representing neighbor states (clockwise from top-left)

    Returns:
        3x3 boolean numpy array
    """
    pattern = np.zeros((3, 3), dtype=bool)

    # Center cell
    pattern[1, 1] = center_alive

    # Decode neighbor mask (8 bits represent 8 neighbors clockwise from top-left)
    neighbor_positions = [
        (0, 0), (0, 1), (0, 2),  # top row
        (1, 2),                  # right
        (2, 2), (2, 1), (2, 0),  # bottom row
        (1, 0)                   # left
    ]

    for i, (y, x) in enumerate(neighbor_positions):
        bit = (neighbors_mask >> (7 - i)) & 1
        pattern[y, x] = bool(bit)

    return pattern


def get_all_possible_rules() -> dict:
    """Get all 512 possible rule outcomes for Conway's Game of Life.

    Returns:
        Dictionary mapping (center_alive, neighbor_mask) to next_center_alive
    """
    rules = {}
    engine = ConwayEngine()

    for center_alive in [False, True]:
        for neighbor_mask in range(256):  # 8 neighbors = 2^8 = 256 combinations
            # Create 3x3 pattern
            pattern = generate_neighborhood_pattern(center_alive, neighbor_mask)

            # Count neighbors (excluding center)
            neighbors = np.sum(pattern) - int(center_alive)

            # Apply Conway rules
            next_state = engine.update_cell(Grid(3, 3, pattern), 1, 1)

            rules[(center_alive, neighbor_mask)] = next_state

    return rules


# Singleton instance for convenience
default_engine = ConwayEngine()
