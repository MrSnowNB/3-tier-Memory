"""Core grid state management for Conway's Game of Life cellular automaton.

This module implements the fundamental grid data structure that serves as the
spatial substrate for the CyberMesh architecture. The grid uses numpy boolean
arrays for efficient state representation and manipulation.
"""

import numpy as np
from typing import Tuple, Optional, Iterator
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class Grid:
    """2D boolean grid representing Conway's Game of Life state.

    Attributes:
        width: Grid width in cells
        height: Grid height in cells
        state: 2D numpy boolean array (True=alive, False=dead)
    """

    def __init__(self, width: int, height: int, initial_state: Optional[np.ndarray] = None):
        """Initialize grid with given dimensions.

        Args:
            width: Grid width (cells)
            height: Grid height (cells)
            initial_state: Optional initial grid state array

        Raises:
            ValueError: If dimensions are invalid or initial_state shape doesn't match
        """
        if width < 8 or height < 8:  # POLICY.md §6.2 minimum for stable patterns
            raise ValueError("Grid dimensions must be at least 8x8 for stable patterns")

        if width > 512 or height > 512:  # Prevent excessive memory usage
            raise ValueError("Grid dimensions cannot exceed 512x512")

        self.width = width
        self.height = height

        if initial_state is not None:
            if initial_state.shape != (height, width):
                raise ValueError(f"Initial state shape {initial_state.shape} doesn't match grid size {(height, width)}")
            if initial_state.dtype != bool:
                raise ValueError("Initial state must be boolean array")
            self.state = initial_state.copy()
        else:
            self.state = np.zeros((height, width), dtype=bool)

    @classmethod
    def from_pattern(cls, pattern: np.ndarray, pad: int = 4) -> 'Grid':
        """Create grid from pattern array with padding.

        Args:
            pattern: 2D boolean array representing pattern
            pad: Padding cells around pattern

        Returns:
            Grid: New grid containing the pattern
        """
        if pattern.dtype != bool:
            pattern = pattern.astype(bool)

        height, width = pattern.shape
        grid = cls(width + 2*pad, height + 2*pad)
        grid.state[pad:pad+height, pad:pad+width] = pattern
        return grid

    def copy(self) -> 'Grid':
        """Create a deep copy of the grid."""
        return Grid(self.width, self.height, self.state)

    def clear(self) -> None:
        """Reset all cells to dead state."""
        self.state.fill(False)

    def randomize(self, density: float = 0.5) -> None:
        """Randomize grid state with given alive cell density.

        Args:
            density: Probability of cell being alive (0.0 to 1.0)
        """
        self.state = np.random.random((self.height, self.width)) < density

    def get(self, x: int, y: int) -> bool:
        """Get cell state at coordinates.

        Args:
            x: X coordinate (column)
            y: Y coordinate (row)

        Returns:
            True if cell is alive, False if dead

        Raises:
            IndexError: If coordinates are out of bounds
        """
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise IndexError(f"Coordinates ({x}, {y}) out of bounds for {self.width}x{self.height} grid")
        return self.state[y, x]

    def set(self, x: int, y: int, alive: bool) -> None:
        """Set cell state at coordinates.

        Args:
            x: X coordinate (column)
            y: Y coordinate (row)
            alive: True to set alive, False to set dead

        Raises:
            IndexError: If coordinates are out of bounds
        """
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise IndexError(f"Coordinates ({x}, {y}) out of bounds for {self.width}x{self.height} grid")
        self.state[y, x] = alive

    def count_alive(self) -> int:
        """Count total number of alive cells."""
        return int(np.sum(self.state))

    def density(self) -> float:
        """Get fraction of cells that are alive."""
        return self.count_alive() / (self.width * self.height)

    def is_empty(self) -> bool:
        """Check if all cells are dead."""
        return not np.any(self.state)

    def bounds(self) -> Tuple[int, int, int, int]:
        """Get bounding box of alive cells (min_x, min_y, max_x, max_y)."""
        if self.is_empty():
            return (0, 0, self.width-1, self.height-1)

        alive_rows, alive_cols = np.where(self.state)
        return (int(alive_cols.min()), int(alive_rows.min()),
                int(alive_cols.max()), int(alive_rows.max()))

    def __getitem__(self, key: Tuple[int, int]) -> bool:
        """Access cell state using grid[x, y] syntax."""
        x, y = key
        return self.get(x, y)

    def __setitem__(self, key: Tuple[int, int], value: bool) -> None:
        """Set cell state using grid[x, y] = value syntax."""
        x, y = key
        self.set(x, y, value)

    def __eq__(self, other: object) -> bool:
        """Check equality with another grid."""
        if not isinstance(other, Grid):
            return False
        return (self.width == other.width and
                self.height == other.height and
                np.array_equal(self.state, other.state))

    def __str__(self) -> str:
        """String representation showing grid state."""
        alive_char = '█'
        dead_char = '░'

        lines = []
        for y in range(min(10, self.height)):  # Show first 10 rows
            line = ''
            for x in range(min(20, self.width)):  # Show first 20 columns
                line += alive_char if self.state[y, x] else dead_char
            if self.width > 20:
                line += '...'
            lines.append(line)

        if self.height > 10:
            lines.append('...')

        return '\n'.join(lines)

    def __repr__(self) -> str:
        """Detailed representation for debugging."""
        alive_count = self.count_alive()
        density_pct = self.density() * 100
        return f"Grid({self.width}x{self.height}, alive={alive_count}, density={density_pct:.1f}%)"
