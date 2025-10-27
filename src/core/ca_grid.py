"""
Simple Cellular Automaton Grid

Minimal 2D boolean grid implementation that evolves using Conway rules.
No routing, no energy fields, no intelligence - just pure Conway evolution.
"""

import numpy as np
from typing import Optional, Tuple
import logging
from .conway_rules import update_cell, count_live_neighbors, ConwayRuleParams

logger = logging.getLogger(__name__)


class CAGrid:
    """Minimal 2D cellular automaton grid using Conway's Game of Life rules.

    This is the pure baseline CA implementation with no environmental
    modifications or agent intelligence.
    """

    def __init__(self, width: int, height: int):
        """Initialize empty CA grid.

        Args:
            width: Grid width in cells
            height: Grid height in cells

        Raises:
            ValueError: If dimensions are invalid
        """
        if width < 1 or height < 1:
            raise ValueError("Grid dimensions must be positive")

        self.width = width
        self.height = height
        self.grid = np.zeros((height, width), dtype=bool)  # False = dead, True = alive

        # Default Conway rule parameters
        self.rule_params = ConwayRuleParams.standard()

        logger.debug(f"Created CA grid {width}x{height} with Conway rules")

    def set_cell(self, x: int, y: int, alive: bool) -> None:
        """Set state of individual cell.

        Args:
            x: Cell x-coordinate (0 to width-1)
            y: Cell y-coordinate (0 to height-1)
            alive: New cell state

        Raises:
            ValueError: If coordinates are out of bounds
        """
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise ValueError(f"Coordinates ({x}, {y}) out of bounds")

        self.grid[y, x] = alive

    def get_cell(self, x: int, y: int) -> bool:
        """Get state of individual cell.

        Args:
            x: Cell x-coordinate
            y: Cell y-coordinate

        Returns:
            Cell state (True=alive, False=dead)

        Raises:
            ValueError: If coordinates are out of bounds
        """
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise ValueError(f"Coordinates ({x}, {y}) out of bounds")

        return self.grid[y, x]

    def clear(self) -> None:
        """Clear entire grid (set all cells to dead)."""
        self.grid.fill(False)

    def randomize(self, density: float = 0.3) -> None:
        """Randomize grid with given live cell density.

        Args:
            density: Fraction of cells to make alive (0.0 to 1.0)
        """
        density = max(0.0, min(1.0, density))
        random_states = np.random.random((self.height, self.width)) < density
        self.grid[:] = random_states

    def load_pattern(self, pattern: np.ndarray, x: int, y: int) -> None:
        """Load a pattern into the grid at specified position.

        Args:
            pattern: 2D boolean array representing the pattern
            x: Top-left x-coordinate for placement
            y: Top-left y-coordinate for placement
        """
        pattern_height, pattern_width = pattern.shape

        # Copy pattern into grid with wrapping
        for py in range(pattern_height):
            for px in range(pattern_width):
                if pattern[py, px]:
                    grid_x = (x + px) % self.width
                    grid_y = (y + py) % self.height
                    self.grid[grid_y, grid_x] = True

    def step(self) -> int:
        """Evolve grid one time step using current rule parameters.

        Returns:
            Number of live cells after evolution

        Note:
            Uses toroidal boundary conditions (grid wraps around edges)
        """
        new_grid = np.zeros_like(self.grid)
        live_count = 0

        for y in range(self.height):
            for x in range(self.width):
                # Count live neighbors
                neighbors = count_live_neighbors(self.grid, x, y)

                # Apply Conway rules to get next state
                current_state = self.grid[y, x]
                new_state = self.rule_params.update_cell(current_state, neighbors)

                new_grid[y, x] = new_state
                if new_state:
                    live_count += 1

        self.grid[:] = new_grid
        return live_count

    def step_multiple(self, steps: int, log_interval: Optional[int] = None) -> list[int]:
        """Evolve grid multiple steps.

        Args:
            steps: Number of evolution steps
            log_interval: If provided, return live counts at these intervals

        Returns:
            List of live cell counts (either all steps or at intervals)
        """
        live_counts = []

        for step_num in range(steps):
            live_count = self.step()

            if log_interval is None or step_num % log_interval == 0:
                live_counts.append(live_count)

        return live_counts

    def get_live_count(self) -> int:
        """Get total number of live cells."""
        return int(np.sum(self.grid))

    def get_center_of_mass(self) -> Tuple[float, float]:
        """Calculate center of mass of live cells.

        Returns:
            (x, y) coordinates of live cell centroid
        """
        live_coords = np.where(self.grid)
        if len(live_coords[0]) == 0:
            return (0.0, 0.0)

        # live_coords[1] is x coordinates, live_coords[0] is y coordinates
        center_x = float(np.mean(live_coords[1]))
        center_y = float(np.mean(live_coords[0]))

        return (center_x, center_y)

    def copy(self) -> 'CAGrid':
        """Create a deep copy of the grid."""
        new_grid = CAGrid(self.width, self.height)
        new_grid.grid[:] = self.grid
        new_grid.rule_params = ConwayRuleParams(
            self.rule_params.survival_set.copy(),
            self.rule_params.birth_set.copy()
        )
        return new_grid

    def __str__(self) -> str:
        """String representation showing live cells as X."""
        lines = []
        for y in range(self.height):
            line = ""
            for x in range(self.width):
                line += "X" if self.grid[y, x] else "."
            lines.append(line)
        return "\n".join(lines)

    def to_array(self) -> np.ndarray:
        """Get grid as numpy array (for compatibility)."""
        return self.grid.copy()


def create_conway_grid(width: int, height: int) -> CAGrid:
    """Factory function for standard Conway grid."""
    return CAGrid(width, height)


# Classic Conway test patterns
def create_glider_pattern(x: int = 0, y: int = 0) -> Tuple[np.ndarray, int, int]:
    """Create classic Conway glider pattern at position.

    Returns:
        (pattern, center_x, center_y) - pattern and center coordinate
    """
    pattern = np.array([
        [False, True, False],
        [False, False, True],
        [True, True, True]
    ], dtype=bool)

    # Center is at the bottom-right cell where glider motion emanates
    center_x, center_y = x + 2, y + 2
    return pattern, center_x, center_y


def create_blinker_pattern() -> np.ndarray:
    """Create horizontal blinker pattern (3 cells)."""
    return np.array([[True, True, True]], dtype=bool)


def create_block_pattern() -> np.ndarray:
    """Create stable 2x2 block still life."""
    return np.array([
        [True, True],
        [True, True]
    ], dtype=bool)
