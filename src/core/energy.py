"""Energy field mechanics for Conway's Game of Life cellular automaton.

Implements exponential decay energy overlays that modulate pattern persistence
and provide distance-based attenuation for the CyberMesh substrate. Energy
fields influence glider lifetimes and routing decisions in Phase 3.
"""

import numpy as np
from typing import Tuple, Optional, Dict, List
from .grid import Grid
import logging

logger = logging.getLogger(__name__)


class EnergyField:
    """Exponential decay energy field overlay for cellular automaton."""

    def __init__(self, width: int, height: int, decay_lambda: float = 10.0):
        """Initialize energy field.

        Args:
            width: Field width in cells
            height: Field height in cells
            decay_lambda: Decay constant (higher = slower decay)
        """
        if not (1.0 <= decay_lambda <= 100.0):
            raise ValueError("Decay lambda must be between 1.0 and 100.0")

        self.width = width
        self.height = height
        self.decay_lambda = decay_lambda

        # Energy field: 2D float array representing energy intensity
        self.field = np.zeros((height, width), dtype=np.float32)

        # Origin point for distance calculations
        self.origin_x: Optional[int] = None
        self.origin_y: Optional[int] = None

    def set_origin(self, x: int, y: int) -> None:
        """Set the energy field origin point.

        Args:
            x: Origin X coordinate
            y: Origin Y coordinate
        """
        self.origin_x = x
        self.origin_y = y

        # Recalculate field with new origin
        self._calculate_field()

    def _calculate_field(self) -> None:
        """Calculate exponential decay field from origin."""
        if self.origin_x is None or self.origin_y is None:
            self.field.fill(0.0)
            return

        # Calculate distance matrix from origin
        y_coords, x_coords = np.ogrid[:self.height, :self.width]
        distances = np.sqrt((x_coords - self.origin_x)**2 + (y_coords - self.origin_y)**2)

        # Exponential decay: E(r) = exp(-r/λ)
        self.field = np.exp(-distances / self.decay_lambda)

    def get_energy(self, x: int, y: int) -> float:
        """Get energy intensity at coordinates.

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            Energy intensity (0.0 to 1.0)
        """
        if not (0 <= x < self.width and 0 <= y < self.height):
            return 0.0
        return float(self.field[y, x])

    def decay_energy(self, factor: float = 0.9) -> None:
        """Apply global energy decay over time.

        Args:
            factor: Decay factor (0.0 = full decay, 1.0 = no decay)
        """
        if not (0.0 <= factor <= 1.0):
            raise ValueError("Decay factor must be between 0.0 and 1.0")

        self.field *= factor

        # Recalculate from origin to maintain spatial decay pattern
        self._calculate_field()

    def copy(self) -> 'EnergyField':
        """Create a deep copy of the energy field."""
        new_field = EnergyField(self.width, self.height, self.decay_lambda)
        new_field.field = self.field.copy()
        new_field.origin_x = self.origin_x
        new_field.origin_y = self.origin_y
        return new_field

    def __eq__(self, other: object) -> bool:
        """Check equality with another energy field."""
        if not isinstance(other, EnergyField):
            return False
        return (self.width == other.width and
                self.height == other.height and
                self.decay_lambda == other.decay_lambda and
                self.origin_x == other.origin_x and
                self.origin_y == other.origin_y and
                np.allclose(self.field, other.field))


class EnhancedConwayEngine:
    """Conway's rules engine with energy field modulation."""

    def __init__(self, energy_decay_lambda: float = 10.0):
        """Initialize enhanced Conway engine with energy field.

        Args:
            energy_decay_lambda: Energy field decay constant
        """
        self.base_engine = __import__('src.core.conway', fromlist=['ConwayEngine']).ConwayEngine()
        self.energy_decay_lambda = energy_decay_lambda
        self.energy_field: Optional[EnergyField] = None

    def attach_energy_field(self, field: EnergyField) -> None:
        """Attach an energy field to modulate rule outcomes.

        Args:
            field: Energy field to attach
        """
        self.energy_field = field

    def detach_energy_field(self) -> None:
        """Detach the energy field."""
        self.energy_field = None

    def update_cell_with_energy(self, grid: Grid, x: int, y: int) -> bool:
        """Apply Conway's rules modulated by energy field.

        Args:
            grid: Current grid state
            x: Cell X coordinate
            y: Cell Y coordinate

        Returns:
            Next cell state (True=alive, False=dead)
        """
        # Get base Conway rule result
        base_result = self.base_engine.update_cell(grid, x, y)

        # If no energy field, return base result
        if self.energy_field is None:
            return base_result

        # Get energy at this location
        energy = self.energy_field.get_energy(x, y)
        current_state = grid[y, x]

        if current_state:  # Alive cell
            # Energy increases survival probability
            # Low energy -> higher chance of dying prematurely
            survival_probability = 0.5 + 0.5 * energy  # 0.5 to 1.0 range

            if np.random.random() > survival_probability:
                return False  # Premature death due to low energy

        else:  # Dead cell
            # High energy slightly increases birth probability for reproduction
            # This creates stability gradients in the field
            if base_result and energy > 0.7:  # High energy stable regions
                return True
            elif base_result and energy < 0.3:  # Low energy chaotic regions
                # 20% chance of failed birth
                if np.random.random() < 0.2:
                    return False

        return base_result

    def step_with_energy(self, grid: Grid) -> None:
        """Evolve grid one step with energy modulation.

        Args:
            grid: Grid to evolve in-place
        """
        if self.energy_field is None:
            # Fall back to base engine
            self.base_engine.step(grid)
            return

        # Create new grid state with energy modulation
        new_grid = Grid(grid.width, grid.height)

        for y in range(grid.height):
            for x in range(grid.width):
                new_state = self.update_cell_with_energy(grid, x, y)
                new_grid[y, x] = new_state

        # Copy new state back
        grid.state[:] = new_grid.state[:]

        # Apply global energy decay
        if self.energy_field is not None:
            self.energy_field.decay_energy(0.95)  # 5% decay per step


def create_stable_energy_regions(grid_size: int = 16, num_regions: int = 3) -> EnergyField:
    """Create energy field with stable high-energy regions.

    This creates natural attractors for glider movement and pattern formation.

    Args:
        grid_size: Square grid size
        num_regions: Number of high-energy regions to create

    Returns:
        Configured energy field
    """
    field = EnergyField(grid_size, grid_size, decay_lambda=5.0)  # Faster decay

    # Instead of single origin, create multiple stable regions
    # Use multiple overlapping fields for complex patterns

    # For Phase 1, just demonstrate the concept with single origin
    # Phase 3 will expand this with multiple energy sources
    center = grid_size // 2
    field.set_origin(center, center)

    return field


def create_gradient_field(grid_size: int = 32, direction: str = 'radial') -> EnergyField:
    """Create directional or radial energy gradients.

    Args:
        grid_size: Square grid size
        direction: 'radial' or 'linear'

    Returns:
        Configured energy field with gradient
    """
    field = EnergyField(grid_size, grid_size)

    if direction == 'radial':
        # Radial gradient from center
        field.set_origin(grid_size // 2, grid_size // 2)

    elif direction == 'linear':
        # Linear gradient (higher at top, lower at bottom)
        y_coords = np.arange(grid_size)
        gradient = 1.0 - (y_coords / (grid_size - 1))  # 1.0 at top, 0.0 at bottom

        # Broadcast to 2D
        field.field = np.tile(gradient.reshape(-1, 1), (1, grid_size))
    else:
        raise ValueError(f"Unsupported direction: {direction}")

    return field


# Example usage functions for Phase 1 demonstration

def demonstrate_energy_effect():
    """Demonstrate energy field effects on pattern evolution.

    Returns:
        Tuple of (grid_with_energy, grid_without_energy, energy_field)
    """
    from src.core.grid import Grid

    # Create identical starting grids
    grid_with_energy = Grid(16, 16)
    grid_without_energy = Grid(16, 16)

    # Add a glider pattern to both
    glider_pattern = np.array([
        [False, True, False],
        [False, False, True],
        [True, True, True]
    ], dtype=bool)

    # Place glider in center
    center = 8
    for dy in range(3):
        for dx in range(3):
            if glider_pattern[dy, dx]:
                grid_with_energy[center + dy - 1, center + dx - 1] = True
                grid_without_energy[center + dy - 1, center + dx - 1] = True

    # Create energy field with center origin
    energy_field = EnergyField(16, 16, decay_lambda=8.0)
    energy_field.set_origin(center, center)

    # Set up enhanced engine
    enhanced_engine = EnhancedConwayEngine()
    enhanced_engine.attach_energy_field(energy_field)

    # Evolve both grids for 8 steps
    base_engine = __import__('src.core.conway', fromlist=['default_engine']).default_engine

    for _ in range(8):
        enhanced_engine.step_with_energy(grid_with_energy)
        base_engine.step(grid_without_energy)

    return grid_with_energy, grid_without_energy, energy_field
