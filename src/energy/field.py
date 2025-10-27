"""
Energy Field Core Implementation

Dynamic energy gradients that propagate through the CA substrate and influence
glider movement. Energy fields create attractive and repulsive forces that shape
emergent routing behavior.
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
import logging
import time

logger = logging.getLogger(__name__)

def _convolve2d_fallback(grid: np.ndarray, kernel: np.ndarray, mode: str = 'same') -> np.ndarray:
    """Fallback 2D convolution implementation using numpy only.

    Args:
        grid: Input grid to convolve
        kernel: Convolution kernel
        mode: Convolution mode ('same' for our use case)

    Returns:
        Convolved grid with toroidal boundary conditions
    """
    grid_height, grid_width = grid.shape
    kernel_size = kernel.shape[0]
    pad = kernel_size // 2

    # Create padded grid with toroidal boundary conditions
    padded = np.zeros((grid_height + 2*pad, grid_width + 2*pad))
    padded[pad:grid_height+pad, pad:grid_width+pad] = grid

    # Wrap boundaries toroidally
    padded[:pad, :] = padded[grid_height:grid_height+pad, :]  # Top
    padded[grid_height+pad:, :] = padded[pad:2*pad, :]        # Bottom
    padded[:, :pad] = padded[:, grid_width:grid_width+pad]   # Left
    padded[:, grid_width+pad:] = padded[:, pad:2*pad]        # Right
    # Corners are handled by the above assignments

    # Perform convolution
    result = np.zeros_like(grid, dtype=np.float32)
    for y in range(grid_height):
        for x in range(grid_width):
            # Extract kernel-sized region from padded grid
            region = padded[y:y+kernel_size, x:x+kernel_size]
            result[y, x] = np.sum(region * kernel)

    return result


class EnergyField:
    """Dynamic energy field that creates gradients affecting CA glider movement.

    Energy fields diffuse across the grid, creating attractive/repulsive forces
    that influence glider routing decisions and create emergent behavior.
    """

    def __init__(self, width: int, height: int, decay_rate: float = 0.95,
                 diffusion_kernel_size: int = 3):
        """Initialize energy field.

        Args:
            width: Grid width in cells
            height: Grid height in cells
            decay_rate: Energy decay per diffusion step (0.0-1.0)
            diffusion_kernel_size: Size of diffusion convolution kernel (odd integer)

        Raises:
            ValueError: If dimensions are invalid or decay_rate out of range
        """
        if width < 1 or height < 1:
            raise ValueError("Grid dimensions must be positive")
        if not (0.0 < decay_rate <= 1.0):
            raise ValueError("Decay rate must be in (0.0, 1.0]")
        if diffusion_kernel_size % 2 != 1 or diffusion_kernel_size < 1:
            raise ValueError("Diffusion kernel size must be odd positive integer")

        self.width = width
        self.height = height
        self.decay_rate = decay_rate
        self.diffusion_kernel_size = diffusion_kernel_size

        # Energy grid: 2D array storing energy values (0.0 to 1.0)
        self.energy_grid = np.zeros((height, width), dtype=np.float32)

        # Energy sources: position -> strength mapping for maintaining fixed energy
        self.sources: Dict[Tuple[int, int], float] = {}

        # Performance tracking
        self.last_diffusion_time = 0.0
        self.diffusion_call_count = 0

        # Create diffusion kernel (Gaussian-like smoothing)
        self.diffusion_kernel = self._create_diffusion_kernel()

        logger.debug(f"Created energy field {width}x{height} with decay={decay_rate}")

    def _create_diffusion_kernel(self) -> np.ndarray:
        """Create a diffusion convolution kernel for smoothing energy propagation."""
        size = self.diffusion_kernel_size
        center = size // 2

        # Create Gaussian-like kernel
        y_coords, x_coords = np.ogrid[:size, :size]
        distances = np.sqrt((x_coords - center)**2 + (y_coords - center)**2)

        # Gaussian kernel with sigma equal to kernel radius
        sigma = center / 2.0
        kernel = np.exp(-distances**2 / (2 * sigma**2))

        # Normalize so sum equals 1.0
        kernel = kernel / kernel.sum()

        return kernel.astype(np.float32)

    def add_energy_source(self, position: Tuple[int, int], energy: float = 1.0) -> None:
        """Add a persistent energy source at a grid position.

        Args:
            position: (x, y) coordinates in grid
            energy: Energy strength (0.0 to 1.0)

        Raises:
            ValueError: If position is invalid or energy out of range
        """
        x, y = position
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise ValueError(f"Position {position} out of grid bounds")

        if not (0.0 <= energy <= 1.0):
            raise ValueError("Energy strength must be in [0.0, 1.0]")

        self.sources[position] = energy
        self.energy_grid[y, x] = energy  # Note: numpy indexing is [y, x]

        logger.debug(f"Added energy source at {position} with strength {energy}")

    def remove_energy_source(self, position: Tuple[int, int]) -> None:
        """Remove an energy source.

        Args:
            position: (x, y) coordinates to remove source from
        """
        if position in self.sources:
            del self.sources[position]
            logger.debug(f"Removed energy source at {position}")

    def add_energy_burst(self, center: Tuple[int, int], radius: int = 3,
                        strength: float = 0.5, decay_exponent: float = 2.0) -> None:
        """Add a temporary energy burst around a center point.

        Args:
            center: (x, y) center coordinates
            radius: Affected radius in cells
            strength: Maximum energy strength at center
            decay_exponent: How quickly energy decays with distance (higher = sharper)
        """
        cx, cy = center
        y_coords, x_coords = np.ogrid[:self.height, :self.width]

        # Calculate distances from center
        distances = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)

        # Create energy burst pattern (exponential decay)
        energy_burst = strength * np.exp(-distances / radius ** decay_exponent)

        # Clip to maximum grid value
        energy_burst = np.clip(energy_burst, 0.0, 1.0)

        # Add to existing energy (temporary, will be diffused)
        self.energy_grid = np.clip(self.energy_grid + energy_burst, 0.0, 1.0)

        logger.debug(f"Added energy burst at {center} with radius {radius}, strength {strength}")

    def diffuse(self, steps: int = 1) -> float:
        """Apply energy diffusion across the grid.

        Uses convolution to smoothly propagate energy according to diffusion equation.

        Args:
            steps: Number of diffusion steps to apply

        Returns:
            Time taken for diffusion in seconds
        """
        start_time = time.time()
        original_grid = self.energy_grid.copy()

        # Check if scipy is available locally
        scipy_signal = None
        try:
            import scipy.signal  # type: ignore
            scipy_signal = scipy.signal
            has_scipy = True
        except ImportError:
            has_scipy = False

        for step in range(steps):
            # Apply diffusion convolution
            if has_scipy and scipy_signal is not None:
                diffused = scipy_signal.convolve2d(
                    self.energy_grid,
                    self.diffusion_kernel,
                    mode='same',
                    boundary='wrap'  # Toroidal boundary for seamless wrapping
                )
            else:
                diffused = _convolve2d_fallback(
                    self.energy_grid,
                    self.diffusion_kernel,
                    mode='same'
                )

            # Apply decay
            self.energy_grid = diffused * self.decay_rate

            # Reapply persistent energy sources (resist diffusion)
            for (x, y), strength in self.sources.items():
                self.energy_grid[y, x] = strength

            # Ensure values stay in valid range
            self.energy_grid = np.clip(self.energy_grid, 0.0, 1.0)

        elapsed = time.time() - start_time
        self.last_diffusion_time = elapsed
        self.diffusion_call_count += 1

        # Calculate diffusion stability (how much change occurred)
        total_change = np.abs(self.energy_grid - original_grid).sum()
        stability = 1.0 - (total_change / (self.width * self.height))

        logger.debug(".2f")
        return elapsed

    def get_energy(self, position: Tuple[int, int]) -> float:
        """Get energy value at a specific grid position.

        Args:
            position: (x, y) coordinates

        Returns:
            Energy value (0.0 to 1.0)

        Raises:
            ValueError: If position is out of bounds
        """
        x, y = position
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise ValueError(f"Position {position} out of grid bounds")

        return float(self.energy_grid[y, x])

    def get_gradient(self, position: Tuple[int, int]) -> Tuple[float, float]:
        """Calculate energy gradient (slope) at a position.

        Uses Sobel operator to estimate the rate of energy change in x and y directions.

        Args:
            position: (x, y) coordinates

        Returns:
            (dx, dy) gradient vector showing energy increase direction

        Raises:
            ValueError: If position is out of bounds or too close to boundary
        """
        x, y = position

        if not (1 <= x < self.width - 1 and 1 <= y < self.height - 1):
            raise ValueError(f"Position {position} too close to boundary for gradient calculation")

        # Sobel operator for gradient approximation
        # Horizontal gradient (dx)
        dx = (
            self.energy_grid[y-1, x+1] + 2*self.energy_grid[y, x+1] + self.energy_grid[y+1, x+1] -
            self.energy_grid[y-1, x-1] - 2*self.energy_grid[y, x-1] - self.energy_grid[y+1, x-1]
        ) / 8.0

        # Vertical gradient (dy)
        dy = (
            self.energy_grid[y+1, x-1] + 2*self.energy_grid[y+1, x] + self.energy_grid[y+1, x+1] -
            self.energy_grid[y-1, x-1] - 2*self.energy_grid[y-1, x] - self.energy_grid[y-1, x+1]
        ) / 8.0

        return (dx, dy)

    def get_gradient_field(self) -> np.ndarray:
        """Generate complete gradient field as vector array.

        Returns:
            Array of shape (height, width, 2) where [:,:,0] is dx, [:,:,1] is dy
        """
        gradient_field = np.zeros((self.height, self.width, 2), dtype=np.float32)

        # Calculate gradients for interior points
        for y in range(1, self.height - 1):
            for x in range(1, self.width - 1):
                dx, dy = self.get_gradient((x, y))
                gradient_field[y, x, 0] = dx
                gradient_field[y, x, 1] = dy

        return gradient_field

    def find_energy_maxima(self, threshold: float = 0.5) -> List[Tuple[int, int]]:
        """Find local energy maxima above threshold.

        Useful for identifying stable attractor positions.

        Args:
            threshold: Minimum energy value to consider

        Returns:
            List of (x, y) positions with local energy maxima
        """
        maxima = []

        for y in range(1, self.height - 1):
            for x in range(1, self.width - 1):
                energy = self.energy_grid[y, x]
                if energy < threshold:
                    continue

                # Check if local maximum (greater than all 8 neighbors)
                neighbors = [
                    self.energy_grid[y-1, x-1], self.energy_grid[y-1, x], self.energy_grid[y-1, x+1],
                    self.energy_grid[y, x-1],   self.energy_grid[y, x+1],
                    self.energy_grid[y+1, x-1], self.energy_grid[y+1, x], self.energy_grid[y+1, x+1]
                ]

                if energy >= max(neighbors):
                    maxima.append((x, y))

        return maxima

    def get_statistics(self) -> Dict[str, float]:
        """Get statistical summary of energy field state."""
        return {
            'mean_energy': float(np.mean(self.energy_grid)),
            'max_energy': float(np.max(self.energy_grid)),
            'min_energy': float(np.min(self.energy_grid)),
            'std_energy': float(np.std(self.energy_grid)),
            'energy_sources': len(self.sources),
            'non_zero_cells': int(np.count_nonzero(self.energy_grid)),
            'last_diffusion_time': self.last_diffusion_time,
            'diffusion_call_count': self.diffusion_call_count
        }

    def reset(self) -> None:
        """Reset energy field to zero, clearing all sources and state."""
        self.energy_grid.fill(0.0)
        self.sources.clear()
        self.last_diffusion_time = 0.0
        self.diffusion_call_count = 0
        logger.info("Energy field reset")

    def copy(self) -> 'EnergyField':
        """Create a deep copy of the energy field."""
        new_field = EnergyField(
            self.width, self.height, self.decay_rate, self.diffusion_kernel_size
        )
        new_field.energy_grid = self.energy_grid.copy()
        new_field.sources = self.sources.copy()
        new_field.last_diffusion_time = self.last_diffusion_time
        new_field.diffusion_call_count = self.diffusion_call_count
        return new_field


def create_linear_energy_field(width: int, height: int,
                              direction: str = 'horizontal',
                              gradient_strength: float = 0.1) -> EnergyField:
    """Create an energy field with linear gradient.

    Args:
        width: Grid width
        height: Grid height
        direction: 'horizontal', 'vertical', or 'diagonal'
        gradient_strength: Maximum gradient strength

    Returns:
        EnergyField with linear gradient pattern
    """
    field = EnergyField(width, height)

    if direction == 'horizontal':
        # Left to right decreasing gradient (higher on left, lower on right)
        for x in range(width):
            energy = (1.0 - x / (width - 1)) * gradient_strength
            for y in range(height):
                field.energy_grid[y, x] = energy

    elif direction == 'vertical':
        # Top to bottom gradient
        for y in range(height):
            energy = (y / (height - 1)) * gradient_strength
            for x in range(width):
                field.energy_grid[y, x] = energy

    elif direction == 'diagonal':
        # Create diagonal gradient
        for y in range(height):
            for x in range(width):
                # Diagonal from top-left to bottom-right
                energy = ((x + y) / (width + height - 2)) * gradient_strength
                field.energy_grid[y, x] = energy
    else:
        raise ValueError(f"Unknown direction: {direction}")

    return field


def create_radial_energy_field(width: int, height: int,
                              center: Optional[Tuple[int, int]] = None,
                              max_radius: Optional[int] = None) -> EnergyField:
    """Create an energy field with radial gradient pattern.

    Args:
        width: Grid width
        height: Grid height
        center: (x, y) center position, defaults to grid center
        max_radius: Maximum radius for energy distribution

    Returns:
        EnergyField with radial energy pattern
    """
    field = EnergyField(width, height)

    if center is None:
        center_x, center_y = width // 2, height // 2
    else:
        center_x, center_y = center

    if max_radius is None:
        max_radius = min(width, height) // 2

    # Create radial energy distribution
    for y in range(height):
        for x in range(width):
            distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)

            if distance <= max_radius:
                # Energy decreases with distance (inverse square law)
                energy = max(0.0, 1.0 - (distance / max_radius)**2)
                field.energy_grid[y, x] = energy

    return field


# Convenience function for testing
def create_test_energy_field() -> EnergyField:
    """Create a small test energy field for unit testing."""
    field = EnergyField(8, 8, decay_rate=0.9)

    # Add some energy sources
    field.add_energy_source((2, 2), 1.0)   # Top-left
    field.add_energy_source((5, 5), 0.8)   # Bottom-right
    field.add_energy_source((1, 6), 0.6)   # Bottom-left

    # Diffuse a few times to create realistic gradient
    field.diffuse(3)

    return field
