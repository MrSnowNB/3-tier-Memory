"""Unit tests for core grid operations.

Tests basic grid functionality including initialization, manipulation,
and utility methods. Ensures memory usage and performance requirements
are met for Phase 1.
"""

import pytest
import numpy as np
import psutil
import os
from src.core.grid import Grid


class TestGridInitialization:
    """Test grid initialization and basic properties."""

    def test_minimum_grid_size(self):
        """Grid must be at least 8x8 per POLICY.md §6.2."""
        Grid(8, 8)  # Should work
        Grid(16, 16)  # Should work

        with pytest.raises(ValueError, match="must be at least 8x8"):
            Grid(7, 8)

        with pytest.raises(ValueError, match="must be at least 8x8"):
            Grid(8, 7)

    def test_maximum_grid_size(self):
        """Grid cannot exceed 512x512 to prevent memory issues."""
        Grid(512, 512)  # Should work

        with pytest.raises(ValueError, match="cannot exceed 512x512"):
            Grid(513, 512)

        with pytest.raises(ValueError, match="cannot exceed 512x512"):
            Grid(512, 513)

    def test_initial_state_dead(self):
        """New grid should be all dead by default."""
        grid = Grid(16, 16)
        assert grid.is_empty()
        assert grid.count_alive() == 0
        assert grid.density() == 0.0

    def test_initial_state_from_array(self):
        """Grid can be initialized from numpy array."""
        initial = np.array([[True, False], [False, True]], dtype=bool)
        grid = Grid(2, 2, initial)

        assert grid[0, 0] is True
        assert grid[1, 0] is False
        assert grid[0, 1] is False
        assert grid[1, 1] is True

    def test_initial_state_validation(self):
        """Invalid initial states raise ValueError."""
        # Wrong shape
        with pytest.raises(ValueError, match="shape.*doesn't match"):
            Grid(4, 4, np.zeros((2, 2), dtype=bool))

        # Wrong dtype
        with pytest.raises(ValueError, match="must be boolean"):
            Grid(2, 2, np.array([[1, 0], [0, 1]], dtype=int))


class TestGridManipulation:
    """Test grid state manipulation methods."""

    def test_get_set_operations(self):
        """Basic get/set operations work correctly."""
        grid = Grid(16, 16)

        # Set cell alive
        grid.set(5, 3, True)
        assert grid.get(5, 3) is True

        # Set cell dead
        grid.set(5, 3, False)
        assert grid.get(5, 3) is False

    def test_indexing_syntax(self):
        """Array-style indexing works."""
        grid = Grid(16, 16)

        grid[2, 4] = True
        assert grid[2, 4] is True

        grid[2, 4] = False
        assert grid[2, 4] is False

    def test_bounds_checking(self):
        """Out-of-bounds access raises IndexError."""
        grid = Grid(16, 16)

        with pytest.raises(IndexError):
            grid.get(-1, 0)

        with pytest.raises(IndexError):
            grid.get(16, 0)

        with pytest.raises(IndexError):
            grid.get(0, -1)

        with pytest.raises(IndexError):
            grid.get(0, 16)

        with pytest.raises(IndexError):
            grid.set(-1, 0, True)

        with pytest.raises(IndexError):
            grid[16, 0] = True

    def test_clear_operation(self):
        """Clear resets all cells to dead."""
        grid = Grid(16, 16)

        # Set some cells alive
        grid[0, 0] = True
        grid[5, 5] = True
        grid[10, 10] = True

        # Clear should reset all
        grid.clear()
        assert grid.is_empty()


def test_memory_usage_verification():
    """Verify grid memory usage meets Phase 1 requirements.

    Target: <50MB for 32x32 grid operations (per Phase 1 spec)
    """
    import psutil
    import os

    # Get memory before creating grid
    process = psutil.Process(os.getpid())
    memory_before = process.memory_info().rss / 1024 / 1024  # MB

    # Create 32x32 grid (Phase 1 target size)
    grid = Grid(32, 32)

    # Perform operations to ensure memory is allocated
    grid.randomize(0.5)  # Fill with random data
    alive_count = grid.count_alive()
    density = grid.density()
    bounds = grid.bounds()

    # Verify operations work
    assert alive_count > 0
    assert 0.3 < density < 0.7  # Approximately half

    # Get memory after operations
    memory_after = process.memory_info().rss / 1024 / 1024  # MB
    memory_used = memory_after - memory_before

    # Phase 1 requirement: <50MB for grid operations
    assert memory_used < 50, f"Grid memory usage {memory_used:.1f}MB exceeds 50MB limit"

    # Also verify reasonable memory per cell
    bytes_per_cell = memory_used * 1024 * 1024 / (32 * 32)
    assert bytes_per_cell < 100, f"Memory per cell {bytes_per_cell:.1f} bytes is too high"

    print(".1f")


class TestGridCopy:
    """Test grid copy operations."""

    def test_copy_operation(self):
        """Copy creates independent grid instance."""
        grid1 = Grid(16, 16)
        grid1[2, 3] = True

        grid2 = grid1.copy()

        # Copy should have same state
        assert grid2[2, 3] is True
        assert grid1 == grid2

        # But be independent
        grid2[2, 3] = False
        assert grid1 != grid2
        assert grid1[2, 3] is True


class TestGridFromPattern:
    """Test creating grids from pattern arrays."""

    def test_from_pattern_basic(self):
        """Basic pattern creation works."""
        pattern = np.array([[True, False],
                           [False, True]], dtype=bool)
        grid = Grid.from_pattern(pattern, pad=2)

        # Should have padding around pattern
        assert grid.width == 6  # 2 + 2*2
        assert grid.height == 6  # 2 + 2*2

        # Pattern should be in center (offset by padding)
        assert grid[2, 2] is True   # pattern[0,0]
        assert grid[3, 2] is False  # pattern[1,0]
        assert grid[2, 3] is False  # pattern[0,1]
        assert grid[3, 3] is True   # pattern[1,1]

    def test_from_pattern_no_padding(self):
        """Pattern creation with zero padding."""
        pattern = np.array([[True]], dtype=bool)
        grid = Grid.from_pattern(pattern, pad=0)

        assert grid.width == 1
        assert grid.height == 1
        assert grid[0, 0] is True


class TestGridQueries:
    """Test grid state query methods."""

    def test_count_alive(self):
        """Count alive cells correctly."""
        grid = Grid(4, 4)

        assert grid.count_alive() == 0

        grid[0, 0] = True
        assert grid.count_alive() == 1

        grid[1, 1] = True
        grid[2, 2] = True
        assert grid.count_alive() == 3

    def test_density_calculation(self):
        """Density calculation is accurate."""
        grid = Grid(4, 4)  # 16 cells total

        assert grid.density() == 0.0

        # Add 4 alive cells
        grid[0, 0] = True
        grid[1, 1] = True
        grid[2, 2] = True
        grid[3, 3] = True

        assert grid.density() == 4/16  # 0.25

    def test_is_empty(self):
        """Empty detection works."""
        grid = Grid(8, 8)

        assert grid.is_empty()

        grid[3, 3] = True
        assert not grid.is_empty()

        grid.clear()
        assert grid.is_empty()

    def test_bounds_calculation(self):
        """Bounding box calculation correct."""
        grid = Grid(16, 16)

        # Empty grid bounds
        assert grid.bounds() == (0, 0, 15, 15)

        # Single cell
        grid[5, 7] = True
        assert grid.bounds() == (5, 7, 5, 7)

        # Multiple cells
        grid[10, 2] = True
        grid[1, 12] = True
        assert grid.bounds() == (1, 2, 10, 12)


class TestGridRandomization:
    """Test grid randomization functionality."""

    def test_randomize_densities(self):
        """Randomization respects density parameter."""
        grid = Grid(100, 100)

        # Low density
        grid.randomize(0.1)
        density = grid.density()
        assert 0.05 < density < 0.15  # Approximate range

        # High density
        grid.randomize(0.9)
        density = grid.density()
        assert 0.85 < density < 0.95  # Approximate range

    def test_randomize_edge_cases(self):
        """Edge case densities work."""
        grid = Grid(10, 10)

        grid.randomize(0.0)  # No cells alive
        assert grid.is_empty()

        grid.randomize(1.0)  # All cells alive
        assert grid.count_alive() == 100


class TestGridEquality:
    """Test grid equality comparison."""

    def test_identity_equality(self):
        """Grid is equal to itself."""
        grid = Grid(8, 8)
        assert grid == grid

    def test_deep_copy_equality(self):
        """Grid equals its copy."""
        grid1 = Grid(8, 8)
        grid1[2, 3] = True
        grid2 = grid1.copy()

        assert grid1 == grid2

    def test_different_sizes_inequal(self):
        """Grids of different sizes are not equal."""
        grid1 = Grid(8, 8)
        grid2 = Grid(8, 10)

        assert grid1 != grid2

    def test_different_patterns_inequal(self):
        """Grids with different patterns are not equal."""
        grid1 = Grid(8, 8)
        grid2 = Grid(8, 8)

        grid1[0, 0] = True
        grid2[1, 1] = True

        assert grid1 != grid2

    def test_non_grid_comparison(self):
        """Grid is not equal to non-grid objects."""
        grid = Grid(8, 8)
        assert grid != "not a grid"
        assert grid != 42
        assert grid != None


@pytest.mark.parametrize("width,height", [
    (8, 8),
    (16, 16),
    (32, 32),
    (64, 64),
])
class TestGridSizes:
    """Parameterized tests for different grid sizes."""

    def test_get_set_any_size(self, width, height):
        """Basic operations work for any valid grid size."""
        grid = Grid(width, height)

        # Test basic operations
        grid[width//2, height//2] = True
        assert grid[width//2, height//2] is True
        assert grid.count_alive() == 1

        grid.clear()
        assert grid.is_empty()
