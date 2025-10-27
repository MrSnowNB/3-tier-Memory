"""Comprehensive tests for Conway's Game of Life rules.

Tests all rule variations systematically, including all 512 possible
neighborhood configurations. Verifies correct implementation of birth,
survival, and death rules.
"""

import pytest
import numpy as np
from src.core.grid import Grid
from src.core.conway import ConwayEngine, generate_neighborhood_pattern, get_all_possible_rules


class TestConwayEngine:
    """Test Conway rules engine initialization and basic functionality."""

    def test_engine_creation(self):
        """Engine can be created without configuration."""
        engine = ConwayEngine()
        assert engine is not None

    def test_singleton_engine(self):
        """Default singleton engine is available."""
        from src.core.conway import default_engine
        assert isinstance(default_engine, ConwayEngine)


class TestNeighborCounting:
    """Test neighborhood analysis functions."""

    def test_neighbor_count_center_cell(self):
        """Center cell is not counted as a neighbor."""
        engine = ConwayEngine()
        grid = Grid(3, 3)

        # Set center alive, others dead - should have 0 neighbors
        grid[1, 1] = True

        neighbors = engine.count_neighbors(grid, 1, 1)
        assert neighbors == 0

    def test_neighbor_count_all_around(self):
        """Count all 8 neighbors correctly."""
        engine = ConwayEngine()
        grid = Grid(3, 3)

        # Set all neighbors alive, center dead
        for y in range(3):
            for x in range(3):
                if not (x == 1 and y == 1):  # Skip center
                    grid[y, x] = True

        neighbors = engine.count_neighbors(grid, 1, 1)
        assert neighbors == 8

    def test_neighbor_count_boundary_effects(self):
        """Boundary cells have fewer neighbors."""
        engine = ConwayEngine()
        grid = Grid(3, 3)

        # Set all cells alive
        for y in range(3):
            for x in range(3):
                grid[y, x] = True

        # Corner cell (0,0) should have 3 neighbors: (0,1), (1,0), (1,1)
        neighbors = engine.count_neighbors(grid, 0, 0)
        assert neighbors == 3

        # Edge cell (0,1) should have 5 neighbors: all except (0,0) which is itself
        neighbors = engine.count_neighbors(grid, 0, 1)
        assert neighbors == 5

        # Center cell (1,1) should have 8 neighbors
        neighbors = engine.count_neighbors(grid, 1, 1)
        assert neighbors == 8

    def test_neighbor_count_empty_grid(self):
        """Empty grid has zero neighbors everywhere."""
        engine = ConwayEngine()
        grid = Grid(5, 5)

        for y in range(5):
            for x in range(5):
                neighbors = engine.count_neighbors(grid, x, y)
                assert neighbors == 0


class TestSingleCellRules:
    """Test Conway rules for individual cells."""

    def setup_method(self):
        """Create fresh engine for each test."""
        self.engine = ConwayEngine()

    def test_live_cell_underpopulation(self):
        """Live cell with <2 neighbors dies."""
        grid = Grid(3, 3)

        # Single live cell - 0 neighbors (dies)
        grid[1, 1] = True
        next_state = self.engine.update_cell(grid, 1, 1)
        assert next_state is False

        # Live cell with 1 neighbor (dies)
        grid.clear()
        grid[1, 1] = True
        grid[1, 2] = True  # 1 neighbor
        next_state = self.engine.update_cell(grid, 1, 1)
        assert next_state is False

    def test_live_cell_survival(self):
        """Live cell with 2-3 neighbors survives."""
        grid = Grid(3, 3)

        # Live cell with 2 neighbors (survives)
        grid[1, 1] = True
        grid[0, 1] = True  # neighbor 1
        grid[1, 0] = True  # neighbor 2
        next_state = self.engine.update_cell(grid, 1, 1)
        assert next_state is True

        # Live cell with 3 neighbors (survives)
        grid.clear()
        grid[1, 1] = True
        grid[0, 1] = True  # neighbor 1
        grid[1, 0] = True  # neighbor 2
        grid[1, 2] = True  # neighbor 3
        next_state = self.engine.update_cell(grid, 1, 1)
        assert next_state is True

    def test_live_cell_overpopulation(self):
        """Live cell with >3 neighbors dies."""
        grid = Grid(3, 3)

        # Live cell with 4 neighbors (dies)
        grid[1, 1] = True
        # Set all neighbors alive
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx != 0 or dy != 0:
                    ny, nx = 1 + dy, 1 + dx
                    grid[ny, nx] = True
        next_state = self.engine.update_cell(grid, 1, 1)
        assert next_state is False

    def test_dead_cell_reproduction(self):
        """Dead cell with exactly 3 neighbors becomes alive."""
        grid = Grid(3, 3)

        # Dead cell with 3 neighbors (birth)
        grid[0, 1] = True  # neighbor 1
        grid[1, 0] = True  # neighbor 2
        grid[1, 2] = True  # neighbor 3
        # grid[1,1] stays False

        next_state = self.engine.update_cell(grid, 1, 1)
        assert next_state is True

    def test_dead_cell_other_counts(self):
        """Dead cell with !=3 neighbors stays dead."""
        grid = Grid(3, 3)

        # Dead cell with 2 neighbors (stays dead)
        grid[0, 1] = True
        grid[1, 0] = True
        next_state = self.engine.update_cell(grid, 1, 1)
        assert next_state is False

        # Dead cell with 4 neighbors (stays dead)
        grid.clear()
        grid[0, 1] = True
        grid[1, 0] = True
        grid[1, 2] = True
        grid[2, 1] = True
        next_state = self.engine.update_cell(grid, 1, 1)
        assert next_state is False

    def test_rule_table_generation(self):
        """Rule table covers all possibilities."""
        rules = self.engine.get_rule_table()

        # Should have 2 states × 9 neighbor counts = 18 rules
        assert len(rules) == 18

        # Test specific rules
        assert rules[(True, 2)] is True   # Live cell survives with 2 neighbors
        assert rules[(True, 3)] is True   # Live cell survives with 3 neighbors
        assert rules[(True, 1)] is False  # Live cell dies with 1 neighbor
        assert rules[(True, 4)] is False  # Live cell dies with 4 neighbors
        assert rules[(False, 3)] is True  # Dead cell births with 3 neighbors
        assert rules[(False, 2)] is False # Dead cell stays dead with 2 neighbors


class TestGridUpdate:
    """Test full grid update operations."""

    def setup_method(self):
        """Create fresh engine for each test."""
        self.engine = ConwayEngine()

    def test_update_grid_preserves_size(self):
        """Grid update preserves dimensions."""
        grid = Grid(16, 16)
        grid[5, 5] = True  # Add a live cell

        new_grid = self.engine.update_grid(grid)

        assert new_grid.width == 16
        assert new_grid.height == 16
        assert new_grid.state.shape == (16, 16)

    def test_update_grid_immutable_input(self):
        """Original grid is not modified by update_grid."""
        grid = Grid(3, 3)
        grid[1, 1] = True
        grid[1, 2] = True
        grid[2, 1] = True
        original_state = grid.state.copy()

        self.engine.update_grid(grid)

        # Original grid should be unchanged
        np.testing.assert_array_equal(grid.state, original_state)

    def test_step_modifies_inplace(self):
        """Step method modifies grid in-place."""
        grid = Grid(3, 3)
        grid[1, 1] = True
        grid[1, 2] = True
        grid[2, 1] = True

        # Block pattern should be stable
        self.engine.step(grid)

        # Should still be a block
        assert grid[1, 1] is True
        assert grid[1, 2] is True
        assert grid[2, 1] is True
        assert grid[2, 2] is True

    def test_step_preserves_size(self):
        """Step preserves grid dimensions."""
        grid = Grid(32, 32)
        original_width = grid.width
        original_height = grid.height

        self.engine.step(grid)

        assert grid.width == original_width
        assert grid.height == original_height


class TestClassicalPatterns:
    """Test Conway's Game of Life classical patterns."""

    def setup_method(self):
        """Create fresh engine for each test."""
        self.engine = ConwayEngine()

    def test_block_pattern(self):
        """Block (still life) pattern is stable."""
        grid = Grid(6, 6)

        # Create block pattern (2x2 square)
        grid[2, 2] = True
        grid[2, 3] = True
        grid[3, 2] = True
        grid[3, 3] = True

        # Should be stable for at least 5 generations
        for _ in range(5):
            self.engine.step(grid)

            # Block should persist
            assert grid[2, 2] is True
            assert grid[2, 3] is True
            assert grid[3, 2] is True
            assert grid[3, 3] is True

    def test_blinker_pattern(self):
        """Blinker (oscillator) pattern oscillates."""
        grid = Grid(5, 5)

        # Create vertical blinker (period 2)
        grid[1, 2] = True
        grid[2, 2] = True
        grid[3, 2] = True

        # After step 1: should become horizontal
        self.engine.step(grid)
        assert grid[1, 2] is False
        assert grid[2, 1] is True
        assert grid[2, 2] is True
        assert grid[2, 3] is True
        assert grid[3, 2] is False

        # After step 2: should become vertical again
        self.engine.step(grid)
        assert grid[1, 2] is True
        assert grid[2, 1] is False
        assert grid[2, 2] is True
        assert grid[2, 3] is False
        assert grid[3, 2] is True


class TestAll512Rules:
    """Comprehensive testing of all 512 possible rule cases."""

    def test_all_neighborhood_configurations(self):
        """Test all 512 possible (center_alive, 8_neighbors) combinations."""
        all_rules = get_all_possible_rules()

        # Should have 2 states × 256 neighborhood masks = 512 rules
        assert len(all_rules) == 512

        engine = ConwayEngine()

        # Test each rule by creating the pattern and applying rules
        for center_alive in [False, True]:
            for neighbor_mask in range(256):
                # Create 3x3 pattern for this configuration
                pattern = generate_neighborhood_pattern(center_alive, neighbor_mask)

                # Count neighbors manually to verify
                manual_count = np.sum(pattern) - int(center_alive)
                expected_neighbors = int(np.sum(pattern)) - int(center_alive)

                assert manual_count == expected_neighbors

                # Create grid and apply Conway rules
                grid = Grid(3, 3, pattern)
                next_state = engine.update_cell(grid, 1, 1)

                # Verify result matches rule table
                expected_next = all_rules[(center_alive, neighbor_mask)]
                assert next_state == expected_next

    def test_rule_table_completeness(self):
        """Verify rule table is generated correctly."""
        rules = get_all_possible_rules()

        # Test some specific known cases
        # Empty neighborhood: dead center stays dead, live center dies
        assert rules[(False, 0)] is False  # Dead center, 0 neighbors -> stays dead
        assert rules[(True, 0)] is False   # Live center, 0 neighbors -> dies

        # Three neighbors: dead center births, live center survives
        # Create mask for three adjacent neighbors (e.g., positions 0,1,2)
        three_neighbors_mask = 0b11100000  # First 3 bits set (top row)
        assert rules[(False, three_neighbors_mask)] is True   # Birth
        assert rules[(True, three_neighbors_mask)] is True    # Survive

        # Four neighbors: live center dies
        four_neighbors_mask = 0b11110000  # First 4 bits set
        assert rules[(True, four_neighbors_mask)] is False    # Dies


@pytest.mark.parametrize("grid_size", [
    (8, 8),
    (16, 16),
    (32, 32),
])
class TestPerformanceRegression:
    """Test performance doesn't regress with different grid sizes."""

    def test_grid_update_performance(self, grid_size):
        """Grid update completes in reasonable time."""
        width, height = grid_size
        grid = Grid(width, height)

        # Fill with random pattern
        grid.randomize(0.5)

        engine = ConwayEngine()

        import time
        start_time = time.time()

        # Update grid 3 times
        for _ in range(3):
            new_grid = engine.update_grid(grid)
            grid = new_grid

        elapsed = time.time() - start_time

        # Conservative performance target: <100ms per update per 1000 cells
        cells_per_second = (width * height * 3) / elapsed
        target_cells_per_second = 10000  # 10k cells/second

        assert cells_per_second > target_cells_per_second, \
            f"Performance regression: {cells_per_second:.0f} cells/sec < {target_cells_per_second}"

        print(f"Performance: {cells_per_second:.0f} cells/sec for {width}x{height} grid")
