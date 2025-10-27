"""
Phase 3 Emergency Reset: Conway Baseline Validation

Tests pure Conway's Game of Life behavior to establish baseline
emergence before adding any environmental modifications.
"""

import pytest
import numpy as np
from src.core.ca_grid import CAGrid, create_glider_pattern, create_blinker_pattern, create_block_pattern
from src.core.conway_rules import ConwayRuleParams


class TestConwayBaseline:
    """Test fundamental Conway behaviors to ensure correct implementation."""

    def test_block_stable_still_life(self):
        """Test that 2x2 block remains stable (still life)."""
        grid = CAGrid(6, 6)

        # Place 2x2 block in center
        block_pattern = create_block_pattern()  # 2x2 block
        grid.load_pattern(block_pattern, 2, 2)

        # Block should be stable across many generations
        initial_live_count = grid.get_live_count()
        assert initial_live_count == 4, "Initial block should have 4 live cells"

        # Test stability over 20 generations
        for generation in range(20):
            live_count = grid.step()
            assert live_count == 4, f"Block unstable at generation {generation}"

        # Center of mass should not move (block is stationary)
        initial_com = grid.get_center_of_mass()
        final_com = grid.get_center_of_mass()
        assert abs(initial_com[0] - final_com[0]) < 0.1
        assert abs(initial_com[1] - final_com[1]) < 0.1

    def test_blinker_oscillates_period_2(self):
        """Test that horizontal blinker oscillates with period 2."""
        grid = CAGrid(8, 8)

        # Place horizontal blinker in center
        blinker_pattern = create_blinker_pattern()  # 1x3 horizontal
        grid.load_pattern(blinker_pattern, 3, 4)

        # Record initial state (horizontal: 3 live cells)
        initial_pattern = grid.to_array()[3:5, 2:6]  # 2x4 region around blinker
        initial_live_count = grid.get_live_count()
        assert initial_live_count == 3, "Initial blinker should have 3 live cells"

        # Step 1: Should become vertical (still 3 live cells)
        grid.step()
        live_count_1 = grid.get_live_count()
        assert live_count_1 == 3, "Blinker should maintain 3 live cells after step 1"

        # Step 2: Should become horizontal again (back to original)
        grid.step()
        live_count_2 = grid.get_live_count()
        assert live_count_2 == 3, "Blinker should maintain 3 live cells after step 2"

        # Check it returned to original state (period 2)
        final_pattern = grid.to_array()[3:5, 2:6]
        pattern_unchanged = np.array_equal(initial_pattern, final_pattern)
        assert pattern_unchanged, "Blinker should return to original state after 2 steps"

    @pytest.mark.parametrize("grid_size", [(20, 20), (30, 30), (40, 40)])
    def test_glider_moves_diagonally_min_3_cells_in_30_steps(self, grid_size):
        """Test that glider moves diagonally at least 3 cells within 30 steps."""
        width, height = grid_size
        grid = CAGrid(width, height)

        # Place glider near corner (far from boundaries to test free movement)
        glider_pattern, center_x, center_y = create_glider_pattern(2, 2)
        grid.load_pattern(glider_pattern, 2, 2)

        # Record initial center of mass
        initial_com = grid.get_center_of_mass()

        # Track COM over 30 steps
        com_positions = []
        for step in range(30):
            grid.step()
            com_positions.append(grid.get_center_of_mass())

        # Calculate net movement
        final_com = com_positions[-1]
        delta_x = final_com[0] - initial_com[0]
        delta_y = final_com[1] - initial_com[1]

        # Diagonal movement: should move in both x and y directions
        assert abs(delta_x) >= 3.0, f"Insufficient x movement: {delta_x}"
        assert abs(delta_y) >= 3.0, f"Insufficient y movement: {delta_y}"

        # Should move roughly equally in both directions (diagonal)
        movement_ratio = abs(delta_x / delta_y if delta_y != 0 else float('inf'))
        assert 0.8 <= movement_ratio <= 1.25, f"Non-diagonal movement ratio: {movement_ratio}"

        # Live cell count should remain constant (glider conserves mass)
        live_counts = [grid.get_live_count()]
        grid_copy = grid.copy()

        for _ in range(30):
            live_count = grid_copy.step()   # Use copy because we already stepped the original
            assert live_count == 5, "Glider should maintain 5 live cells"

        # Position should be different from start
        distance_moved = np.sqrt(delta_x**2 + delta_y**2)
        assert distance_moved >= 4.0, f"Glider didn't move far enough: {distance_moved}"

    def test_glider_movement_consistency(self):
        """Test that glider movement is consistent and predictable."""
        # Test the same glider evolution multiple times
        grid1 = CAGrid(25, 25)
        grid2 = CAGrid(25, 25)

        # Place identical gliders in both grids
        glider_pattern, _, _ = create_glider_pattern(5, 5)
        grid1.load_pattern(glider_pattern, 5, 5)
        grid2.load_pattern(glider_pattern, 5, 5)

        # Evolve both identically
        for step in range(10):
            grid1.step()
            grid2.step()

            # States should remain identical
            assert np.array_equal(grid1.grid, grid2.grid), f"Grids diverged at step {step}"

            # Live count should be consistent
            assert grid1.get_live_count() == grid2.get_live_count() == 5

    def test_empty_grid_stays_empty(self):
        """Test that empty grid remains empty."""
        grid = CAGrid(10, 10)

        # Empty grid should stay empty
        for step in range(10):
            live_count = grid.step()
            assert live_count == 0, f"Empty grid became populated at step {step}"

    def test_full_grid_birth_death_cycles(self):
        """Test that densely populated grids follow Conway dynamics."""
        grid = CAGrid(12, 12)
        grid.randomize(density=0.5)  # Half cells alive

        # Track live cell evolution over several steps
        live_counts = []
        for step in range(15):
            live_count = grid.step()
            live_counts.append(live_count)

        # Live counts should vary but not go to zero immediately (emergent behavior)
        # Conway rules produce complex dynamics, not instant death
        max_live = max(live_counts)
        min_live = min(live_counts)

        # Should have some variation in live cell counts
        assert max_live > min_live, "Grid evolution too static"

        # Should not immediately go to all-alive or all-dead states
        assert max_live < grid.width * grid.height, "Grid didn't stabilize"
        assert min_live > 0, "Grid died out too quickly"


class TestGridOperations:
    """Test basic grid operations are correct."""

    def test_pattern_loading_and_wrapping(self):
        """Test that patterns load correctly with toroidal wrapping."""
        grid = CAGrid(6, 6)

        # Load glider at top-left (should wrap around)
        glider_pattern, _, _ = create_glider_pattern(0, 0)
        grid.load_pattern(glider_pattern, 4, 4)  # Near bottom-right

        # Should have 5 live cells even with wrapping
        assert grid.get_live_count() == 5, "Glider pattern not loaded correctly"

    def test_center_of_mass_calculation(self):
        """Test center of mass calculations."""
        grid = CAGrid(8, 8)

        # Test empty grid
        com_empty = grid.get_center_of_mass()
        assert com_empty == (0.0, 0.0), "Empty grid should have zero COM"

        # Test single cell
        grid.set_cell(3, 4, True)
        com_single = grid.get_center_of_mass()
        assert abs(com_single[0] - 3.0) < 0.01 and abs(com_single[1] - 4.0) < 0.01

        # Test multiple cells
        grid.set_cell(5, 6, True)
        com_double = grid.get_center_of_mass()
        expected_x = (3 + 5) / 2.0
        expected_y = (4 + 6) / 2.0
        assert abs(com_double[0] - expected_x) < 0.01
        assert abs(com_double[1] - expected_y) < 0.01


def test_conway_rule_implementation():
    """Integration test ensuring Conway rules are implemented correctly."""
    # Test birth: dead cell with 3 neighbors born
    assert ConwayRuleParams.standard().update_cell(False, 3) == True
    assert ConwayRuleParams.standard().update_cell(False, 2) == False
    assert ConwayRuleParams.standard().update_cell(False, 4) == False

    # Test survival: live cell survives with 2-3 neighbors
    assert ConwayRuleParams.standard().update_cell(True, 2) == True
    assert ConwayRuleParams.standard().update_cell(True, 3) == True
    assert ConwayRuleParams.standard().update_cell(True, 1) == False
    assert ConwayRuleParams.standard().update_cell(True, 4) == False

    # Test underpopulation: live cell dies with < 2 neighbors
    assert ConwayRuleParams.standard().update_cell(True, 0) == False
    assert ConwayRuleParams.standard().update_cell(True, 1) == False

    # Test overcrowding: live cell dies with > 3 neighbors
    assert ConwayRuleParams.standard().update_cell(True, 4) == False
    assert ConwayRuleParams.standard().update_cell(True, 8) == False
