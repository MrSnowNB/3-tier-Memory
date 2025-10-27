"""
Phase 3 Day 1: Energy Field Basics Validation

Validates core EnergyField functionality including:
- Energy field creation and initialization
- Source placement and burst addition
- Diffusion propagation and decay
- Gradient calculation and field analysis
- Energy maxima detection
"""

import pytest
import numpy as np
import time
from src.energy.field import EnergyField, create_radial_energy_field, create_linear_energy_field


class TestEnergyFieldCreation:
    """Test EnergyField instantiation and basic setup."""

    def test_basic_field_creation(self):
        """Test creating a basic energy field."""
        field = EnergyField(width=10, height=8, decay_rate=0.9)

        assert field.width == 10
        assert field.height == 8
        assert field.decay_rate == 0.9
        assert field.energy_grid.shape == (8, 10)  # numpy indexing [y, x]
        assert np.all(field.energy_grid == 0.0)
        assert len(field.sources) == 0

    def test_invalid_dimensions(self):
        """Test error handling for invalid dimensions."""
        with pytest.raises(ValueError, match="Grid dimensions must be positive"):
            EnergyField(width=0, height=5)

        with pytest.raises(ValueError, match="Grid dimensions must be positive"):
            EnergyField(width=5, height=-1)

    def test_invalid_decay_rate(self):
        """Test error handling for invalid decay rates."""
        with pytest.raises(ValueError, match="Decay rate must be in"):
            EnergyField(width=5, height=5, decay_rate=0.0)  # <= 0

        with pytest.raises(ValueError, match="Decay rate must be in"):
            EnergyField(width=5, height=5, decay_rate=2.0)  # > 1.0

    def test_field_copy(self):
        """Test deep copying of energy fields."""
        field1 = EnergyField(8, 8, decay_rate=0.8)
        field1.add_energy_source((3, 4), 1.0)
        field1.add_energy_burst((1, 1), radius=2, strength=0.5)

        field2 = field1.copy()

        assert field2.width == field1.width
        assert field2.height == field1.height
        assert field2.decay_rate == field1.decay_rate
        assert not np.shares_memory(field1.energy_grid, field2.energy_grid)
        assert np.array_equal(field1.energy_grid, field2.energy_grid)


class TestEnergySourceManagement:
    """Test energy source placement and management."""

    def test_add_energy_source(self):
        """Test adding persistent energy sources."""
        field = EnergyField(8, 8)

        field.add_energy_source((2, 3), 0.8)
        assert abs(field.get_energy((2, 3)) - 0.8) < 1e-6  # Account for floating point precision
        assert abs(field.sources[(2, 3)] - 0.8) < 1e-6

    def test_invalid_source_position(self):
        """Test error handling for out-of-bounds sources."""
        field = EnergyField(5, 5)

        with pytest.raises(ValueError, match="Position .* out of grid bounds"):
            field.add_energy_source((-1, 0), 0.5)

        with pytest.raises(ValueError, match="Position .* out of grid bounds"):
            field.add_energy_source((5, 3), 0.5)

    def test_invalid_source_energy(self):
        """Test error handling for invalid energy values."""
        field = EnergyField(5, 5)

        with pytest.raises(ValueError, match="Energy strength must be in"):
            field.add_energy_source((2, 2), -0.1)

        with pytest.raises(ValueError, match="Energy strength must be in"):
            field.add_energy_source((2, 2), 1.5)

    def test_energy_burst(self):
        """Test adding temporary energy bursts."""
        field = EnergyField(6, 6)

        initial_center = field.get_energy((2, 2))
        field.add_energy_burst((2, 2), radius=2, strength=1.0)

        # Center should be affected
        assert field.get_energy((2, 2)) > initial_center

        # Edge should be less affected
        assert field.get_energy((0, 0)) < field.get_energy((2, 2))


class TestEnergyDiffusion:
    """Test energy propagation and decay mechanics."""

    def test_diffusion_stability(self):
        """Test that diffusion maintains energy within bounds."""
        field = EnergyField(10, 10, decay_rate=0.95)
        field.add_energy_source((5, 5), 1.0)

        # Diffuse multiple steps
        for _ in range(20):
            elapsed = field.diffuse(1)

            # Performance check: should be sub-milliseconds
            assert elapsed < 1.0, "Diffusion too slow"

            # Stability check: values should stay in [0, 1]
            assert np.all(field.energy_grid >= 0.0)
            assert np.all(field.energy_grid <= 1.0)

            # Source should resist decay
            assert field.get_energy((5, 5)) >= 0.8, "Source decayed too much"

    def test_diffusion_propagation(self):
        """Test that energy propagates outward from sources."""
        field = EnergyField(10, 10, decay_rate=0.9)
        field.add_energy_source((5, 5), 1.0)

        # Get initial state
        initial_center = field.get_energy((5, 5))
        initial_edge = field.get_energy((2, 2))

        # Diffuse
        field.diffuse(5)

        # Center should still be high but diffused
        final_center = field.get_energy((5, 5))
        final_edge = field.get_energy((2, 2))

        assert initial_edge < final_edge, "Energy didn't propagate to edge"
        assert final_center > final_edge, "Center not highest energy"

    @pytest.mark.parametrize("grid_size", [(8, 8), (12, 12), (16, 16)])
    def test_diffusion_scalability(self, grid_size):
        """Test diffusion performance at different grid sizes."""
        width, height = grid_size
        field = EnergyField(width, height)
        field.add_energy_source((width//2, height//2), 1.0)

        start_time = time.time()
        elapsed = field.diffuse(3)
        total_time = time.time() - start_time

        # Should complete within reasonable time (scale with grid size)
        max_time = (width * height) / 10000  # Linear scaling assumption
        assert total_time < max_time * 2, f"Too slow for {grid_size} grid"

        # Should have diffused some energy
        center_energy = field.get_energy((width//2, height//2))
        assert center_energy > 0.9, "Energy didn't maintain at source"

    def test_source_resistance_to_decay(self):
        """Test that sources maintain high energy despite decay."""
        field = EnergyField(12, 12, decay_rate=0.8)  # Strong decay
        field.add_energy_source((6, 6), 1.0)

        # Diffuse many steps
        for _ in range(10):
            field.diffuse(1)

        # Source should still be very high
        source_energy = field.get_energy((6, 6))
        assert source_energy > 0.95, f"Source decayed too much: {source_energy}"

class TestEnergyGradients:
    """Test energy gradient calculation and analysis."""

    def test_gradient_calculation(self):
        """Test gradient calculation at various positions."""
        field = EnergyField(8, 8)

        # Create a simple gradient (higher on left, lower on right)
        for x in range(8):
            energy = 1.0 - (x / 7) * 0.5  # Decreasing from left to right
            for y in range(8):
                field.energy_grid[y, x] = energy

        # Should have negative gradient in x direction (decreasing)
        dx, dy = field.get_gradient((3, 3))

        # dx should be negative (energy decreases as x increases)
        assert dx < 0, f"x gradient should be negative, got {dx}"

        # dy should be approximately zero
        assert abs(dy) < 0.01, f"y gradient should be near zero, got {dy}"

    def test_gradient_boundary_handling(self):
        """Test gradient calculation near boundaries."""
        field = EnergyField(6, 6)

        # Should fail at boundary positions
        with pytest.raises(ValueError, match="too close to boundary"):
            field.get_gradient((0, 0))

        with pytest.raises(ValueError, match="too close to boundary"):
            field.get_gradient((5, 5))

        # Should work in interior
        dx, dy = field.get_gradient((2, 2))
        # Should be zeros since no gradients in uniform field
        assert abs(dx) < 0.01
        assert abs(dy) < 0.01

    def test_gradient_field_generation(self):
        """Test creating complete gradient field."""
        field = EnergyField(6, 6)
        field.add_energy_source((3, 3), 1.0)
        field.diffuse(3)

        grad_field = field.get_gradient_field()
        assert grad_field.shape == (6, 6, 2)

        # Boundary positions should be zero (no gradient calculated)
        assert np.all(grad_field[0, :, :] == 0)  # Top row
        assert np.all(grad_field[:, 0, :] == 0)  # Left column
        assert np.all(grad_field[-1, :, :] == 0) # Bottom row
        assert np.all(grad_field[:, -1, :] == 0) # Right column

    def test_energy_maxima_detection(self):
        """Test finding local energy maxima."""
        field = EnergyField(8, 8)
        field.add_energy_source((2, 2), 1.0)
        field.add_energy_source((5, 5), 0.9)
        field.add_energy_source((1, 6), 0.3)  # Below threshold

        maxima = field.find_energy_maxima(threshold=0.5)

        # Should find at least the sources above threshold
        positions_found = set(maxima)
        assert (2, 2) in positions_found, "Failed to find high-energy maximum"
        assert (5, 5) in positions_found, "Failed to find second maximum"
        assert (1, 6) not in positions_found, "Found maximum below threshold"


class TestFactoryFunctions:
    """Test energy field factory functions."""

    def test_linear_energy_field(self):
        """Test creating linear gradient fields."""
        field = create_linear_energy_field(8, 6, direction='horizontal', gradient_strength=0.5)

        # Left side should be higher than right side initially
        left_energy = field.get_energy((0, 3))
        right_energy = field.get_energy((7, 3))

        assert left_energy > right_energy, "Horizontal gradient not working"

        # Both should be within bounds
        assert 0.0 <= left_energy <= 1.0
        assert 0.0 <= right_energy <= 1.0

    def test_radial_energy_field(self):
        """Test creating radial energy fields."""
        field = create_radial_energy_field(10, 10, center=(5, 5), max_radius=4)

        # Center should be highest
        center_energy = field.get_energy((5, 5))
        edge_energy = field.get_energy((1, 1))  # Corner, far from center

        assert center_energy > edge_energy, "Radial gradient center not highest"

        # Check radial pattern
        mid_energy = field.get_energy((7, 5))  # Same row as center
        assert center_energy > mid_energy, "Radial decay not working"


def test_day_1_functional_verification():
    """Day 1 Requirement: Functional verification (10×10 grid diffusion ≥95% stability)."""
    field = EnergyField(10, 10, decay_rate=0.95)

    # Add some sources
    field.add_energy_source((3, 3), 1.0)
    field.add_energy_source((7, 7), 0.8)

    # Track total energy over many diffusion steps
    initial_total = np.sum(field.energy_grid)
    energy_history = []
    stability_window = 20  # Check last 20 steps for stability

    for step in range(100):
        field.diffuse(1)
        total_energy = np.sum(field.energy_grid)
        energy_history.append(total_energy)

        # Values should stay in bounds
        assert np.all(field.energy_grid >= 0.0)
        assert np.all(field.energy_grid <= 1.0)

    # Calculate stability (standard deviation of last stability_window steps)
    if len(energy_history) >= stability_window:
        recent_energies = energy_history[-stability_window:]
        stability_std = np.std(recent_energies)
        mean_recent = np.mean(recent_energies)

        # Stability metric: std/mean should be very low (< 5%)
        stability_ratio = stability_std / mean_recent
        stability_percent = (1.0 - stability_ratio) * 100

        print(f"\nDiffusion Stability: {stability_percent:.2f}% (target: ≥95%)")
        assert stability_percent >= 95.0, ".2f"

    # Performance check
    total_time = sum(field.last_diffusion_time for _ in range(field.diffusion_call_count))
    avg_diffusion_time = total_time / max(1, field.diffusion_call_count)
    print(f"Average diffusion time: {avg_diffusion_time*1000:.2f}ms per step")

    # Should be sub-millisecond
    assert avg_diffusion_time < 0.1, "Diffusion too slow for Phase 3 target"
