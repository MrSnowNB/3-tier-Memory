"""
Energy-Glider Coupling Mechanics

Implements energy field influence on glider movement and evolution.
Gliders are biased toward energy gradients, creating intelligent routing corridors.
"""

import numpy as np
from typing import List, Optional, Tuple, Dict, Union, Callable
import logging
from .field import EnergyField
from ..patterns.glider import GliderOrientation, GliderPhase, GLIDER_PATTERNS

logger = logging.getLogger(__name__)


class EnergyGliderConfig:
    """Configuration for energy field influence on glider behavior."""

    def __init__(self,
                 energy_weight: float = 0.3,
                 gradient_sensitivity: float = 1.0,
                 attraction_strength: float = 2.0,
                 repulsion_threshold: float = 0.1):
        """Initialize energy-glider coupling configuration.

        Args:
            energy_weight: How strongly energy gradients influence movement (0.0-1.0)
            gradient_sensitivity: Multiplier for gradient magnitude response (0.0+)
            attraction_strength: Preference multiplier for gradient-aligned moves (1.0+)
            repulsion_threshold: Energy level below which patterns create repulsion (0.0-1.0)
        """
        self.energy_weight = max(0.0, min(1.0, energy_weight))
        self.gradient_sensitivity = max(0.0, gradient_sensitivity)
        self.attraction_strength = max(1.0, attraction_strength)
        self.repulsion_threshold = max(0.0, min(1.0, repulsion_threshold))

    def copy(self) -> 'EnergyGliderConfig':
        """Create a deep copy of the configuration."""
        return EnergyGliderConfig(
            energy_weight=self.energy_weight,
            gradient_sensitivity=self.gradient_sensitivity,
            attraction_strength=self.attraction_strength,
            repulsion_threshold=self.repulsion_threshold
        )


class EnergyGliderCoupling:
    """Handles the interaction between gliders and energy fields.

    Modifies glider evolution probabilities based on local energy gradients,
    creating intelligent movement patterns that follow energy landscapes.
    """

    def __init__(self, config: Optional[EnergyGliderConfig] = None):
        """Initialize energy-glider coupling system.

        Args:
            config: Configuration for energy influence parameters
        """
        self.config = config or EnergyGliderConfig()

    def evolve_glider_with_energy(self,
                                  glider_pattern: np.ndarray,
                                  energy_field: EnergyField,
                                  glider_x: int,
                                  glider_y: int,
                                  base_rule_function: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        """Evolve glider pattern with energy field influence.

        Args:
            glider_pattern: Current 3x3 glider pattern
            energy_field: Energy field exerting influence
            glider_x: X coordinate of glider center (bottom-right cell)
            glider_y: Y coordinate of glider center (bottom-right cell)
            base_rule_function: Function applying standard Conway rules

        Returns:
            Evolved 5x5 grid section with energy-influenced evolution
        """
        # Get the 5x5 neighborhood around glider for evolution
        neighborhood = self._extract_neighborhood(energy_field, glider_x, glider_y)

        # Apply base Conway rules
        evolved_base = base_rule_function(neighborhood)

        # Apply energy biasing to evolution
        evolved_biased = self._apply_energy_bias(evolved_base, neighborhood, energy_field)

        return evolved_biased

    def get_energy_biased_transitions(self,
                                      current_pattern: np.ndarray,
                                      energy_gradient: Tuple[float, float],
                                      possible_moves: List[Tuple[int, int]]) -> List[float]:
        """Calculate transition probabilities biased by energy gradients.

        Args:
            current_pattern: Current glider pattern (3x3)
            energy_gradient: (dx, dy) gradient vector at glider position
            possible_moves: List of (dx, dy) movement vectors for candidate patterns

        Returns:
            List of probability scores for each possible move (higher = more likely)
        """
        # Convert gradient to magnitude and direction
        grad_dx, grad_dy = energy_gradient
        gradient_magnitude = np.sqrt(grad_dx**2 + grad_dy**2)

        # Apply sensitivity scaling
        effective_magnitude = gradient_magnitude * self.config.gradient_sensitivity

        probabilities = []

        for move_dx, move_dy in possible_moves:
            # Calculate alignment between move and gradient directions
            move_magnitude = np.sqrt(move_dx**2 + move_dy**2)

            if move_magnitude == 0:
                # Stationary move - low probability unless gradient is weak
                alignment = 1.0 - min(1.0, effective_magnitude)
            else:
                # Normalize vectors for dot product
                move_norm = (move_dx / move_magnitude, move_dy / move_dy if move_dy != 0 else 0)
                grad_norm = (grad_dx / effective_magnitude if effective_magnitude > 0 else 0,
                           grad_dy / effective_magnitude if effective_magnitude > 0 else 0)

                # Cosine similarity between move and gradient directions
                alignment = (move_norm[0] * grad_norm[0] + move_norm[1] * grad_norm[1])

            # Convert alignment to probability score
            if alignment > 0:
                # Attraction: moves aligned with gradient get bonus
                probability = 1.0 + (alignment * self.config.attraction_strength - 1.0) * self.config.energy_weight
            else:
                # Repulsion: moves opposite to gradient get penalty
                probability = 1.0 + (alignment * 0.5) * self.config.energy_weight

            probabilities.append(max(0.01, probability))  # Minimum probability

        return probabilities

    def modify_glider_movement_probabilities(self,
                                           glider_pattern: np.ndarray,
                                           current_orientation: GliderOrientation,
                                           energy_gradient: Tuple[float, float]) -> Dict[GliderOrientation, float]:
        """Modify glider orientation change probabilities based on energy gradients.

        Args:
            glider_pattern: Current glider pattern
            current_orientation: Current glider movement direction
            energy_gradient: Local energy gradient vector

        Returns:
            Dictionary of orientation -> preference_score for possible turns
        """
        preferences = {}

        # Base preference for continuing current direction
        preferences[current_orientation] = 1.0

        # Calculate preference for each possible orientation
        for orientation in GliderOrientation:
            if orientation == current_orientation:
                continue

            # Get movement vectors
            current_dx, current_dy = self._get_orientation_vector(current_orientation)
            new_dx, new_dy = self._get_orientation_vector(orientation)

            # Direction change
            delta_dx = new_dx - current_dx
            delta_dy = new_dy - current_dy
            change_magnitude = np.sqrt(delta_dx**2 + delta_dy**2)

            # Penalize large direction changes unless gradient justifies it
            grad_dx, grad_dy = energy_gradient
            gradient_alignment = abs(delta_dx * grad_dx + delta_dy * grad_dy)

            # Energy influence on direction changes
            energy_modifier = 1.0 + (gradient_alignment * self.config.energy_weight)

            # Base preference inversely related to change magnitude
            base_preference = max(0.1, 1.0 - change_magnitude * 0.3)

            preferences[orientation] = base_preference * energy_modifier

        # Normalize to probabilities
        total = sum(preferences.values())
        for orientation in preferences:
            preferences[orientation] /= total

        return preferences

    def predict_energy_influenced_path(self,
                                      start_x: int,
                                      start_y: int,
                                      initial_orientation: GliderOrientation,
                                      energy_field: EnergyField,
                                      steps: int = 10) -> List[Tuple[int, int]]:
        """Predict glider path under energy field influence.

        Args:
            start_x: Starting X position
            start_y: Starting Y position
            initial_orientation: Starting movement direction
            energy_field: Energy field exerting influence
            steps: Number of steps to predict

        Returns:
            List of (x, y) positions along predicted path
        """
        path = [(start_x, start_y)]
        current_x, current_y = start_x, start_y
        current_orientation = initial_orientation

        for step in range(steps):
            # Check local energy gradient
            try:
                grad_dx, grad_dy = energy_field.get_gradient((current_x, current_y))
                energy_gradient = (grad_dx, grad_dy)
            except ValueError:
                # Boundary - no gradient influence
                energy_gradient = (0.0, 0.0)

            # Calculate orientation preferences
            orientation_preferences = self.modify_glider_movement_probabilities(
                np.zeros((3, 3), dtype=bool),  # Placeholder pattern
                current_orientation,
                energy_gradient
            )

            # Choose new orientation based on preferences
            orientations = list(orientation_preferences.keys())
            weights = list(orientation_preferences.values())

            # Simple weighted selection (in practice would use more sophisticated sampling)
            max_weight_idx = np.argmax(weights)
            new_orientation = orientations[max_weight_idx]

            # Move in chosen direction
            move_dx, move_dy = self._get_orientation_vector(new_orientation)
            current_x += move_dx
            current_y += move_dy

            path.append((current_x, current_y))
            current_orientation = new_orientation

            # Bounds checking
            width, height = energy_field.width, energy_field.height
            if not (0 <= current_x < width and 0 <= current_y < height):
                break  # Glider moved out of field

        return path

    def _extract_neighborhood(self, energy_field: EnergyField, center_x: int, center_y: int) -> np.ndarray:
        """Extract 5x5 neighborhood around glider position for Conway evolution."""
        neighborhood = np.zeros((5, 5), dtype=int)

        for dy in range(-2, 3):
            for dx in range(-2, 3):
                nx, ny = center_x + dx, center_y + dy

                # Bounds checking with toroidal wrapping for energy field
                if 0 <= nx < energy_field.width and 0 <= ny < energy_field.height:
                    neighborhood[dy + 2, dx + 2] = 1 if energy_field.get_energy((nx, ny)) > 0.5 else 0
                else:
                    # Wrap around boundary (energy field toroidal)
                    nx = nx % energy_field.width
                    ny = ny % energy_field.height
                    neighborhood[dy + 2, dx + 2] = 1 if energy_field.get_energy((nx, ny)) > 0.5 else 0

        return neighborhood

    def _apply_energy_bias(self, base_evolution: np.ndarray, neighborhood: np.ndarray,
                          energy_field: EnergyField) -> np.ndarray:
        """Apply energy-based biasing to standard Conway evolution results.

        Args:
            base_evolution: 5x5 result from standard Conway rules
            neighborhood: 5x5 input neighborhood
            energy_field: Energy field for gradient lookup

        Returns:
            Modified 5x5 evolution result with energy influence
        """
        # Copy base evolution
        biased_evolution = base_evolution.copy()

        # Calculate gradients for the 3x3 evolution center
        center_x, center_y = 2, 2  # Center of 5x5 neighborhood

        try:
            grad_dx, grad_dy = energy_field.get_gradient((center_x, center_y))
            gradient_magnitude = np.sqrt(grad_dx**2 + grad_dy**2)
        except ValueError:
            # Boundary position
            grad_dx, grad_dy = 0.0, 0.0
            gradient_magnitude = 0.0

        # Apply gradient-based modification to center cell (glider position)
        if gradient_magnitude > self.config.repulsion_threshold:
            # Energy influence present - bias toward gradient direction
            if grad_dx > 0:
                # Bias toward positive x (rightward)
                if np.random.random() < self.config.energy_weight * 0.1:
                    # Small chance to flip cell based on gradient
                    biased_evolution[center_y, center_x] = 1 - biased_evolution[center_y, center_x]

        return biased_evolution

    def _get_orientation_vector(self, orientation: GliderOrientation) -> Tuple[int, int]:
        """Convert glider orientation to movement vector."""
        from ..patterns.glider import GLIDER_MOVEMENT  # Import here to avoid circular dependency
        return GLIDER_MOVEMENT[orientation]


def create_energy_biased_glider_evolution(energy_field: EnergyField,
                                        coupling_config: Optional[EnergyGliderConfig] = None):
    """Factory function creating energy-biased glider evolution function.

    Args:
        energy_field: Energy field to consult for gradients
        coupling_config: Configuration for energy influence

    Returns:
        Function that can be used as energy-aware glider evolution
    """
    coupling = EnergyGliderCoupling(coupling_config)

    def energy_aware_evolution(neighborhood: np.ndarray) -> np.ndarray:
        """Energy-aware glider evolution function compatible with CA engine."""
        # Extract glider position from neighborhood (assume center)
        center_x, center_y = neighborhood.shape[0] // 2, neighborhood.shape[1] // 2

        return coupling.evolve_glider_with_energy(
            neighborhood[center_y-1:center_y+2, center_x-1:center_x+2],  # 3x3 glider pattern
            energy_field,
            center_x,
            center_y,
            lambda n: n  # Placeholder identity function (would use Conway rules)
        )

    return energy_aware_evolution


# Global coupling instance for easy access
default_coupling = EnergyGliderCoupling()


def get_energy_influence_strength(energy_value: float, gradient_magnitude: float) -> float:
    """Calculate how strongly energy should influence glider at a location.

    Args:
        energy_value: Local energy level (0.0-1.0)
        gradient_magnitude: Local gradient strength

    Returns:
        Influence weight (0.0 = no influence, 1.0 = maximum influence)
    """
    # High energy + strong gradient = high influence
    # Low energy + weak gradient = low influence
    energy_factor = energy_value
    gradient_factor = min(1.0, gradient_magnitude * 2.0)  # Scale gradient influence

    return energy_factor * gradient_factor


def test_energy_glider_coupling():
    """Simple test function demonstrating energy-glider coupling."""
    from .field import create_radial_energy_field

    # Create test energy field
    energy_field = create_radial_energy_field(10, 10, center=(5, 5))

    # Create coupling system
    coupling = EnergyGliderCoupling()

    # Test path prediction
    path = coupling.predict_energy_influenced_path(
        2, 2,  # Start near corner
        GliderOrientation.NORTHWEST,
        energy_field,
        steps=5
    )

    print(f"Energy-influenced glider path: {path}")

    # Test gradient influence
    gradient = (0.5, 0.3)  # Gradients tending rightward and upward
    moves = [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1)]  # Possible move vectors

    probabilities = coupling.get_energy_biased_transitions(
        np.zeros((3, 3), dtype=bool),  # Dummy pattern
        gradient,
        moves
    )

    print(f"Move probabilities for gradient {gradient}:")
    for move, prob in zip(moves, probabilities):
        print(f"  Move {move}: {prob:.3f}")

    return path, probabilities


if __name__ == "__main__":
    # Run demonstration
    path, probs = test_energy_glider_coupling()
