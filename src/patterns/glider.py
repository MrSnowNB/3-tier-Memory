"""Glider pattern definitions and canonical forms for Conway's Game of Life.

Provides standardized glider patterns, rotations, and classification
for the CyberMesh CA substrate. Gliders are the fundamental mobile
patterns that will carry shard data in the 3-tier architecture.
"""

import numpy as np
from typing import Dict, List, Tuple, Set
from enum import Enum


class GliderOrientation(Enum):
    """Standard glider orientations referenced from initial northwest movement."""
    NORTHWEST = 0  # Points northwest (default starting orientation)
    NORTHEAST = 1  # Rotated 90° clockwise
    SOUTHEAST = 2  # Rotated 180°
    SOUTHWEST = 3  # Rotated 270°


class GliderPhase(Enum):
    """Glider phases in its 4-step movement cycle."""
    PHASE_0 = 0  # Initial position
    PHASE_1 = 1  # After 1 step northwest
    PHASE_2 = 2  # After 2 steps northwest
    PHASE_3 = 3  # After 3 steps northwest


# Canonical glider pattern for each orientation and phase
# These are the standard 5-cell glider configurations
GLIDER_PATTERNS: Dict[Tuple[GliderOrientation, GliderPhase], np.ndarray] = {

    # NORTHWEST orientation (canonical starting position)
    (GliderOrientation.NORTHWEST, GliderPhase.PHASE_0): np.array([
        [False, True, False],
        [False, False, True],
        [True, True, True]
    ], dtype=bool),

    (GliderOrientation.NORTHWEST, GliderPhase.PHASE_1): np.array([
        [True, False, True],
        [False, True, True],
        [False, True, False]
    ], dtype=bool),

    (GliderOrientation.NORTHWEST, GliderPhase.PHASE_2): np.array([
        [False, False, True],
        [True, False, True],
        [False, True, True]
    ], dtype=bool),

    (GliderOrientation.NORTHWEST, GliderPhase.PHASE_3): np.array([
        [True, False, False],
        [False, True, False],
        [True, True, True]
    ], dtype=bool),

    # NORTHEAST orientation (90° clockwise rotation)
    (GliderOrientation.NORTHEAST, GliderPhase.PHASE_0): np.array([
        [True, True, True],
        [True, False, False],
        [False, True, False]
    ], dtype=bool),

    (GliderOrientation.NORTHEAST, GliderPhase.PHASE_1): np.array([
        [False, True, False],
        [False, False, True],
        [True, True, True]
    ], dtype=bool),

    (GliderOrientation.NORTHEAST, GliderPhase.PHASE_2): np.array([
        [True, False, False],
        [True, True, False],
        [False, True, True]
    ], dtype=bool),

    (GliderOrientation.NORTHEAST, GliderPhase.PHASE_3): np.array([
        [True, True, True],
        [False, False, True],
        [False, True, False]
    ], dtype=bool),

    # SOUTHEAST orientation (180° rotation)
    (GliderOrientation.SOUTHEAST, GliderPhase.PHASE_0): np.array([
        [False, True, False],
        [True, False, False],
        [True, True, True]
    ], dtype=bool),

    (GliderOrientation.SOUTHEAST, GliderPhase.PHASE_1): np.array([
        [True, False, False],
        [False, True, False],
        [True, True, True]
    ], dtype=bool),

    (GliderOrientation.SOUTHEAST, GliderPhase.PHASE_2): np.array([
        [True, False, False],
        [True, True, False],
        [False, True, True]
    ], dtype=bool),

    (GliderOrientation.SOUTHEAST, GliderPhase.PHASE_3): np.array([
        [False, True, False],
        [False, False, True],
        [True, True, True]
    ], dtype=bool),

    # SOUTHWEST orientation (270° rotation)
    (GliderOrientation.SOUTHWEST, GliderPhase.PHASE_0): np.array([
        [False, True, False],
        [False, False, True],
        [True, True, True]
    ], dtype=bool),

    (GliderOrientation.SOUTHWEST, GliderPhase.PHASE_1): np.array([
        [True, False, False],
        [True, True, False],
        [False, True, True]
    ], dtype=bool),

    (GliderOrientation.SOUTHWEST, GliderPhase.PHASE_2): np.array([
        [False, True, False],
        [True, False, False],
        [True, True, True]
    ], dtype=bool),

    (GliderOrientation.SOUTHWEST, GliderPhase.PHASE_3): np.array([
        [True, True, True],
        [False, False, True],
        [False, True, False]
    ], dtype=bool),
}


def get_glider_pattern(orientation: GliderOrientation,
                       phase: GliderPhase) -> np.ndarray:
    """Get the canonical 3x3 glider pattern for given orientation and phase.

    Args:
        orientation: Direction the glider is pointing/moving toward
        phase: Current phase in the glider's movement cycle (0-3)

    Returns:
        3x3 boolean numpy array representing the glider pattern
    """
    return GLIDER_PATTERNS[(orientation, phase)].copy()


def get_all_glider_patterns() -> List[np.ndarray]:
    """Get list of all unique glider patterns across orientations and phases."""
    patterns = []
    seen = set()

    for (orientation, phase), pattern in GLIDER_PATTERNS.items():
        # Convert to bytes for hashing
        pattern_bytes = pattern.tobytes()
        if pattern_bytes not in seen:
            seen.add(pattern_bytes)
            patterns.append(pattern.copy())

    return patterns


def rotate_pattern(pattern: np.ndarray, clockwise_rotations: int = 1) -> np.ndarray:
    """Rotate a 3x3 pattern clockwise by specified number of 90° rotations.

    Args:
        pattern: 3x3 boolean array to rotate
        clockwise_rotations: Number of 90° clockwise rotations (1-3)

    Returns:
        Rotated 3x3 boolean array
    """
    rotated = pattern.copy()
    for _ in range(clockwise_rotations % 4):
        rotated = np.rot90(rotated, k=-1)  # k=-1 for clockwise rotation

    return rotated


def reflect_pattern(pattern: np.ndarray, axis: str = 'horizontal') -> np.ndarray:
    """Reflect a 3x3 pattern across specified axis.

    Args:
        pattern: 3x3 boolean array to reflect
        axis: 'horizontal', 'vertical', or 'diagonal'

    Returns:
        Reflected 3x3 boolean array
    """
    if axis == 'horizontal':
        return np.flipud(pattern)  # Flip upside down
    elif axis == 'vertical':
        return np.fliplr(pattern)  # Flip left-right
    elif axis == 'diagonal':
        # Diagonal flip (transpose)
        return pattern.T
    else:
        raise ValueError(f"Unsupported axis: {axis}")


def pattern_similarity(pattern1: np.ndarray, pattern2: np.ndarray) -> float:
    """Calculate similarity between two patterns (0.0 = identical, 1.0 = completely different).

    Args:
        pattern1: First 3x3 boolean pattern
        pattern2: Second 3x3 boolean pattern

    Returns:
        Similarity score (0.0 = identical, 1.0 = completely different)
    """
    if pattern1.shape != pattern2.shape:
        return 1.0

    # Hamming distance normalized by total cells
    differences = np.sum(pattern1 != pattern2)
    return differences / pattern1.size


def find_glider_centroid(pattern: np.ndarray) -> Tuple[float, float]:
    """Find the center of mass (centroid) of a glider pattern.

    Args:
        pattern: 3x3 boolean glider pattern

    Returns:
        (x, y) coordinates of pattern centroid (0.5 = dead center)
    """
    if pattern.size == 0 or not np.any(pattern):
        return (0.0, 0.0)

    # Get coordinates of alive cells
    y_coords, x_coords = np.where(pattern)

    # Calculate centroid
    centroid_y = float(np.mean(y_coords))
    centroid_x = float(np.mean(x_coords))

    return (centroid_x, centroid_y)


def classify_glider_orientation(pattern: np.ndarray) -> GliderOrientation:
    """Classify the orientation of a glider pattern by comparing to canonical forms.

    Args:
        pattern: 3x3 boolean glider pattern

    Returns:
        Most likely GliderOrientation, or NORTHWEST if ambiguous
    """
    min_similarity = float('inf')
    best_orientation = GliderOrientation.NORTHWEST

    # Compare against phase 0 of each orientation (canonical form)
    for orientation in GliderOrientation:
        canonical = get_glider_pattern(orientation, GliderPhase.PHASE_0)
        similarity = pattern_similarity(pattern, canonical)

        if similarity < min_similarity:
            min_similarity = similarity
            best_orientation = orientation

    return best_orientation


def is_valid_glider_pattern(pattern: np.ndarray) -> bool:
    """Check if a 3x3 pattern could be a valid glider configuration.

    Args:
        pattern: 3x3 boolean array to validate

    Returns:
        True if pattern matches a known glider configuration
    """
    if pattern.shape != (3, 3):
        return False

    # Count alive cells (gliders have exactly 5)
    alive_count = np.sum(pattern)
    if alive_count != 5:
        return False

    # Check against all known glider patterns
    for known_pattern in get_all_glider_patterns():
        if np.array_equal(pattern, known_pattern):
            return True

    return False


# Movement vectors for each orientation (steps per phase)
GLIDER_MOVEMENT: Dict[GliderOrientation, Tuple[int, int]] = {
    GliderOrientation.NORTHWEST: (-1, -1),  # Up-left
    GliderOrientation.NORTHEAST: (1, -1),   # Up-right
    GliderOrientation.SOUTHEAST: (1, 1),    # Down-right
    GliderOrientation.SOUTHWEST: (-1, 1),   # Down-left
}


def predict_glider_position(initial_x: int, initial_y: int,
                           orientation: GliderOrientation,
                           steps: int) -> Tuple[int, int]:
    """Predict glider position after N steps.

    Args:
        initial_x: Starting X coordinate
        initial_y: Starting Y coordinate
        orientation: Glider movement direction
        steps: Number of steps to simulate

    Returns:
        (x, y) predicted position after steps
    """
    dx, dy = GLIDER_MOVEMENT[orientation]

    # Glider moves 1 step every 4 phases, but we can approximate
    # For prediction purposes, assume continuous movement
    total_dx = dx * (steps // 4)
    total_dy = dy * (steps // 4)

    # Account for fractional movement in current phase cycle
    phase_offset = steps % 4
    if phase_offset > 0:
        # Partial movement based on current phase
        partial_dx = dx * (phase_offset / 4.0)
        partial_dy = dy * (phase_offset / 4.0)
        total_dx += partial_dx
        total_dy += partial_dy

    return (initial_x + int(round(total_dx)), initial_y + int(round(total_dy)))
