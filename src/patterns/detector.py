"""Glider detection and tracking algorithms for Conway's Game of Life.

Implements algorithms to detect, track, and predict glider movement
in cellular automaton grids. Uses template matching and motion analysis
to identify gliders with high accuracy and low false positive rates.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, NamedTuple
from dataclasses import dataclass
from .glider import (
    GliderOrientation, GliderPhase, get_glider_pattern,
    GLIDER_PATTERNS, pattern_similarity, find_glider_centroid,
    classify_glider_orientation, is_valid_glider_pattern
)
from ..core.grid import Grid
import logging

logger = logging.getLogger(__name__)


@dataclass
class GliderDetection:
    """Represents a detected glider in the grid."""
    x: int              # Center X coordinate
    y: int              # Center Y coordinate
    orientation: GliderOrientation
    phase: GliderPhase
    confidence: float   # Detection confidence (0.0-1.0)
    pattern_match: np.ndarray  # The 3x3 detected pattern
    frame_idx: int = 0  # Frame index in sequence (for tracking)


@dataclass
class GliderTrack:
    """Represents a tracked glider movement over time."""
    glider_id: int
    detections: List[GliderDetection]
    predicted_path: List[Tuple[int, int]]


class GliderDetector:
    """Advanced glider detection using template matching and spatial reasoning."""

    def __init__(self, confidence_threshold: float = 0.8, max_distance: float = 2.5):
        """Initialize glider detector.

        Args:
            confidence_threshold: Minimum confidence for detection (0.0-1.0)
            max_distance: Maximum distance for track association (cells)
        """
        self.confidence_threshold = confidence_threshold
        self.max_distance = max_distance

        # Pre-compute all glider templates for fast matching
        self.templates = self._build_template_cache()

    def _build_template_cache(self) -> Dict[Tuple[GliderOrientation, GliderPhase], np.ndarray]:
        """Build cache of all glider templates for fast access."""
        return GLIDER_PATTERNS.copy()

    def detect_gliders(self, grid: Grid, region: Optional[Tuple[int, int, int, int]] = None) -> List[GliderDetection]:
        """Detect all gliders in the grid or specified region.

        Args:
            grid: The grid to search for gliders
            region: Optional (min_x, min_y, max_x, max_y) search region

        Returns:
            List of GliderDetection objects, sorted by confidence (highest first)
        """
        detections = []

        # Define search bounds
        if region is None:
            min_x, min_y, max_x, max_y = 0, 0, grid.width - 1, grid.height - 1
        else:
            min_x, min_y, max_x, max_y = region

        # Slide 3x3 window over search area
        for y in range(max(0, min_y), min(max_y - 2, grid.height - 3) + 1):
            for x in range(max(0, min_x), min(max_x - 2, grid.width - 3) + 1):
                pattern_3x3 = self._extract_3x3(grid, x, y)
                detection = self._match_glider_pattern(pattern_3x3, x, y)

                if detection and detection.confidence >= self.confidence_threshold:
                    detections.append(detection)

        # Sort by confidence (highest first)
        detections.sort(key=lambda d: d.confidence, reverse=True)

        # Apply non-maximum suppression to avoid duplicate detections
        detections = self._non_max_suppression(detections)

        return detections

    def _extract_3x3(self, grid: Grid, center_x: int, center_y: int) -> np.ndarray:
        """Extract 3x3 pattern centered at given coordinates."""
        pattern = np.zeros((3, 3), dtype=bool)

        for dy in range(-1, 2):
            for dx in range(-1, 2):
                grid_y = center_y + dy
                grid_x = center_x + dx

                # Boundary check
                if (0 <= grid_x < grid.width and 0 <= grid_y < grid.height):
                    pattern[dy + 1, dx + 1] = grid[grid_y, grid_x]

        return pattern

    def _match_glider_pattern(self, pattern_3x3: np.ndarray, center_x: int, center_y: int) -> Optional[GliderDetection]:
        """Match 3x3 pattern against all glider templates.

        Returns the best matching glider detection, or None if no good match.
        """
        if pattern_3x3.shape != (3, 3):
            return None

        # Must have exactly 5 alive cells (glider signature)
        alive_count = np.sum(pattern_3x3)
        if alive_count != 5:
            return None

        best_match = None
        best_confidence = 0.0
        best_orientation = GliderOrientation.NORTHWEST
        best_phase = GliderPhase.PHASE_0

        # Test against all glider templates
        for (orientation, phase), template in self.templates.items():
            # Calculate similarity (0.0 = identical, 1.0 = completely different)
            similarity = pattern_similarity(pattern_3x3, template)

            # Convert to confidence (1.0 = perfect match, 0.0 = completely different)
            confidence = 1.0 - min(similarity, 1.0)  # Cap similarity at 1.0

            if confidence > best_confidence:
                best_confidence = confidence
                best_match = template.copy()
                best_orientation = orientation
                best_phase = phase

        # Only return if confidence meets threshold
        if best_confidence >= self.confidence_threshold and best_match is not None:
            return GliderDetection(
                x=center_x,
                y=center_y,
                orientation=best_orientation,
                phase=best_phase,
                confidence=best_confidence,
                pattern_match=pattern_3x3.copy()
            )

        return None

    def _non_max_suppression(self, detections: List[GliderDetection], overlap_threshold: float = 0.3) -> List[GliderDetection]:
        """Apply non-maximum suppression to avoid overlapping detections.

        Args:
            detections: List of detections sorted by confidence (highest first)
            overlap_threshold: Maximum allowed overlap ratio

        Returns:
            Filtered list with overlaps removed
        """
        filtered = []

        for detection in detections:
            # Check if this detection overlaps significantly with any kept detection
            overlaps = False
            for kept in filtered:
                distance = np.sqrt((detection.x - kept.x)**2 + (detection.y - kept.y)**2)
                if distance < 1.5:  # Centers too close (<1.5 cells)
                    overlaps = True
                    break

            if not overlaps:
                filtered.append(detection)

        return filtered

    def track_glider_movement(self, grid_history: List[Grid], min_track_length: int = 3) -> List[GliderTrack]:
        """Track glider movement across multiple grid generations.

        Args:
            grid_history: List of grids in chronological order (oldest first)
            min_track_length: Minimum number of detections to form a track

        Returns:
            List of GliderTrack objects representing detected trajectories
        """
        if len(grid_history) < min_track_length:
            return []

        # Detect gliders in each frame
        frame_detections = []
        for grid in grid_history:
            detections = self.detect_gliders(grid)
            frame_detections.append(detections)

        # Build tracks by associating detections across frames
        tracks = self._associate_detections(frame_detections)

        # Filter tracks by length
        return [track for track in tracks if len(track.detections) >= min_track_length]

    def _associate_detections(self, frame_detections: List[List[GliderDetection]]) -> List[GliderTrack]:
        """Associate detections across frames to form tracks."""
        tracks = []
        used_detections = set()
        track_id_counter = 0

        for frame_idx, detections in enumerate(frame_detections):
            for detection in detections:
                detection_key = (frame_idx, detection.x, detection.y)

                if detection_key in used_detections:
                    continue

                # Try to find this glider in previous and next frames
                track = self._build_track_from_detection(
                    frame_detections, frame_idx, detection, track_id_counter
                )

                if len(track.detections) >= 2:  # Require at least 2 detections for track
                    tracks.append(track)
                    track_id_counter += 1

                    # Mark all detections in this track as used
                    for track_detection in track.detections:
                        used_key = (track_detection.frame_idx, track_detection.x, track_detection.y)
                        used_detections.add(used_key)

        return tracks

    def _build_track_from_detection(self, frame_detections: List[List[GliderDetection]],
                                   start_frame: int, start_detection: GliderDetection,
                                   track_id: int) -> GliderTrack:
        """Build a track starting from a single detection."""
        track_detections = [start_detection]
        current_frame = start_frame
        current_pos = (start_detection.x, start_detection.y)

        # Track backward from start
        for frame_idx in range(start_frame - 1, -1, -1):
            prev_detection = self._find_nearest_detection(
                frame_detections[frame_idx], current_pos, max_distance=self.max_distance
            )
            if prev_detection:
                track_detections.insert(0, prev_detection)
                current_pos = self._predict_previous_position(prev_detection, start_detection)
            else:
                break

        # Track forward from start
        for frame_idx in range(start_frame + 1, len(frame_detections)):
            next_detection = self._find_nearest_detection(
                frame_detections[frame_idx], current_pos, max_distance=self.max_distance
            )
            if next_detection:
                track_detections.append(next_detection)
                current_pos = (next_detection.x, next_detection.y)
            else:
                break

        # Generate predicted path
        predicted_path = self._predict_full_path(track_detections)

        return GliderTrack(
            glider_id=track_id,
            detections=track_detections,
            predicted_path=predicted_path
        )

    def _find_nearest_detection(self, detections: List[GliderDetection],
                               position: Tuple[int, int], max_distance: float) -> Optional[GliderDetection]:
        """Find detection nearest to given position within max_distance."""
        x, y = position
        nearest = None
        min_distance = float('inf')

        for detection in detections:
            distance = np.sqrt((detection.x - x)**2 + (detection.y - y)**2)
            if distance < min_distance and distance <= max_distance:
                min_distance = distance
                nearest = detection

        return nearest

    def _predict_previous_position(self, earlier: GliderDetection, later: GliderDetection) -> Tuple[int, int]:
        """Predict where glider was before earlier detection."""
        # Simple back-projection based on movement direction
        dx = later.x - earlier.x
        dy = later.y - earlier.y

        return (earlier.x - dx, earlier.y - dy)

    def _predict_full_path(self, detections: List[GliderDetection]) -> List[Tuple[int, int]]:
        """Generate predicted path from detection sequence."""
        if len(detections) < 2:
            return [(d.x, d.y) for d in detections]

        # Fit linear trajectory
        xs = [d.x for d in detections]
        ys = [d.y for d in detections]

        # Simple endpoint extrapolation (can be improved with curve fitting)
        start_x, start_y = xs[0], ys[0]
        end_x, end_y = xs[-1], ys[-1]

        steps = max(abs(end_x - start_x), abs(end_y - start_y)) + 1
        if steps <= 1:
            return list(zip(xs, ys))

        # Linear interpolation
        path = []
        for i in range(len(detections) - 1):
            path.append((detections[i].x, detections[i].y))
        path.append((detections[-1].x, detections[-1].y))

        return path

    def calculate_tracking_accuracy(self, true_path: List[Tuple[int, int]],
                                   detected_path: List[Tuple[int, int]]) -> Tuple[float, float]:
        """Calculate tracking accuracy metrics.

        Args:
            true_path: List of (x,y) true glider positions
            detected_path: List of (x,y) detected glider positions

        Returns:
            (mean_error, max_error) in cells
        """
        if len(true_path) != len(detected_path) or len(true_path) == 0:
            return (float('inf'), float('inf'))

        errors = []
        for true_pos, detected_pos in zip(true_path, detected_path):
            distance = np.sqrt((true_pos[0] - detected_pos[0])**2 +
                             (true_pos[1] - detected_pos[1])**2)
            errors.append(distance)

        return (float(np.mean(errors)), float(max(errors)))


# Default detector instance
default_detector = GliderDetector()
