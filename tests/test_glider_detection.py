"""Comprehensive tests for glider detection and tracking algorithms.

Tests pattern recognition accuracy, tracking precision, and propagation
validation. Ensures Phase 1 accuracy requirements are met (>=95% success
rate with <=2.5 cell tracking accuracy).
"""

import pytest
import numpy as np
from src.core.grid import Grid
from src.core.conway import ConwayEngine, default_engine as conway
from src.patterns.glider import (
    GliderOrientation, GliderPhase, get_glider_pattern,
    predict_glider_position
)
from src.patterns.detector import GliderDetector, GliderDetection


class TestGliderPatternLibrary:
    """Test glider pattern definitions and canonical forms."""

    def test_canonical_northwest_glider(self):
        """Test the standard northwest glider pattern."""
        pattern = get_glider_pattern(GliderOrientation.NORTHWEST, GliderPhase.PHASE_0)

        # Verify 5-cell glider signature
        assert np.sum(pattern) == 5
        assert pattern.shape == (3, 3)

        # Check specific cell positions for northwest glider
        expected_pattern = np.array([
            [False, True, False],
            [False, False, True],
            [True, True, True]
        ], dtype=bool)
        np.testing.assert_array_equal(pattern, expected_pattern)

    def test_all_glider_phases_unique(self):
        """Each phase of each orientation should be unique."""
        seen_patterns = set()

        for orientation in GliderOrientation:
            for phase in GliderPhase:
                pattern = get_glider_pattern(orientation, phase)
                pattern_bytes = pattern.tobytes()

                assert pattern_bytes not in seen_patterns, \
                    f"Duplicate pattern for {orientation}-{phase}"
                seen_patterns.add(pattern_bytes)

                # All should have exactly 5 alive cells
                assert np.sum(pattern) == 5

    def test_orientation_rotations(self):
        """Test that orientations are properly rotated versions."""
        nw_pattern = get_glider_pattern(GliderOrientation.NORTHWEST, GliderPhase.PHASE_0)
        ne_pattern = get_glider_pattern(GliderOrientation.NORTHEAST, GliderPhase.PHASE_0)

        # NE should be 90° rotation of NW (clockwise)
        rotated_nw = np.rot90(nw_pattern, k=-1)  # -1 for clockwise
        np.testing.assert_array_equal(ne_pattern, rotated_nw)

    def test_movement_prediction(self):
        """Test glider position prediction after movement."""
        # Start at (10, 10) facing northwest
        start_x, start_y = 10, 10
        orientation = GliderOrientation.NORTHWEST

        # After 4 steps, should have moved (-1, -1) = northwest
        predicted = predict_glider_position(start_x, start_y, orientation, 4)
        assert predicted == (9, 9)  # 10-1, 10-1


class TestGliderDetector:
    """Test glider detection algorithms."""

    def setup_method(self):
        """Create detector with test-friendly settings."""
        self.detector = GliderDetector(confidence_threshold=0.9, max_distance=3.0)

    def test_perfect_glider_detection(self):
        """Detect glider with perfect pattern match."""
        # Create grid with northwest glider
        grid = Grid(10, 10)
        glider_pattern = get_glider_pattern(GliderOrientation.NORTHWEST, GliderPhase.PHASE_0)

        # Place glider at (5, 5) center
        for dy in range(3):
            for dx in range(3):
                if glider_pattern[dy, dx]:
                    grid[5 + dy - 1, 5 + dx - 1] = True

        detections = self.detector.detect_gliders(grid)

        assert len(detections) == 1
        detection = detections[0]

        assert detection.x == 5
        assert detection.y == 5
        assert detection.orientation == GliderOrientation.NORTHWEST
        assert detection.phase == GliderPhase.PHASE_0
        assert abs(detection.confidence - 1.0) < 0.001  # Perfect match

    def test_multiple_gliders_detection(self):
        """Detect multiple gliders simultaneously."""
        grid = Grid(20, 20)

        # Place two gliders of different orientations
        nw_pattern = get_glider_pattern(GliderOrientation.NORTHWEST, GliderPhase.PHASE_0)
        ne_pattern = get_glider_pattern(GliderOrientation.NORTHEAST, GliderPhase.PHASE_0)

        # NW glider at (5, 5)
        for dy in range(3):
            for dx in range(3):
                if nw_pattern[dy, dx]:
                    grid[5 + dy - 1, 5 + dx - 1] = True

        # NE glider at (15, 15)
        for dy in range(3):
            for dx in range(3):
                if ne_pattern[dy, dx]:
                    grid[15 + dy - 1, 15 + dx - 1] = True

        detections = self.detector.detect_gliders(grid)

        assert len(detections) == 2

        # Check both detections
        detections_by_pos = {(d.x, d.y): d for d in detections}
        assert (5, 5) in detections_by_pos
        assert (15, 15) in detections_by_pos

        nw_detection = detections_by_pos[(5, 5)]
        assert nw_detection.orientation == GliderOrientation.NORTHWEST

        ne_detection = detections_by_pos[(15, 15)]
        assert ne_detection.orientation == GliderOrientation.NORTHEAST

    def test_no_false_positives(self):
        """Avoid detecting gliders in random noise."""
        grid = Grid(20, 20)
        grid.randomize(0.3)  # 30% density random pattern

        # Lower confidence threshold to see if we get false positives
        low_conf_detector = GliderDetector(confidence_threshold=0.0)
        detections = low_conf_detector.detect_gliders(grid)

        # Should detect very few if any false positives in random noise
        # Allow small number but verify low confidence
        for detection in detections:
            assert detection.confidence < 0.5, f"High confidence false positive: {detection.confidence}"

    def test_boundary_detection(self):
        """Detect gliders near grid boundaries."""
        grid = Grid(8, 8)

        # Place glider near corner (but with enough space for 3x3 pattern)
        glider_pattern = get_glider_pattern(GliderOrientation.NORTHWEST, GliderPhase.PHASE_0)

        # Place at (2, 2) - close to boundary but detectable
        for dy in range(3):
            for dx in range(3):
                if glider_pattern[dy, dx]:
                    grid[2 + dy - 1, 2 + dx - 1] = True

        detections = self.detector.detect_gliders(grid)

        # Should still detect despite proximity to boundary
        assert len(detections) == 1
        assert detections[0].x == 2
        assert detections[0].y == 2


class TestGliderTracking:
    """Test glider movement tracking across generations."""

    def setup_method(self):
        """Create detector for tracking tests."""
        self.detector = GliderDetector(confidence_threshold=0.9, max_distance=3.0)

    def test_simple_glider_tracking(self):
        """Track glider movement over 8 generations (2 full cycles)."""
        # Create initial grid with northwest glider
        grid = Grid(16, 16)
        glider_pattern = get_glider_pattern(GliderOrientation.NORTHWEST, GliderPhase.PHASE_0)

        # Place at center-ish position
        center_x, center_y = 8, 8
        for dy in range(3):
            for dx in range(3):
                if glider_pattern[dy, dx]:
                    grid[center_y + dy - 1, center_x + dx - 1] = True

        # Track evolution for 8 steps (should move northwest)
        grid_history = [grid.copy()]
        current_grid = grid

        for _ in range(8):
            conway.step(current_grid)
            grid_history.append(current_grid.copy())

        # Track gliders across frames
        tracks = self.detector.track_glider_movement(grid_history, min_track_length=3)

        # Should find one track
        assert len(tracks) == 1
        track = tracks[0]

        # Track should cover multiple detections
        assert len(track.detections) >= 3

        # Verify movement direction (northwest: decreasing x,y)
        first_pos = track.detections[0]
        last_pos = track.detections[-1]

        assert last_pos.x <= first_pos.x  # Moving west (left)
        assert last_pos.y <= first_pos.y  # Moving north (up)

        # Estimate movement per step should be ~0.25 cells/step (since glider moves 1 every 4 steps)
        total_steps = len(track.detections) - 1
        total_dx = last_pos.x - first_pos.x
        total_dy = last_pos.y - first_pos.y

        if total_steps > 0:
            avg_dx_per_step = total_dx / total_steps
            avg_dy_per_step = total_dy / total_steps

            # Should be roughly -0.25, -0.25 for northwest movement
            assert -0.4 < avg_dx_per_step < -0.1
            assert -0.4 < avg_dy_per_step < -0.1

    def test_tracking_accuracy_2_5_cells(self):
        """Verify tracking accuracy requirement (<=2.5 cells)."""
        # Create controlled glider path
        true_positions = []
        grid_history = []

        # Start glider
        grid = Grid(32, 32)  # Larger grid to avoid boundary effects
        glider_pattern = get_glider_pattern(GliderOrientation.NORTHWEST, GliderPhase.PHASE_0)

        start_x, start_y = 16, 16
        for dy in range(3):
            for dx in range(3):
                if glider_pattern[dy, dx]:
                    grid[start_y + dy - 1, start_x + dx - 1] = True

        true_positions.append((start_x, start_y))
        grid_history.append(grid.copy())

        # Evolve for 12 steps, recording true glider position each time
        # (Note: real tracking wouldn't know true position, but we calculate it for accuracy testing)
        for step in range(12):
            conway.step(grid)

            # For accuracy testing, detect the glider position in the evolved grid
            # (In practice, the detector would need to track from the history)
            detections = self.detector.detect_gliders(grid)
            if detections:
                # Use highest confidence detection as "true" position for this test
                best_detection = max(detections, key=lambda d: d.confidence)
                true_positions.append((best_detection.x, best_detection.y))
            else:
                # If detection fails, use predicted position
                predicted = predict_glider_position(true_positions[-1][0], true_positions[-1][1],
                                                   GliderOrientation.NORTHWEST, step + 1)
                true_positions.append(predicted)

            grid_history.append(grid.copy())

        # Now track the glider using the real tracking algorithm
        tracks = self.detector.track_glider_movement(grid_history, min_track_length=5)

        if tracks:
            # Get the main track
            main_track = max(tracks, key=lambda t: len(t.detections))

            # Extract detected positions (focus on last part where tracking stabilized)
            detected_positions = [(d.x, d.y) for d in main_track.detections]

            # Compare against true positions
            mean_error, max_error = self.detector.calculate_tracking_accuracy(
                true_positions[-len(detected_positions):], detected_positions
            )

            # Phase 1 requirement: tracking accuracy <= 2.5 cells
            assert max_error <= 2.5, f"Tracking accuracy {max_error:.2f} > 2.5 cells max allowed"
            print(f"Tracking accuracy: mean={mean_error:.2f}, max={max_error:.2f} cells")
        else:
            pytest.fail("No glider tracks found in test sequence")


class TestDetectionAccuracy_95Percent:
    """Test that detection meets Phase 1 accuracy requirement (>=95% success)."""

    def test_glider_detection_success_rate(self):
        """Achieve >=95% detection success rate across orientations."""
        self.detector = GliderDetector(confidence_threshold=0.9)  # Strict threshold

        detection_counts = {orientation: 0 for orientation in GliderOrientation}
        success_counts = {orientation: 0 for orientation in GliderOrientation}

        # Test multiple positions for each orientation
        for orientation in GliderOrientation:
            pattern = get_glider_pattern(orientation, GliderPhase.PHASE_0)
            detection_counts[orientation] = 10  # Test 10 positions per orientation

            for offset in range(10):
                # Place glider at different positions (avoiding boundaries)
                center_x, center_y = 5 + offset, 5 + (offset % 3)

                # Create grid and place glider
                grid = Grid(16, 16)
                for dy in range(3):
                    for dx in range(3):
                        if pattern[dy, dx]:
                            grid_y = center_y + dy - 1
                            grid_x = center_x + dx - 1
                            if 0 <= grid_x < grid.width and 0 <= grid_y < grid.height:
                                grid[grid_y, grid_x] = True

                # Attempt detection
                detections = self.detector.detect_gliders(grid)

                # Check if correct glider was detected near expected position
                correct_detection = False
                for detection in detections:
                    # Check if detection is close to expected position and has correct orientation
                    distance = np.sqrt((detection.x - center_x)**2 + (detection.y - center_y)**2)
                    if distance <= 1.0 and detection.orientation == orientation:
                        correct_detection = True
                        break

                if correct_detection:
                    success_counts[orientation] += 1

        # Calculate success rates
        total_attempts = sum(detection_counts.values())
        total_successes = sum(success_counts.values())

        if total_attempts > 0:
            success_rate = total_successes / total_attempts
            print(f"Detection success rate: {success_rate:.1%}")

            # Phase 1 requirement: >=95% success rate
            assert success_rate >= 0.95, f"Detection rate {success_rate:.1%} < 95% required"
        else:
            pytest.fail("No detection attempts made")


class TestDetectorRobustness:
    """Test detector robustness to noise and edge cases."""

    def test_noise_resistance(self):
        """Detect gliders amidst background noise."""
        self.detector = GliderDetector(confidence_threshold=0.85)  # Slightly lower for noisy conditions

        grid = Grid(20, 20)

        # Add 20% background noise
        grid.randomize(0.2)

        # Overlay clean glider on center
        glider_pattern = get_glider_pattern(GliderOrientation.NORTHWEST, GliderPhase.PHASE_0)
        center_x, center_y = 10, 10

        for dy in range(3):
            for dx in range(3):
                if glider_pattern[dy, dx]:
                    grid[center_y + dy - 1, center_x + dx - 1] = True

        detections = self.detector.detect_gliders(grid)

        # Should detect glider amidst noise
        assert len(detections) >= 1

        # Find detection closest to glider center
        best_detection = min(detections, key=lambda d: (d.x - center_x)**2 + (d.y - center_y)**2)

        # Should be reasonably close
        distance = np.sqrt((best_detection.x - center_x)**2 + (best_detection.y - center_y)**2)
        assert distance <= 2.0, f"Noisy detection too far: {distance} cells"
        assert best_detection.orientation == GliderOrientation.NORTHWEST

    def test_confidence_filtering(self):
        """Lower confidence threshold detects more, higher is more selective."""
        grid = Grid(20, 20)

        # Place known glider
        glider_pattern = get_glider_pattern(GliderOrientation.NORTHWEST, GliderPhase.PHASE_0)
        center_x, center_y = 10, 10

        for dy in range(3):
            for dx in range(3):
                if glider_pattern[dy, dx]:
                    grid[center_y + dy - 1, center_x + dx - 1] = True

        # Low threshold
        low_detector = GliderDetector(confidence_threshold=0.0)
        low_detections = low_detector.detect_gliders(grid)

        # High threshold
        high_detector = GliderDetector(confidence_threshold=0.99)
        high_detections = high_detector.detect_gliders(grid)

        # Low threshold should find at least as many detections
        assert len(low_detections) >= len(high_detections)

        # High confidence detections should be more accurate
        if high_detections:
            for detection in high_detections:
                assert detection.confidence >= 0.99
