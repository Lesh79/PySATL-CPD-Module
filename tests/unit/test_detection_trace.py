"""
Tests for DetectionTrace container class.
"""

import numpy as np
import pytest

from pysatl_cpd.detection_trace import DetectionTrace


class TestDetectionTrace:
    """Test suite for DetectionTrace class."""

    def test_initialization_with_detected_changes_only(self) -> None:
        """Test initialization with only detected_changes parameter."""
        trace = DetectionTrace[float](detected_changes=[10, 25, 42])

        assert trace.detected_changes == [10, 25, 42]
        assert trace.observation_scores is None

    def test_initialization_with_observation_scores(self) -> None:
        """Test initialization with both parameters."""
        scores = [0.1, 0.5, 0.3, 0.8]
        trace = DetectionTrace(detected_changes=[2, 3], observation_scores=scores)

        assert trace.detected_changes == [2, 3]
        assert trace.observation_scores == scores

    def test_initialization_with_empty_detected_changes(self) -> None:
        """Test initialization with empty change list."""
        trace = DetectionTrace[float](detected_changes=[])

        assert trace.detected_changes == []
        assert trace.observation_scores is None
        assert len(trace) == 0

    def test_initialization_with_numpy_scores(self) -> None:
        """Test initialization with numpy array as observation scores."""
        scores = np.array([0.1, 0.5, 0.3, 0.8, 0.2])
        trace = DetectionTrace(detected_changes=[1, 3], observation_scores=scores)

        assert trace.detected_changes == [1, 3]
        assert trace.observation_scores is not None
        assert np.array_equal(list(trace.observation_scores), scores)

    def test_raises_value_error_for_negative_change_index(self) -> None:
        """Test ValueError raised when detected change index is negative."""
        with pytest.raises(ValueError, match="Detected change indices must be non-negative"):
            DetectionTrace(detected_changes=[-1, 5, 10])

        with pytest.raises(ValueError, match="Detected change indices must be non-negative"):
            DetectionTrace(detected_changes=[5, -2, 10])

    def test_allows_zero_change_index(self) -> None:
        """Test that zero is allowed as a change index."""
        trace = DetectionTrace[float](detected_changes=[0, 5, 10])

        assert trace.detected_changes == [0, 5, 10]

    def test_len_returns_correct_count(self) -> None:
        """Test that __len__ returns number of detected changes."""
        trace = DetectionTrace[float](detected_changes=[5, 10, 15, 20])

        assert len(trace) == 4

        empty_trace = DetectionTrace[float](detected_changes=[])
        assert len(empty_trace) == 0

    def test_str_without_scores(self) -> None:
        """Test string representation without observation scores."""
        trace = DetectionTrace[float](detected_changes=[1, 2, 3])

        assert str(trace) == "DetectionTrace(changes=3, without scores)"

    def test_str_with_scores(self) -> None:
        """Test string representation with observation scores."""
        trace = DetectionTrace(detected_changes=[1, 2, 3], observation_scores=[0.1, 0.2, 0.3, 0.4])

        assert str(trace) == "DetectionTrace(changes=3, with scores)"

    def test_str_with_empty_changes(self) -> None:
        """Test string representation with no detected changes."""
        trace = DetectionTrace[float](detected_changes=[])

        assert str(trace) == "DetectionTrace(changes=0, without scores)"

    def test_multiple_instances_independence(self) -> None:
        """Test that instances are independent."""
        trace1 = DetectionTrace[float](detected_changes=[1, 2, 3])
        trace2 = DetectionTrace[float](detected_changes=[4, 5, 6])

        assert trace1.detected_changes == [1, 2, 3]
        assert trace2.detected_changes == [4, 5, 6]

    def test_scores_mutability(self) -> None:
        """Test that observation_scores can be mutable collections."""
        scores = [0.1, 0.2, 0.3]
        trace = DetectionTrace(detected_changes=[1], observation_scores=scores)

        # Verify mutation is possible (important for online algorithms)
        scores.append(0.4)
        assert len(trace.observation_scores) == 4  # type: ignore
        assert trace.observation_scores[-1] == 0.4  # type: ignore

    def test_with_online_algorithm_pattern(self) -> None:
        """Test pattern typical for online algorithm output."""
        # Online algorithm may accumulate changes and scores incrementally
        trace = DetectionTrace[float](
            detected_changes=[5, 12, 23],
            observation_scores=[],  # Empty list for incremental updates
        )

        # Simulate online addition
        trace.observation_scores.append(0.1)  # type: ignore
        trace.observation_scores.append(0.2)  # type: ignore

        assert len(trace.observation_scores) == 2  # type: ignore

    def test_with_offline_algorithm_pattern(self) -> None:
        """Test pattern typical for offline algorithm output."""
        # Offline algorithm returns complete results
        scores = np.random.rand(100)
        trace = DetectionTrace(detected_changes=[10, 30, 50, 70], observation_scores=scores)

        assert len(trace.detected_changes) == 4
        assert trace.observation_scores is scores
