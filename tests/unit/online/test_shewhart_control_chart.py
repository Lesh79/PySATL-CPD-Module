"""
Tests for Shewhart control chart algorithm.
"""

import pytest
import re

from pysatl_cpd.online.shewhart_control_chart import (
    ShewhartControlChart,
    ShewhartControlChartConfiguration,
    ShewhartControlChartState,
)


class TestShewhartControlChartConfiguration:
    """Test suite for ShewhartControlChartConfiguration."""

    def test_valid_configuration(self) -> None:
        """Test valid configuration initialization."""
        config = ShewhartControlChartConfiguration(learning_period_size=50, window_size=10)

        assert config.learning_period_size == 50
        assert config.window_size == 10

    def test_raises_value_error_for_non_positive_learning_period(self) -> None:
        """Test ValueError raised when learning_period_size is not positive."""
        with pytest.raises(ValueError, match="learning_period_size must be positive"):
            ShewhartControlChartConfiguration(learning_period_size=0, window_size=10)

        with pytest.raises(ValueError, match="learning_period_size must be positive"):
            ShewhartControlChartConfiguration(learning_period_size=-5, window_size=10)

    def test_raises_value_error_for_non_positive_window_size(self) -> None:
        """Test ValueError raised when window_size is not positive."""
        with pytest.raises(ValueError, match="window_size must be positive"):
            ShewhartControlChartConfiguration(learning_period_size=50, window_size=0)

        with pytest.raises(ValueError, match="window_size must be positive"):
            ShewhartControlChartConfiguration(learning_period_size=50, window_size=-3)

    def test_raises_value_error_when_window_size_exceeds_learning_period(self) -> None:
        """Test ValueError raised when window_size > learning_period_size."""
        msg = "window_size (20) must be less than or equal to learning_period_size (10)"
        with pytest.raises(ValueError, match=f"^{re.escape(msg)}$"):
            ShewhartControlChartConfiguration(learning_period_size=10, window_size=20)


class TestShewhartControlChartState:
    """Test suite for ShewhartControlChartState."""

    def test_default_state(self) -> None:
        """Test default state values."""
        state = ShewhartControlChartState()

        assert state.is_in_learning_period is False
        assert state.mean == 0.0
        assert state.variance == 0.0
        assert state.standard_deviation == 0.0
        assert state.samples_count == 0
        assert state.window_mean == 0.0
        assert state.window_sum == 0.0
        assert state.window_size == 0
        assert state.window_contents == []

    def test_custom_state(self) -> None:
        """Test custom state values."""
        state = ShewhartControlChartState(
            is_in_learning_period=True,
            mean=5.0,
            variance=2.0,
            standard_deviation=1.414,
            samples_count=100,
            window_mean=4.5,
            window_sum=45.0,
            window_size=10,
            window_contents=[1.0, 2.0, 3.0],
        )

        assert state.is_in_learning_period is True
        assert state.mean == 5.0
        assert state.variance == 2.0
        assert state.standard_deviation == 1.414
        assert state.samples_count == 100
        assert state.window_mean == 4.5
        assert state.window_sum == 45.0
        assert state.window_size == 10
        assert state.window_contents == [1.0, 2.0, 3.0]

    def test_state_immutability(self) -> None:
        """Test that state is immutable."""
        state = ShewhartControlChartState()

        with pytest.raises(AttributeError):
            state.mean = 10.0  # type: ignore


class TestShewhartControlChart:
    """Test suite for ShewhartControlChart algorithm."""

    def test_initialization(self) -> None:
        """Test algorithm initialization."""
        chart = ShewhartControlChart(learning_period_size=50, window_size=10)

        assert chart.name == "ShewhartControlChart"
        assert chart.configuration.learning_period_size == 50
        assert chart.configuration.window_size == 10

    def test_initialization_with_invalid_params(self) -> None:
        """Test initialization with invalid parameters raises ValueError."""
        with pytest.raises(ValueError):
            ShewhartControlChart(learning_period_size=0, window_size=10)

        with pytest.raises(ValueError):
            ShewhartControlChart(learning_period_size=50, window_size=0)

        with pytest.raises(ValueError):
            ShewhartControlChart(learning_period_size=10, window_size=20)

    def test_process_during_learning_period_returns_zero(self) -> None:
        """Test that process returns 0 during learning period."""
        chart = ShewhartControlChart(learning_period_size=5, window_size=3)

        for i in range(5):
            result = chart.process(float(i))
            assert result == 0.0

    def test_process_after_learning_period_returns_statistic(self) -> None:
        """Test that process returns non-zero statistic after learning period."""
        chart = ShewhartControlChart(learning_period_size=5, window_size=3)

        # Fill learning period
        observations = [1.0, 2.0, 3.0, 4.0, 5.0]
        for obs in observations:
            chart.process(obs)

        # After learning period, statistic should be computed
        result = chart.process(6.0)
        assert result > 0.0

    def test_statistic_formula(self) -> None:
        """Test that statistic follows expected formula."""
        chart = ShewhartControlChart(learning_period_size=10, window_size=3)

        # Feed constant data
        for _ in range(10):
            chart.process(5.0)

        # Add deviation
        result = chart.process(10.0)

        # With constant data, window_mean = 5, global_mean = 5, std = 0
        # Statistic should be 0 (since std = 0)
        assert result == 0.0

    def test_reset(self) -> None:
        """Test reset functionality."""
        chart = ShewhartControlChart(learning_period_size=5, window_size=3)

        # Process some data
        for i in range(10):
            chart.process(float(i))

        # Reset
        chart.reset()

        # After reset, should be back in learning period
        result = chart.process(1.0)
        assert result == 0.0

        # State should be cleared
        state = chart.state
        assert state.samples_count == 1
        assert state.mean == 1.0

    def test_state_property(self) -> None:
        """Test that state property returns correct snapshot."""
        chart = ShewhartControlChart(learning_period_size=10, window_size=3)

        # Process some observations
        chart.process(1.0)
        chart.process(2.0)
        chart.process(3.0)

        state = chart.state

        assert isinstance(state, ShewhartControlChartState)
        assert state.samples_count == 3
        assert state.window_size == 3
        assert len(state.window_contents) == 3
        assert state.is_in_learning_period is True

    def test_sliding_window_behavior(self) -> None:
        """Test that sliding window maintains correct size."""
        chart = ShewhartControlChart(learning_period_size=10, window_size=3)

        # Process 5 observations
        for i in range(5):
            chart.process(float(i))

        state = chart.state
        assert state is not None
        assert len(state.window_contents) == 3
        assert state.window_contents == [2.0, 3.0, 4.0]

    def test_online_statistic_updates(self) -> None:
        """Test that statistic updates correctly over time."""
        chart = ShewhartControlChart(learning_period_size=5, window_size=2)

        # Learning period
        observations = [1.0, 2.0, 3.0, 4.0, 5.0]
        for obs in observations:
            chart.process(obs)

        # Should have non-zero statistic now
        results = []
        for i in range(10, 15):
            results.append(chart.process(float(i)))

        # All results should be non-zero after learning period
        assert all(r > 0.0 for r in results)

    def test_repr(self) -> None:
        """Test string representation."""
        chart = ShewhartControlChart(learning_period_size=50, window_size=10)

        repr_str = repr(chart)
        assert "ShewhartControlChart" in repr_str
        assert "w = 10" in repr_str

    def test_configuration_property(self) -> None:
        """Test configuration property returns correct object."""
        chart = ShewhartControlChart(learning_period_size=100, window_size=20)

        config = chart.configuration
        assert isinstance(config, ShewhartControlChartConfiguration)
        assert config.learning_period_size == 100
        assert config.window_size == 20

    def test_inheritance(self) -> None:
        """Test that algorithm properly inherits from OnlineAlgorithm."""
        chart = ShewhartControlChart(learning_period_size=10, window_size=3)

        from pysatl_cpd.online.ionline_algorithm import OnlineAlgorithm

        assert isinstance(chart, OnlineAlgorithm)
