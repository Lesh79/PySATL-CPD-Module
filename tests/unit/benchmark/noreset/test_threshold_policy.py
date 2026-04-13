# tests/benchmark/noreset/test_threshold_policy.py

"""Tests for ThresholdPolicy implementations."""

import numpy as np
import pytest

from pysatl_cpd.benchmark.noreset.threshold_policy import (
    EventBasedPolicy,
    PointBasedPolicy,
    ThresholdPolicy,
)
from pysatl_cpd.core.typedefs import UnivariateNumericArray

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_df(*values: float) -> UnivariateNumericArray:
    """Create a UnivariateNumericArray from float values."""
    return np.array(values, dtype=np.float64)


# ---------------------------------------------------------------------------
# TestThresholdProtocol
# ---------------------------------------------------------------------------


class TestThresholdProtocol:
    """Tests that concrete policies satisfy the ThresholdPolicy protocol."""

    def test_point_based_implements_protocol(self) -> None:
        """PointBasedPolicy must be recognised as ThresholdPolicy at runtime."""
        policy: PointBasedPolicy = PointBasedPolicy()
        assert isinstance(policy, ThresholdPolicy)

    def test_event_based_implements_protocol(self) -> None:
        """EventBasedPolicy must be recognised as ThresholdPolicy at runtime."""
        policy: EventBasedPolicy = EventBasedPolicy(max_delay=5)
        assert isinstance(policy, ThresholdPolicy)


# ---------------------------------------------------------------------------
# TestPointBasedPolicyInit
# ---------------------------------------------------------------------------


class TestPointBasedPolicyInit:
    """Tests for PointBasedPolicy constructor."""

    def test_default_strict_is_true(self) -> None:
        """Default strict parameter must be True."""
        policy: PointBasedPolicy = PointBasedPolicy()
        assert policy.strict is True

    def test_explicit_strict_false(self) -> None:
        """Explicit strict=False must be stored correctly."""
        policy: PointBasedPolicy = PointBasedPolicy(strict=False)
        assert policy.strict is False


# ---------------------------------------------------------------------------
# TestPointBasedPolicyApply
# ---------------------------------------------------------------------------


class TestPointBasedPolicyApply:
    """Tests for PointBasedPolicy.apply — parametrized over common cases."""

    @pytest.mark.parametrize(
        "values, threshold, strict, change_points, expected",
        [
            # no signals — all below threshold
            ([0.1, 0.2, 0.3], 1.0, True, [], []),
            # all signals — all strictly above threshold
            ([2.0, 3.0, 4.0], 1.0, True, [], [1, 2, 3]),
            # strict=True: equal value is NOT a signal
            ([1.0, 2.0, 1.0], 1.0, True, [], [2]),
            # strict=False: equal value IS a signal
            ([1.0, 2.0, 0.5], 1.0, False, [], [1, 2]),
            # empty detection function
            ([], 1.0, True, [], []),
            # single element — signal
            ([5.0], 1.0, True, [], [1]),
            # single element — no signal
            ([0.5], 1.0, True, [], []),
            # indices are 1-based
            ([0.0, 0.0, 5.0, 0.0, 5.0], 1.0, True, [], [3, 5]),
            # change_points present but do not affect result
            ([2.0, 0.5, 2.0], 1.0, True, [2], [1, 3]),
        ],
        ids=[
            "all_below",
            "all_above_strict",
            "strict_true_excludes_equal",
            "strict_false_includes_equal",
            "empty_df",
            "single_signal",
            "single_no_signal",
            "returns_1based_indices",
            "change_points_do_not_affect",
        ],
    )
    def test_apply(
        self,
        values: list[float],
        threshold: float,
        strict: bool,
        change_points: list[int],
        expected: list[int],
    ) -> None:
        """
        PointBasedPolicy.apply must return 1-based signal indices.

        Any position where detection_function satisfies the threshold
        condition (strict or non-strict) is a signal. change_points
        are accepted but ignored.
        """
        policy: PointBasedPolicy = PointBasedPolicy(strict=strict)
        df: UnivariateNumericArray = make_df(*values)
        result: list[int] = policy.apply(df, threshold, change_points)
        assert result == expected


# ---------------------------------------------------------------------------
# TestEventBasedPolicyInit
# ---------------------------------------------------------------------------


class TestEventBasedPolicyInit:
    """Tests for EventBasedPolicy constructor."""

    def test_valid_init_stores_fields(self) -> None:
        """
        Constructor must store max_delay, strict_edge, strict_point correctly.

        Default strict_edge=True, strict_point=True.
        """
        policy: EventBasedPolicy = EventBasedPolicy(max_delay=5)
        assert policy.max_delay == 5
        assert policy.strict_edge is True
        assert policy.strict_point is True

    def test_explicit_strict_values_stored(self) -> None:
        """Explicit strict_edge=False and strict_point=False must be stored."""
        policy: EventBasedPolicy = EventBasedPolicy(
            max_delay=3,
            strict_edge=False,
            strict_point=False,
        )
        assert policy.max_delay == 3
        assert policy.strict_edge is False
        assert policy.strict_point is False

    def test_negative_max_delay_raises(self) -> None:
        """Negative max_delay must raise ValueError."""
        with pytest.raises(ValueError):
            EventBasedPolicy(max_delay=-1)

    def test_zero_max_delay_is_valid(self) -> None:
        """max_delay=0 means only the change point itself is in the window."""
        policy: EventBasedPolicy = EventBasedPolicy(max_delay=0)
        assert policy.max_delay == 0


# ---------------------------------------------------------------------------
# TestEventBasedPolicyApplyEdgeMode
# ---------------------------------------------------------------------------


class TestEventBasedPolicyApplyEdgeMode:
    """Tests for edge (rising-edge) detection mode — no delay windows active."""

    @pytest.mark.parametrize(
        "values, threshold, strict_edge, change_points, expected",
        [
            # basic rising edge detected
            # idx2: prev=0.0<1.0, 2.0>1.0 -> signal
            ([0.0, 0.0, 2.0, 2.0], 1.0, True, [], [3]),
            # no repeat signal while staying above threshold
            # idx2: rising edge -> signal, idx3,4: prev>=threshold -> no signal
            ([0.0, 2.0, 3.0, 4.0], 1.0, True, [], [2]),
            # falling then rising produces second signal
            # idx2: rising [2], idx3: falling, idx4: rising [4]
            ([0.0, 2.0, 0.0, 2.0], 1.0, True, [], [2, 4]),
            # strict_edge=True: prev=0.5<1.0, curr=1.0, 1.0>1.0 False -> no signal
            ([0.5, 1.0, 0.5], 1.0, True, [], []),
            # strict_edge=False: prev=0.5<1.0, curr=1.0, 1.0>=1.0 True -> signal
            ([0.5, 1.0, 0.5], 1.0, False, [], [2]),
            # first element above threshold: prev=-inf<1.0, 2.0>1.0 -> signal
            ([2.0, 0.0, 0.0], 1.0, True, [], [1]),
            # first element equal threshold, strict=False: prev=-inf<1.0, 1.0>=1.0 -> signal
            ([1.0, 0.0], 1.0, False, [], [1]),
            # first element equal threshold, strict=True: 1.0>1.0 False -> no signal
            ([1.0, 0.0], 1.0, True, [], []),
            # returns 1-based indices
            ([0.5, 2.0, 0.5], 1.0, True, [], [2]),
            # empty detection function
            ([], 1.0, True, [], []),
        ],
        ids=[
            "basic_rising_edge",
            "no_repeat_while_above",
            "falling_then_rising",
            "strict_edge_true_equal_not_signal",
            "strict_edge_false_equal_is_signal",
            "first_element_above_is_signal",
            "first_element_equal_strict_false",
            "first_element_equal_strict_true",
            "returns_1based_indices",
            "empty_df",
        ],
    )
    def test_edge_mode(
        self,
        values: list[float],
        threshold: float,
        strict_edge: bool,
        change_points: list[int],
        expected: list[int],
    ) -> None:
        """
        In edge mode (no delay windows), only rising-edge crossings are signals.

        prev is -inf for the first element. strict_edge controls whether
        the crossing condition uses strict (>) or non-strict (>=) inequality
        for the current value. prev is always checked with strict (<).
        """
        policy: EventBasedPolicy = EventBasedPolicy(
            max_delay=0,
            strict_edge=strict_edge,
            strict_point=True,
        )
        df: UnivariateNumericArray = make_df(*values)
        result: list[int] = policy.apply(df, threshold, change_points)
        assert result == expected


# ---------------------------------------------------------------------------
# TestEventBasedPolicyApplyDelayWindow
# ---------------------------------------------------------------------------


class TestEventBasedPolicyApplyDelayWindow:
    """Tests for point-based mode inside delay windows [true_cp, true_cp + max_delay]."""

    @pytest.mark.parametrize(
        "values, threshold, change_points, max_delay, strict_point, expected",
        [
            # all above in window — all are signals
            # cp=3, max_delay=2 -> window [3,5] (1-based, inclusive)
            # idx3=2.0, idx4=2.0, idx5=2.0 -> all signals
            ([0.0, 0.0, 2.0, 2.0, 2.0], 1.0, [3], 2, True, [3, 4, 5]),
            # partial signals in window
            # idx3=2.0 signal, idx4=0.5 no, idx5=2.0 signal
            ([0.0, 0.0, 2.0, 0.5, 2.0], 1.0, [3], 2, True, [3, 5]),
            # strict_point=True: equal not a signal in window
            # window [3,5], idx3=1.0, idx4=1.0: 1.0>1.0 False -> no signals
            ([0.0, 0.0, 1.0, 1.0, 0.0], 1.0, [3], 2, True, []),
            # strict_point=False: equal IS a signal in window
            # window [3,5], idx3=1.0, idx4=1.0: 1.0>=1.0 True -> signals
            ([0.0, 0.0, 1.0, 1.0, 0.0], 1.0, [3], 2, False, [3, 4]),
            # max_delay=0: window is just [cp, cp] — single point
            # cp=3, window={3}, idx3=2.0 -> signal
            ([0.0, 0.0, 2.0, 0.0], 1.0, [3], 0, True, [3]),
            # right boundary is INCLUSIVE: cp=3, max_delay=2 -> idx5 in window
            ([0.0, 0.0, 0.0, 0.0, 2.0], 1.0, [3], 2, True, [5]),
            # two change points — two windows
            # cp=[2,5], max_delay=1 -> windows [2,3] and [5,6]
            # idx2=2.0, idx3=2.0, idx5=2.0, idx6=2.0 -> all signals
            ([0.0, 2.0, 2.0, 0.0, 2.0, 2.0], 1.0, [2, 5], 1, True, [2, 3, 5, 6]),
        ],
        ids=[
            "all_above_in_window",
            "partial_signals_in_window",
            "strict_point_true_equal_not_signal",
            "strict_point_false_equal_is_signal",
            "max_delay_zero_single_point",
            "right_boundary_inclusive",
            "two_change_points_two_windows",
        ],
    )
    def test_delay_window(
        self,
        values: list[float],
        threshold: float,
        change_points: list[int],
        max_delay: int,
        strict_point: bool,
        expected: list[int],
    ) -> None:
        """
        Inside [true_cp, true_cp + max_delay] policy uses point-based mode.

        strict_point controls whether equal values are signals.
        Right boundary is inclusive. change_points are 1-based.
        """
        policy: EventBasedPolicy = EventBasedPolicy(
            max_delay=max_delay,
            strict_edge=True,
            strict_point=strict_point,
        )
        df: UnivariateNumericArray = make_df(*values)
        result: list[int] = policy.apply(df, threshold, change_points)
        assert result == expected


# ---------------------------------------------------------------------------
# TestEventBasedPolicyApplyMixed
# ---------------------------------------------------------------------------


class TestEventBasedPolicyApplyMixed:
    """Tests combining edge mode and delay windows in the same series."""

    @pytest.mark.parametrize(
        "values, threshold, change_points, max_delay, strict_edge, strict_point, expected",
        [
            # edge signal before window, point-based inside window
            # df=[0.0, 2.0, 0.0, 0.0, 2.0, 2.0], cp=[4], max_delay=1
            # window=[4,5]
            # idx1: edge, 0.0<=1.0 -> no
            # idx2: edge, prev=0.0<1.0, 2.0>1.0 -> signal [2]
            # idx3: edge, prev=2.0>=1.0 -> no (not rising)
            # idx4: window, 0.0<=1.0 -> no
            # idx5: window, 2.0>1.0 -> signal [5]
            # idx6: edge, prev=df[4]=2.0>=1.0 (variant A) -> no
            (
                [0.0, 2.0, 0.0, 0.0, 2.0, 2.0],
                1.0,
                [4],
                1,
                True,
                True,
                [2, 5],
            ),
            # signal before window is independent of window detection
            # df=[0.0, 2.0, 0.0, 2.0], cp=[4], max_delay=1
            # window=[4,4] (df length=4, so only idx4)
            # idx2: edge -> signal [2]
            # idx4: window, 2.0>1.0 -> signal [4]
            (
                [0.0, 2.0, 0.0, 2.0],
                1.0,
                [4],
                1,
                True,
                True,
                [2, 4],
            ),
            # edge resets correctly after window (variant A: prev=last window value)
            # df=[0.0, 0.0, 2.0, 0.0, 0.0, 2.0], cp=[3], max_delay=0
            # window={3}
            # idx3: window, 2.0>1.0 -> signal [3]
            # idx4: edge, prev=df[2]=2.0>=1.0 -> no (not rising)
            # idx5: edge, prev=0.0<1.0, 0.0<=1.0 -> no
            # idx6: edge, prev=0.0<1.0, 2.0>1.0 -> signal [6]
            (
                [0.0, 0.0, 2.0, 0.0, 0.0, 2.0],
                1.0,
                [3],
                0,
                True,
                True,
                [3, 6],
            ),
            # after window: value stays above threshold — no edge signal (variant A)
            # df=[0.0, 0.0, 2.0, 2.0, 2.0], cp=[3], max_delay=1
            # window=[3,4]
            # idx3: window, 2.0>1.0 -> signal [3]
            # idx4: window, 2.0>1.0 -> signal [4]
            # idx5: edge, prev=df[3]=2.0>=1.0 (variant A) -> no signal
            (
                [0.0, 0.0, 2.0, 2.0, 2.0],
                1.0,
                [3],
                1,
                True,
                True,
                [3, 4],
            ),
        ],
        ids=[
            "edge_before_and_point_inside_window",
            "signal_before_window_independent",
            "edge_resets_after_window",
            "after_window_above_no_edge_signal",
        ],
    )
    def test_mixed(
        self,
        values: list[float],
        threshold: float,
        change_points: list[int],
        max_delay: int,
        strict_edge: bool,
        strict_point: bool,
        expected: list[int],
    ) -> None:
        """
        Edge mode and delay windows must work correctly together.

        prev (for edge detection) tracks the last seen value including
        values inside the window (variant A). This means that if the
        detection function is above threshold at the end of a window,
        the first element after the window will NOT produce an edge signal.
        """
        policy: EventBasedPolicy = EventBasedPolicy(
            max_delay=max_delay,
            strict_edge=strict_edge,
            strict_point=strict_point,
        )
        df: UnivariateNumericArray = make_df(*values)
        result: list[int] = policy.apply(df, threshold, change_points)
        assert result == expected
