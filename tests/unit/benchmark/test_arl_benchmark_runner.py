# -*- coding: ascii -*-
"""
Tests for ARLBenchmarkRunner.

Covers initialization validation, _collect_runs behavior, run() output
structure and exact ARL values, max_runlength interaction, reset vs
noreset mode semantics, and reset behavior verification.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import math
from typing import Any, Literal

import pytest

from pysatl_cpd.analysis.labeled_data import LabeledData
from pysatl_cpd.benchmark.arl_benchmark_runner import ARLBenchmarkRunner
from pysatl_cpd.benchmark.metrics.online.arl_metric import ARLMetric
from pysatl_cpd.core.algorithm_entry import AlgorithmEntry
from pysatl_cpd.core.online.ionline_algorithm import OnlineAlgorithmConfiguration
from pysatl_cpd.core.online.online_cpd_solver import OnlineCpdSolver
from pysatl_cpd.core.online.online_detection_trace import OnlineDetectionTrace
from tests.mocks.algorithms.online.simple import MockOnlineAlgorithm


def _make_provider(
    length: int,
    change_points: list[int] | None = None,
    name: str = "test_data",
) -> LabeledData[float]:
    """Create a LabeledData provider with the given length and change points.

    Parameters
    ----------
    length : int
        Number of observations in the raw data.
    change_points : list[int] | None
        Known change point indices. Defaults to empty list.
    name : str
        Human-readable identifier for the provider.

    Returns
    -------
    LabeledData[float]
        Provider filled with constant 1.0 observations.
    """
    cp: list[int] = change_points if change_points is not None else []
    return LabeledData(raw_data=[1.0] * length, change_points=cp, name=name)


# ---------------------------------------------------------------------------
# 1. Initialization and validation
# ---------------------------------------------------------------------------
class TestARLBenchmarkRunnerInit:
    """Tests for ARLBenchmarkRunner.__init__ validation logic."""

    def test_raises_if_provider_has_change_points(self) -> None:
        """Should raise ValueError when a single provider has non-empty change_points."""
        algorithm: MockOnlineAlgorithm[float] = MockOnlineAlgorithm(name="algo", return_sequence=[0.0])
        entry = AlgorithmEntry(algorithm=algorithm, thresholds=[1.0])
        provider: LabeledData[float] = _make_provider(10, change_points=[5], name="bad")
        solver: OnlineCpdSolver = OnlineCpdSolver()

        with pytest.raises(ValueError):
            ARLBenchmarkRunner(
                entries=[entry],
                providers=[provider],
                solver=solver,
                mode="reset",
            )

    def test_raises_if_any_provider_has_change_points(self) -> None:
        """Should raise ValueError when at least one of several providers has change_points."""
        algorithm: MockOnlineAlgorithm[float] = MockOnlineAlgorithm(name="algo", return_sequence=[0.0])
        entry = AlgorithmEntry(algorithm=algorithm, thresholds=[1.0])
        ok_provider: LabeledData[float] = _make_provider(10, name="ok")
        bad_provider: LabeledData[float] = _make_provider(10, change_points=[3], name="bad")
        solver: OnlineCpdSolver = OnlineCpdSolver()

        with pytest.raises(ValueError):
            ARLBenchmarkRunner(
                entries=[entry],
                providers=[ok_provider, bad_provider],
                solver=solver,
                mode="reset",
            )

    def test_raises_if_any_provider_has_change_points_noreset_mode(self) -> None:
        """Validation should apply in noreset mode as well."""
        algorithm: MockOnlineAlgorithm[float] = MockOnlineAlgorithm(name="algo", return_sequence=[0.0])
        entry = AlgorithmEntry(algorithm=algorithm, thresholds=[1.0])
        bad_provider: LabeledData[float] = _make_provider(10, change_points=[3], name="bad")
        solver: OnlineCpdSolver = OnlineCpdSolver()

        with pytest.raises(ValueError):
            ARLBenchmarkRunner(
                entries=[entry],
                providers=[bad_provider],
                solver=solver,
                mode="noreset",
            )

    def test_valid_init_with_empty_change_points(self) -> None:
        """Should succeed when all providers have empty change_points."""
        algorithm: MockOnlineAlgorithm[float] = MockOnlineAlgorithm(name="algo", return_sequence=[0.0])
        entry = AlgorithmEntry(algorithm=algorithm, thresholds=[1.0])
        provider: LabeledData[float] = _make_provider(10, name="clean")
        solver: OnlineCpdSolver = OnlineCpdSolver()

        runner: ARLBenchmarkRunner[OnlineDetectionTrace[Any], LabeledData[float]] = ARLBenchmarkRunner(
            entries=[entry],
            providers=[provider],
            solver=solver,
            mode="reset",
        )
        assert runner is not None

    def test_metrics_contain_arl_metric(self) -> None:
        """Internal _metrics dict should contain 'arl' key with ARLMetric instance."""
        algorithm: MockOnlineAlgorithm[float] = MockOnlineAlgorithm(name="algo", return_sequence=[0.0])
        entry = AlgorithmEntry(algorithm=algorithm, thresholds=[1.0])
        provider: LabeledData[float] = _make_provider(10, name="data")
        solver: OnlineCpdSolver = OnlineCpdSolver()

        runner: ARLBenchmarkRunner[OnlineDetectionTrace[Any], LabeledData[float]] = ARLBenchmarkRunner(
            entries=[entry],
            providers=[provider],
            solver=solver,
            mode="reset",
        )
        assert "arl" in runner._metrics
        assert isinstance(runner._metrics["arl"], ARLMetric)

    @pytest.mark.parametrize("mode", ["reset", "noreset"])
    def test_accepts_both_modes(self, mode: Literal["reset", "noreset"]) -> None:
        """Constructor should accept both 'reset' and 'noreset' mode values."""
        algorithm: MockOnlineAlgorithm[float] = MockOnlineAlgorithm(name="algo", return_sequence=[0.0])
        entry = AlgorithmEntry(algorithm=algorithm, thresholds=[1.0])
        provider: LabeledData[float] = _make_provider(5, name="d")
        solver: OnlineCpdSolver = OnlineCpdSolver()

        runner: ARLBenchmarkRunner[OnlineDetectionTrace[Any], LabeledData[float]] = ARLBenchmarkRunner(
            entries=[entry],
            providers=[provider],
            solver=solver,
            mode=mode,
        )
        assert runner is not None


# ---------------------------------------------------------------------------
# 2. _collect_runs
# ---------------------------------------------------------------------------
class TestARLBenchmarkRunnerCollectRuns:
    """Tests for _collect_runs method."""

    def test_returns_correct_number_of_pairs_reset(self) -> None:
        """Should return one (trace, provider) pair per provider in reset mode."""
        algorithm: MockOnlineAlgorithm[float] = MockOnlineAlgorithm(name="algo", return_sequence=[0.0])
        entry = AlgorithmEntry(algorithm=algorithm, thresholds=[1.0])
        providers: list[LabeledData[float]] = [
            _make_provider(10, name="d1"),
            _make_provider(10, name="d2"),
            _make_provider(10, name="d3"),
        ]
        solver: OnlineCpdSolver = OnlineCpdSolver()

        runner: ARLBenchmarkRunner[OnlineDetectionTrace[Any], LabeledData[float]] = ARLBenchmarkRunner(
            entries=[entry],
            providers=providers,
            solver=solver,
            mode="reset",
        )
        runs: list[tuple[OnlineDetectionTrace[Any], LabeledData[float]]] = runner._collect_runs(entry, 1.0, providers)
        assert len(runs) == 3

    def test_returns_correct_number_of_pairs_noreset(self) -> None:
        """Should return one (trace, provider) pair per provider in noreset mode."""
        algorithm: MockOnlineAlgorithm[float] = MockOnlineAlgorithm(name="algo", return_sequence=[0.0])
        entry = AlgorithmEntry(algorithm=algorithm, thresholds=[1.0])
        providers: list[LabeledData[float]] = [
            _make_provider(10, name="d1"),
            _make_provider(10, name="d2"),
        ]
        solver: OnlineCpdSolver = OnlineCpdSolver()

        runner: ARLBenchmarkRunner[OnlineDetectionTrace[Any], LabeledData[float]] = ARLBenchmarkRunner(
            entries=[entry],
            providers=providers,
            solver=solver,
            mode="noreset",
        )
        runs: list[tuple[OnlineDetectionTrace[Any], LabeledData[float]]] = runner._collect_runs(entry, 1.0, providers)
        assert len(runs) == 2

    def test_pairs_traces_with_correct_providers(self) -> None:
        """Each trace should be paired with its corresponding provider by name."""
        algorithm: MockOnlineAlgorithm[float] = MockOnlineAlgorithm(name="algo", return_sequence=[0.0])
        entry = AlgorithmEntry(algorithm=algorithm, thresholds=[1.0])
        providers: list[LabeledData[float]] = [
            _make_provider(10, name="alpha"),
            _make_provider(15, name="beta"),
        ]
        solver: OnlineCpdSolver = OnlineCpdSolver()

        runner: ARLBenchmarkRunner[OnlineDetectionTrace[Any], LabeledData[float]] = ARLBenchmarkRunner(
            entries=[entry],
            providers=providers,
            solver=solver,
            mode="reset",
        )
        runs: list[tuple[OnlineDetectionTrace[Any], LabeledData[float]]] = runner._collect_runs(entry, 1.0, providers)
        names: list[str] = [prov.name for _, prov in runs]
        assert names == ["alpha", "beta"]

    def test_empty_providers_returns_empty_list(self) -> None:
        """Empty providers sequence should return empty list."""
        algorithm: MockOnlineAlgorithm[float] = MockOnlineAlgorithm(name="algo", return_sequence=[0.0])
        entry = AlgorithmEntry(algorithm=algorithm, thresholds=[1.0])
        solver: OnlineCpdSolver = OnlineCpdSolver()

        runner: ARLBenchmarkRunner[OnlineDetectionTrace[Any], LabeledData[float]] = ARLBenchmarkRunner(
            entries=[entry],
            providers=[_make_provider(10)],
            solver=solver,
            mode="reset",
        )
        runs: list[tuple[OnlineDetectionTrace[Any], LabeledData[float]]] = runner._collect_runs(entry, 1.0, [])
        assert runs == []


# ---------------------------------------------------------------------------
# 3. run() - structure and values
# ---------------------------------------------------------------------------
class TestARLBenchmarkRunnerRun:
    """Tests for run() output structure and ARL values."""

    def test_run_returns_correct_key_structure(self) -> None:
        """Result key should be (entry.full_name, algorithm.configuration)."""
        algorithm: MockOnlineAlgorithm[float] = MockOnlineAlgorithm(name="KeyAlgo", return_sequence=[0.0])
        entry = AlgorithmEntry(algorithm=algorithm, thresholds=[1.0])
        provider: LabeledData[float] = _make_provider(10)
        solver: OnlineCpdSolver = OnlineCpdSolver()

        runner: ARLBenchmarkRunner[OnlineDetectionTrace[Any], LabeledData[float]] = ARLBenchmarkRunner(
            entries=[entry],
            providers=[provider],
            solver=solver,
            mode="reset",
        )
        results: dict[
            tuple[str, OnlineAlgorithmConfiguration],
            list[tuple[float, dict[str, Any]]],
        ] = runner.run()

        assert len(results) == 1
        key: tuple[str, OnlineAlgorithmConfiguration] = next(iter(results))
        assert key[0] == entry.full_name
        assert key[1] == algorithm.configuration

    def test_run_arl_infinity_when_no_detections(self) -> None:
        """ARL should be inf when the detection function never exceeds the threshold."""
        algorithm: MockOnlineAlgorithm[float] = MockOnlineAlgorithm(name="QuietAlgo", return_sequence=[0.5])
        entry = AlgorithmEntry(algorithm=algorithm, thresholds=[3.0])
        provider: LabeledData[float] = _make_provider(20)
        solver: OnlineCpdSolver = OnlineCpdSolver()

        runner: ARLBenchmarkRunner[OnlineDetectionTrace[Any], LabeledData[float]] = ARLBenchmarkRunner(
            entries=[entry],
            providers=[provider],
            solver=solver,
            mode="reset",
        )
        results = runner.run()
        key: tuple[str, OnlineAlgorithmConfiguration] = next(iter(results))
        _, metrics = results[key][0]
        arl_value: float = metrics["arl"]

        assert math.isinf(arl_value)

    def test_run_arl_infinity_noreset_when_no_detections(self) -> None:
        """ARL should be inf in noreset mode when no threshold crossing occurs."""
        algorithm: MockOnlineAlgorithm[float] = MockOnlineAlgorithm(name="QuietAlgo", return_sequence=[0.5])
        entry = AlgorithmEntry(algorithm=algorithm, thresholds=[3.0])
        provider: LabeledData[float] = _make_provider(20)
        solver: OnlineCpdSolver = OnlineCpdSolver()

        runner: ARLBenchmarkRunner[OnlineDetectionTrace[Any], LabeledData[float]] = ARLBenchmarkRunner(
            entries=[entry],
            providers=[provider],
            solver=solver,
            mode="noreset",
        )
        results = runner.run()
        key: tuple[str, OnlineAlgorithmConfiguration] = next(iter(results))
        _, metrics = results[key][0]
        arl_value: float = metrics["arl"]

        assert math.isinf(arl_value)

    def test_run_multiple_thresholds(self) -> None:
        """Each threshold should produce its own entry with 'arl' metric."""
        algorithm: MockOnlineAlgorithm[float] = MockOnlineAlgorithm(name="Multi", return_sequence=[0.0, 2.0, 5.0])
        thresholds: list[float] = [1.0, 3.0, 10.0]
        entry = AlgorithmEntry(algorithm=algorithm, thresholds=thresholds)
        provider: LabeledData[float] = _make_provider(20)
        solver: OnlineCpdSolver = OnlineCpdSolver()

        runner: ARLBenchmarkRunner[OnlineDetectionTrace[Any], LabeledData[float]] = ARLBenchmarkRunner(
            entries=[entry],
            providers=[provider],
            solver=solver,
            mode="reset",
        )
        results = runner.run()
        key: tuple[str, OnlineAlgorithmConfiguration] = next(iter(results))
        entries_res: list[tuple[float, dict[str, Any]]] = results[key]

        assert len(entries_res) == 3
        recorded: list[float] = [t for t, _ in entries_res]
        assert recorded == thresholds
        for _, m in entries_res:
            assert "arl" in m

    def test_run_arl_aggregated_across_providers(self) -> None:
        """ARL should aggregate run lengths from all providers."""
        algorithm: MockOnlineAlgorithm[float] = MockOnlineAlgorithm(name="Agg", return_sequence=[0.0, 5.0])
        entry = AlgorithmEntry(algorithm=algorithm, thresholds=[3.0])
        providers: list[LabeledData[float]] = [
            _make_provider(4, name="p1"),
            _make_provider(6, name="p2"),
        ]
        solver: OnlineCpdSolver = OnlineCpdSolver()

        runner: ARLBenchmarkRunner[OnlineDetectionTrace[Any], LabeledData[float]] = ARLBenchmarkRunner(
            entries=[entry],
            providers=providers,
            solver=solver,
            mode="reset",
        )
        results = runner.run()
        key: tuple[str, OnlineAlgorithmConfiguration] = next(iter(results))
        _, metrics = results[key][0]
        arl_value: float = metrics["arl"]

        expected_arl: float = 8.0 / 5.0
        assert abs(arl_value - expected_arl) < 1e-10


# ---------------------------------------------------------------------------
# 4. Reset vs NoReset mode semantics
# ---------------------------------------------------------------------------
class TestARLBenchmarkRunnerModeSemantics:
    """Tests verifying different ARL behavior between reset and noreset modes."""

    def test_reset_vs_noreset_produce_different_arl(self) -> None:
        """Reset and noreset modes should produce different ARL values."""
        algorithm: MockOnlineAlgorithm[float] = MockOnlineAlgorithm(
            name="ModeTest",
            return_sequence=[0.0, 5.0, 0.0, 0.0],
        )
        entry = AlgorithmEntry(algorithm=algorithm, thresholds=[3.0])
        provider: LabeledData[float] = _make_provider(20)
        solver: OnlineCpdSolver = OnlineCpdSolver()

        runner_reset: ARLBenchmarkRunner[OnlineDetectionTrace[Any], LabeledData[float]] = ARLBenchmarkRunner(
            entries=[entry],
            providers=[provider],
            solver=solver,
            mode="reset",
        )
        res_reset = runner_reset.run()
        key_reset: tuple[str, OnlineAlgorithmConfiguration] = next(iter(res_reset))
        arl_reset: float = res_reset[key_reset][0][1]["arl"]

        runner_noreset: ARLBenchmarkRunner[OnlineDetectionTrace[Any], LabeledData[float]] = ARLBenchmarkRunner(
            entries=[entry],
            providers=[provider],
            solver=solver,
            mode="noreset",
        )
        res_noreset = runner_noreset.run()
        key_noreset: tuple[str, OnlineAlgorithmConfiguration] = next(iter(res_noreset))
        arl_noreset: float = res_noreset[key_noreset][0][1]["arl"]

        assert math.isfinite(arl_reset)
        assert math.isfinite(arl_noreset)
        assert arl_reset < arl_noreset

    def test_reset_mode_exact_arl_with_immediate_signal(self) -> None:
        """Verify exact ARL in reset mode."""
        algorithm: MockOnlineAlgorithm[float] = MockOnlineAlgorithm(
            name="Immediate",
            return_sequence=[0.0, 5.0, 0.0, 0.0],
        )
        entry = AlgorithmEntry(algorithm=algorithm, thresholds=[3.0])
        provider: LabeledData[float] = _make_provider(12)
        solver: OnlineCpdSolver = OnlineCpdSolver()

        runner: ARLBenchmarkRunner[OnlineDetectionTrace[Any], LabeledData[float]] = ARLBenchmarkRunner(
            entries=[entry],
            providers=[provider],
            solver=solver,
            mode="reset",
        )
        results = runner.run()
        key: tuple[str, OnlineAlgorithmConfiguration] = next(iter(results))
        _, metrics = results[key][0]
        arl_value: float = metrics["arl"]

        expected_arl: float = 11.0 / 6.0
        assert abs(arl_value - expected_arl) < 1e-10

    def test_noreset_mode_exact_arl_with_periodic_signal(self) -> None:
        """Verify exact ARL in noreset mode with periodic signal."""
        algorithm: MockOnlineAlgorithm[float] = MockOnlineAlgorithm(
            name="Periodic",
            return_sequence=[5.0, 0.0, 0.0, 0.0],
        )
        entry = AlgorithmEntry(algorithm=algorithm, thresholds=[3.0])
        provider: LabeledData[float] = _make_provider(12)
        solver: OnlineCpdSolver = OnlineCpdSolver()

        runner: ARLBenchmarkRunner[OnlineDetectionTrace[Any], LabeledData[float]] = ARLBenchmarkRunner(
            entries=[entry],
            providers=[provider],
            solver=solver,
            mode="noreset",
        )
        results = runner.run()
        key: tuple[str, OnlineAlgorithmConfiguration] = next(iter(results))
        _, metrics = results[key][0]
        arl_value: float = metrics["arl"]

        expected_arl: float = 9.0 / 3.0
        assert abs(arl_value - expected_arl) < 1e-10

    def test_noreset_lower_threshold_shorter_arl(self) -> None:
        """Lower threshold in noreset mode should detect more, producing shorter ARL."""
        algorithm: MockOnlineAlgorithm[float] = MockOnlineAlgorithm(
            name="Gradual",
            return_sequence=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        )
        entry = AlgorithmEntry(algorithm=algorithm, thresholds=[1.5, 4.5])
        provider: LabeledData[float] = _make_provider(24)
        solver: OnlineCpdSolver = OnlineCpdSolver()

        runner: ARLBenchmarkRunner[OnlineDetectionTrace[Any], LabeledData[float]] = ARLBenchmarkRunner(
            entries=[entry],
            providers=[provider],
            solver=solver,
            mode="noreset",
        )
        results = runner.run()
        key: tuple[str, OnlineAlgorithmConfiguration] = next(iter(results))
        entries_res: list[tuple[float, dict[str, Any]]] = results[key]

        arl_low: float = entries_res[0][1]["arl"]
        arl_high: float = entries_res[1][1]["arl"]

        assert math.isfinite(arl_low)
        assert math.isfinite(arl_high)
        assert arl_low < arl_high

    def test_noreset_same_arl_for_same_threshold_different_runs(self) -> None:
        """In noreset mode, same algorithm+provider+threshold should give same ARL."""
        algorithm: MockOnlineAlgorithm[float] = MockOnlineAlgorithm(
            name="Stable",
            return_sequence=[0.0, 0.0, 5.0],
        )
        entry = AlgorithmEntry(algorithm=algorithm, thresholds=[3.0])
        provider: LabeledData[float] = _make_provider(15)
        solver: OnlineCpdSolver = OnlineCpdSolver()

        runner1: ARLBenchmarkRunner[OnlineDetectionTrace[Any], LabeledData[float]] = ARLBenchmarkRunner(
            entries=[entry],
            providers=[provider],
            solver=solver,
            mode="noreset",
        )
        runner2: ARLBenchmarkRunner[OnlineDetectionTrace[Any], LabeledData[float]] = ARLBenchmarkRunner(
            entries=[entry],
            providers=[provider],
            solver=solver,
            mode="noreset",
        )

        res1 = runner1.run()
        res2 = runner2.run()

        key1: tuple[str, OnlineAlgorithmConfiguration] = next(iter(res1))
        key2: tuple[str, OnlineAlgorithmConfiguration] = next(iter(res2))
        arl1: float = res1[key1][0][1]["arl"]
        arl2: float = res2[key2][0][1]["arl"]

        assert arl1 == arl2


# ---------------------------------------------------------------------------
# 5. max_runlength - forced resets
# ---------------------------------------------------------------------------
class TestARLBenchmarkRunnerMaxRunlength:
    """Tests for ARL interaction with solver max_runlength (forced change points)."""

    def test_forced_detections_produce_finite_arl(self) -> None:
        """Forced detections via max_runlength give finite ARL with unreachable threshold."""
        algorithm: MockOnlineAlgorithm[float] = MockOnlineAlgorithm(name="Silent", return_sequence=[0.0])
        entry = AlgorithmEntry(algorithm=algorithm, thresholds=[100.0])
        provider: LabeledData[float] = _make_provider(18)
        solver: OnlineCpdSolver = OnlineCpdSolver(max_runlength=5)

        runner: ARLBenchmarkRunner[OnlineDetectionTrace[Any], LabeledData[float]] = ARLBenchmarkRunner(
            entries=[entry],
            providers=[provider],
            solver=solver,
            mode="reset",
        )
        results = runner.run()
        key: tuple[str, OnlineAlgorithmConfiguration] = next(iter(results))
        _, metrics = results[key][0]
        arl_value: float = metrics["arl"]

        assert math.isfinite(arl_value)
        assert arl_value > 0

    def test_exact_arl_with_max_runlength(self) -> None:
        """Verify exact ARL with max_runlength=5 on 18 observations."""
        algorithm: MockOnlineAlgorithm[float] = MockOnlineAlgorithm(name="Silent", return_sequence=[0.0])
        entry = AlgorithmEntry(algorithm=algorithm, thresholds=[100.0])
        provider: LabeledData[float] = _make_provider(18)
        solver: OnlineCpdSolver = OnlineCpdSolver(max_runlength=5)

        runner: ARLBenchmarkRunner[OnlineDetectionTrace[Any], LabeledData[float]] = ARLBenchmarkRunner(
            entries=[entry],
            providers=[provider],
            solver=solver,
            mode="reset",
        )
        results = runner.run()
        key: tuple[str, OnlineAlgorithmConfiguration] = next(iter(results))
        _, metrics = results[key][0]
        arl_value: float = metrics["arl"]

        expected_arl: float = 17.0 / 3.0
        assert abs(arl_value - expected_arl) < 1e-10

    def test_signal_before_forced_prevents_forced(self) -> None:
        """Signal detections happening before max_runlength prevent forced detections."""
        algorithm: MockOnlineAlgorithm[float] = MockOnlineAlgorithm(name="Fast", return_sequence=[0.0, 0.0, 5.0])
        entry = AlgorithmEntry(algorithm=algorithm, thresholds=[3.0])
        provider: LabeledData[float] = _make_provider(18)
        solver: OnlineCpdSolver = OnlineCpdSolver(max_runlength=10)

        runner: ARLBenchmarkRunner[OnlineDetectionTrace[Any], LabeledData[float]] = ARLBenchmarkRunner(
            entries=[entry],
            providers=[provider],
            solver=solver,
            mode="reset",
        )
        results = runner.run()
        key: tuple[str, OnlineAlgorithmConfiguration] = next(iter(results))
        _, metrics = results[key][0]
        arl_value: float = metrics["arl"]

        expected_arl: float = 17.0 / 6.0
        assert abs(arl_value - expected_arl) < 1e-10

    def test_max_runlength_noreset_inf_trace_still_forces(self) -> None:
        """In noreset mode, max_runlength affects the inf-trace run."""
        algorithm: MockOnlineAlgorithm[float] = MockOnlineAlgorithm(name="Silent", return_sequence=[0.0])
        entry = AlgorithmEntry(algorithm=algorithm, thresholds=[100.0])
        provider: LabeledData[float] = _make_provider(15)
        solver_forced: OnlineCpdSolver = OnlineCpdSolver(max_runlength=4)
        solver_no_forced: OnlineCpdSolver = OnlineCpdSolver()

        runner_forced: ARLBenchmarkRunner[OnlineDetectionTrace[Any], LabeledData[float]] = ARLBenchmarkRunner(
            entries=[entry],
            providers=[provider],
            solver=solver_forced,
            mode="reset",
        )
        runner_no_forced: ARLBenchmarkRunner[OnlineDetectionTrace[Any], LabeledData[float]] = ARLBenchmarkRunner(
            entries=[entry],
            providers=[provider],
            solver=solver_no_forced,
            mode="reset",
        )

        res_forced = runner_forced.run()
        res_no_forced = runner_no_forced.run()

        key_f: tuple[str, OnlineAlgorithmConfiguration] = next(iter(res_forced))
        key_nf: tuple[str, OnlineAlgorithmConfiguration] = next(iter(res_no_forced))

        arl_forced: float = res_forced[key_f][0][1]["arl"]
        arl_no_forced: float = res_no_forced[key_nf][0][1]["arl"]

        assert math.isfinite(arl_forced)
        assert math.isinf(arl_no_forced)


# ---------------------------------------------------------------------------
# 6. Reset behavior - sequence restart verification
# ---------------------------------------------------------------------------
class TestARLBenchmarkRunnerResetBehavior:
    """Tests verifying that algorithm reset after each detection affects ARL."""

    def test_reset_restarts_return_sequence(self) -> None:
        """After reset, return_sequence restarts producing periodic detections."""
        algorithm: MockOnlineAlgorithm[float] = MockOnlineAlgorithm(name="Reset", return_sequence=[0.0, 5.0])
        entry = AlgorithmEntry(algorithm=algorithm, thresholds=[3.0])
        provider: LabeledData[float] = _make_provider(8)
        solver: OnlineCpdSolver = OnlineCpdSolver()

        runner: ARLBenchmarkRunner[OnlineDetectionTrace[Any], LabeledData[float]] = ARLBenchmarkRunner(
            entries=[entry],
            providers=[provider],
            solver=solver,
            mode="reset",
        )
        results = runner.run()
        key: tuple[str, OnlineAlgorithmConfiguration] = next(iter(results))
        _, metrics = results[key][0]
        arl_value: float = metrics["arl"]

        expected_arl: float = 7.0 / 4.0
        assert abs(arl_value - expected_arl) < 1e-10

    def test_reset_restarts_learning_period(self) -> None:
        """Reset re-enters learning period, creating longer gaps between detections."""
        algorithm: MockOnlineAlgorithm[float] = MockOnlineAlgorithm(
            name="Learn",
            return_sequence=[5.0],
            learning_period_size=2,
        )
        entry = AlgorithmEntry(algorithm=algorithm, thresholds=[3.0])
        provider: LabeledData[float] = _make_provider(9)
        solver: OnlineCpdSolver = OnlineCpdSolver()

        runner: ARLBenchmarkRunner[OnlineDetectionTrace[Any], LabeledData[float]] = ARLBenchmarkRunner(
            entries=[entry],
            providers=[provider],
            solver=solver,
            mode="reset",
        )
        results = runner.run()
        key: tuple[str, OnlineAlgorithmConfiguration] = next(iter(results))
        _, metrics = results[key][0]
        arl_value: float = metrics["arl"]

        expected_arl: float = 8.0 / 3.0
        assert abs(arl_value - expected_arl) < 1e-10

    def test_lower_threshold_produces_shorter_arl_reset(self) -> None:
        """Lower threshold detects more often, resulting in shorter ARL in reset mode."""
        algorithm: MockOnlineAlgorithm[float] = MockOnlineAlgorithm(
            name="Gradual",
            return_sequence=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        )
        entry = AlgorithmEntry(algorithm=algorithm, thresholds=[1.5, 4.5])
        provider: LabeledData[float] = _make_provider(30)
        solver: OnlineCpdSolver = OnlineCpdSolver()

        runner: ARLBenchmarkRunner[OnlineDetectionTrace[Any], LabeledData[float]] = ARLBenchmarkRunner(
            entries=[entry],
            providers=[provider],
            solver=solver,
            mode="reset",
        )
        results = runner.run()
        key: tuple[str, OnlineAlgorithmConfiguration] = next(iter(results))
        entries_res: list[tuple[float, dict[str, Any]]] = results[key]

        arl_low: float = entries_res[0][1]["arl"]
        arl_high: float = entries_res[1][1]["arl"]

        assert math.isfinite(arl_low)
        assert math.isfinite(arl_high)
        assert arl_low < arl_high
