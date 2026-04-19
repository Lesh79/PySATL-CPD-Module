# -*- coding: ascii -*-
"""
Tests for BenchmarkExecutor and BenchmarkRecord.

Covers result count for various combinations, trace content verification,
record metadata, and disk caching behavior.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import csv
from pathlib import Path
from typing import Any

import numpy as np

from pysatl_cpd.analysis.labeled_data import LabeledData
from pysatl_cpd.benchmark.core.benchmark_executor import (
    BenchmarkExecutor,
    BenchmarkRecord,
)
from pysatl_cpd.core.algorithm_entry import AlgorithmEntry
from pysatl_cpd.core.online.online_cpd_solver import OnlineCpdSolver
from pysatl_cpd.core.online.online_detection_trace import OnlineDetectionTrace
from tests.mocks.algorithms.online.simple import MockOnlineAlgorithm
from tests.mocks.core.data_transformers.data_transformer import MockDataTransformer


def _make_provider(
    length: int,
    name: str = "test_data",
) -> LabeledData[float]:
    """Create a LabeledData provider with constant observations.

    Parameters
    ----------
    length : int
        Number of observations.
    name : str
        Provider identifier.

    Returns
    -------
    LabeledData[float]
        Provider with ``length`` observations of 1.0 and no change points.
    """
    return LabeledData(raw_data=[1.0] * length, change_points=[], name=name)


# ---------------------------------------------------------------------------
# 1. BenchmarkRecord
# ---------------------------------------------------------------------------
class TestBenchmarkRecord:
    """Tests for BenchmarkRecord dataclass."""

    def test_key_returns_correct_tuple(self) -> None:
        """Key property should return (algorithm, config_hash, data, threshold)."""
        record: BenchmarkRecord = BenchmarkRecord(
            algorithm="TestAlgo",
            configuration_hash=42,
            data="dataset",
            threshold=2.5,
            trace_path="/tmp/trace.pkl",
        )
        expected: tuple[str, int, str, float] = ("TestAlgo", 42, "dataset", 2.5)
        assert record.key == expected

    def test_default_trace_path_is_none(self) -> None:
        """trace_path should default to None when not provided."""
        record: BenchmarkRecord = BenchmarkRecord(
            algorithm="A",
            configuration_hash=0,
            data="d",
            threshold=1.0,
        )
        assert record.trace_path is None


# ---------------------------------------------------------------------------
# 2. Basic execution - result counts
# ---------------------------------------------------------------------------
class TestBenchmarkExecutorBasic:
    """Tests for correct number of results across combinations."""

    def test_single_combination(self) -> None:
        """1 algorithm x 1 threshold x 1 provider -> 1 result."""
        algo = MockOnlineAlgorithm[float](name="A", return_sequence=[0.0])
        entry = AlgorithmEntry(algorithm=algo, thresholds=[1.0])
        provider: LabeledData[float] = _make_provider(5)
        solver: OnlineCpdSolver = OnlineCpdSolver()

        executor: BenchmarkExecutor[float] = BenchmarkExecutor(
            entries=[entry],
            providers=[provider],
            solver=solver,
        )
        results: list[tuple[BenchmarkRecord, OnlineDetectionTrace[Any]]] = executor.execute()
        assert len(results) == 1

    def test_multiple_thresholds(self) -> None:
        """1 algorithm x 3 thresholds x 1 provider -> 3 results."""
        algo = MockOnlineAlgorithm[float](name="A", return_sequence=[0.0])
        entry = AlgorithmEntry(algorithm=algo, thresholds=[1.0, 2.0, 3.0])
        provider: LabeledData[float] = _make_provider(5)
        solver: OnlineCpdSolver = OnlineCpdSolver()

        executor: BenchmarkExecutor[float] = BenchmarkExecutor(
            entries=[entry],
            providers=[provider],
            solver=solver,
        )
        results = executor.execute()
        assert len(results) == 3

    def test_multiple_providers(self) -> None:
        """1 algorithm x 1 threshold x 3 providers -> 3 results."""
        algo = MockOnlineAlgorithm[float](name="A", return_sequence=[0.0])
        entry = AlgorithmEntry(algorithm=algo, thresholds=[1.0])
        providers: list[LabeledData[float]] = [
            _make_provider(5, name="p1"),
            _make_provider(5, name="p2"),
            _make_provider(5, name="p3"),
        ]
        solver: OnlineCpdSolver = OnlineCpdSolver()

        executor: BenchmarkExecutor[float] = BenchmarkExecutor(
            entries=[entry],
            providers=providers,
            solver=solver,
        )
        results = executor.execute()
        assert len(results) == 3

    def test_multiple_algorithms(self) -> None:
        """2 algorithms x 1 threshold each x 1 provider -> 2 results."""
        algo1 = MockOnlineAlgorithm[float](name="A1", return_sequence=[0.0])
        algo2 = MockOnlineAlgorithm[float](name="A2", return_sequence=[1.0])
        entries = [
            AlgorithmEntry(algorithm=algo1, thresholds=[1.0]),
            AlgorithmEntry(algorithm=algo2, thresholds=[2.0]),
        ]
        provider: LabeledData[float] = _make_provider(5)
        solver: OnlineCpdSolver = OnlineCpdSolver()

        executor: BenchmarkExecutor[float] = BenchmarkExecutor(
            entries=entries,
            providers=[provider],
            solver=solver,
        )
        results = executor.execute()
        assert len(results) == 2

    def test_cartesian_product(self) -> None:
        """2 algorithms x 2 thresholds x 2 providers -> 8 results."""
        algo1 = MockOnlineAlgorithm[float](name="A1", return_sequence=[0.0])
        algo2 = MockOnlineAlgorithm[float](name="A2", return_sequence=[0.0])
        entries = [
            AlgorithmEntry(algorithm=algo1, thresholds=[1.0, 2.0]),
            AlgorithmEntry(algorithm=algo2, thresholds=[3.0, 4.0]),
        ]
        providers: list[LabeledData[float]] = [
            _make_provider(5, name="p1"),
            _make_provider(5, name="p2"),
        ]
        solver: OnlineCpdSolver = OnlineCpdSolver()

        executor: BenchmarkExecutor[float] = BenchmarkExecutor(
            entries=entries,
            providers=providers,
            solver=solver,
        )
        results = executor.execute()
        assert len(results) == 8

    def test_empty_algorithms(self) -> None:
        """No algorithms -> empty results."""
        provider: LabeledData[float] = _make_provider(5)
        solver: OnlineCpdSolver = OnlineCpdSolver()

        executor: BenchmarkExecutor[float] = BenchmarkExecutor(
            entries=[],
            providers=[provider],
            solver=solver,
        )
        results = executor.execute()
        assert results == []

    def test_empty_providers(self) -> None:
        """No providers -> empty results."""
        algo = MockOnlineAlgorithm[float](name="A", return_sequence=[0.0])
        entry = AlgorithmEntry(algorithm=algo, thresholds=[1.0])
        solver: OnlineCpdSolver = OnlineCpdSolver()

        executor: BenchmarkExecutor[float] = BenchmarkExecutor(
            entries=[entry],
            providers=[],
            solver=solver,
        )
        results = executor.execute()
        assert results == []

    def test_empty_thresholds(self) -> None:
        """Algorithm with no thresholds -> no results for that algorithm."""
        algo = MockOnlineAlgorithm[float](name="A", return_sequence=[0.0])
        entry = AlgorithmEntry(algorithm=algo, thresholds=[])
        provider: LabeledData[float] = _make_provider(5)
        solver: OnlineCpdSolver = OnlineCpdSolver()

        executor: BenchmarkExecutor[float] = BenchmarkExecutor(
            entries=[entry],
            providers=[provider],
            solver=solver,
        )
        results = executor.execute()
        assert results == []


# ---------------------------------------------------------------------------
# 3. Trace content
# ---------------------------------------------------------------------------
class TestBenchmarkExecutorTraceContent:
    """Tests for detection trace correctness."""

    def test_detections_at_correct_steps(self) -> None:
        """Verify detected change points match expected steps."""
        algo = MockOnlineAlgorithm[float](name="A", return_sequence=[0.0, 0.0, 5.0])
        entry = AlgorithmEntry(algorithm=algo, thresholds=[3.0])
        provider: LabeledData[float] = _make_provider(6)
        solver: OnlineCpdSolver = OnlineCpdSolver()

        executor: BenchmarkExecutor[float] = BenchmarkExecutor(
            entries=[entry],
            providers=[provider],
            solver=solver,
        )
        results = executor.execute()
        trace: OnlineDetectionTrace[Any] = results[0][1]

        assert list(trace.detected_change_points) == [2, 5]

    def test_no_detections_with_high_threshold(self) -> None:
        """No detections when threshold is unreachable."""
        algo = MockOnlineAlgorithm[float](name="A", return_sequence=[5.0])
        entry = AlgorithmEntry(algorithm=algo, thresholds=[100.0])
        provider: LabeledData[float] = _make_provider(10)
        solver: OnlineCpdSolver = OnlineCpdSolver()

        executor: BenchmarkExecutor[float] = BenchmarkExecutor(
            entries=[entry],
            providers=[provider],
            solver=solver,
        )
        results = executor.execute()
        trace: OnlineDetectionTrace[Any] = results[0][1]

        assert list(trace.detected_change_points) == []

    def test_trace_algorithm_name(self) -> None:
        """Trace should carry the entry.full_name as algorithm_name."""
        algo = MockOnlineAlgorithm[float](name="NamedAlgo", return_sequence=[0.0])
        entry = AlgorithmEntry(algorithm=algo, thresholds=[1.0])
        provider: LabeledData[float] = _make_provider(5)
        solver: OnlineCpdSolver = OnlineCpdSolver()

        executor: BenchmarkExecutor[float] = BenchmarkExecutor(
            entries=[entry],
            providers=[provider],
            solver=solver,
        )
        results = executor.execute()
        trace: OnlineDetectionTrace[Any] = results[0][1]

        assert trace.algorithm_name == entry.full_name

    def test_detection_function_values(self) -> None:
        """Detection function array should contain correct statistic values."""
        algo = MockOnlineAlgorithm[float](name="A", return_sequence=[1.0, 2.0, 3.0])
        entry = AlgorithmEntry(algorithm=algo, thresholds=[float("inf")])
        provider: LabeledData[float] = _make_provider(6)
        solver: OnlineCpdSolver = OnlineCpdSolver()

        executor: BenchmarkExecutor[float] = BenchmarkExecutor(
            entries=[entry],
            providers=[provider],
            solver=solver,
        )
        results = executor.execute()
        trace: OnlineDetectionTrace[Any] = results[0][1]

        expected: list[float] = [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
        np.testing.assert_array_almost_equal(trace.detection_function, expected)


# ---------------------------------------------------------------------------
# 4. Record content
# ---------------------------------------------------------------------------
class TestBenchmarkExecutorRecordContent:
    """Tests for BenchmarkRecord fields in executor output."""

    def test_record_fields_match_input(self) -> None:
        """Record fields should match the algorithm, provider, and threshold."""
        algo = MockOnlineAlgorithm[float](name="RecAlgo", return_sequence=[0.0])
        entry = AlgorithmEntry(algorithm=algo, thresholds=[7.5])
        provider: LabeledData[float] = _make_provider(5, name="my_data")
        solver: OnlineCpdSolver = OnlineCpdSolver()

        executor: BenchmarkExecutor[float] = BenchmarkExecutor(
            entries=[entry],
            providers=[provider],
            solver=solver,
        )
        results = executor.execute()
        record: BenchmarkRecord = results[0][0]

        assert record.algorithm == entry.full_name
        assert record.configuration_hash == entry.full_hash
        assert record.data == "my_data"
        assert record.threshold == 7.5

    def test_record_trace_path_none_without_dump_dir(self) -> None:
        """trace_path should be None when dump_dir is not set."""
        algo = MockOnlineAlgorithm[float](name="A", return_sequence=[0.0])
        entry = AlgorithmEntry(algorithm=algo, thresholds=[1.0])
        provider: LabeledData[float] = _make_provider(5)
        solver: OnlineCpdSolver = OnlineCpdSolver()

        executor: BenchmarkExecutor[float] = BenchmarkExecutor(
            entries=[entry],
            providers=[provider],
            solver=solver,
            dump_dir=None,
        )
        results = executor.execute()
        record: BenchmarkRecord = results[0][0]

        assert record.trace_path is None

    def test_record_trace_path_set_with_dump_dir(self, tmp_path: Path) -> None:
        """trace_path should point to an existing pickle file when dump_dir is set."""
        algo = MockOnlineAlgorithm[float](name="A", return_sequence=[0.0])
        entry = AlgorithmEntry(algorithm=algo, thresholds=[1.0])
        provider: LabeledData[float] = _make_provider(5, name="data")
        solver: OnlineCpdSolver = OnlineCpdSolver()

        executor: BenchmarkExecutor[float] = BenchmarkExecutor(
            entries=[entry],
            providers=[provider],
            solver=solver,
            dump_dir=tmp_path,
        )
        results = executor.execute()
        record: BenchmarkRecord = results[0][0]

        assert record.trace_path is not None
        assert Path(record.trace_path).exists()
        assert record.trace_path.endswith(".pkl")


# ---------------------------------------------------------------------------
# 5. Caching
# ---------------------------------------------------------------------------
class TestBenchmarkExecutorCaching:
    """Tests for disk caching via CSV registry and pickle files."""

    def test_creates_registry_and_pickle_files(self, tmp_path: Path) -> None:
        """Execute with dump_dir should create registry CSV and pickle file(s)."""
        algo = MockOnlineAlgorithm[float](name="A", return_sequence=[0.0])
        entry = AlgorithmEntry(algorithm=algo, thresholds=[1.0])
        provider: LabeledData[float] = _make_provider(5, name="data")
        solver: OnlineCpdSolver = OnlineCpdSolver()

        executor: BenchmarkExecutor[float] = BenchmarkExecutor(
            entries=[entry],
            providers=[provider],
            solver=solver,
            dump_dir=tmp_path,
        )
        executor.execute()

        registry_path: Path = tmp_path / "benchmark_registry.csv"
        assert registry_path.exists()

        pkl_files: list[Path] = list(tmp_path.glob("*.pkl"))
        assert len(pkl_files) == 1

    def test_cache_prevents_reprocessing(self, tmp_path: Path) -> None:
        """Second execute should load from cache without calling solver."""
        algo = MockOnlineAlgorithm[float](name="A", return_sequence=[0.0])
        entry = AlgorithmEntry(algorithm=algo, thresholds=[1.0])
        provider: LabeledData[float] = _make_provider(5, name="data")
        solver: OnlineCpdSolver = OnlineCpdSolver()

        executor1: BenchmarkExecutor[float] = BenchmarkExecutor(
            entries=[entry],
            providers=[provider],
            solver=solver,
            dump_dir=tmp_path,
        )
        executor1.execute()
        history_after_first: int = len(algo.get_call_history())
        assert history_after_first == 5

        executor2: BenchmarkExecutor[float] = BenchmarkExecutor(
            entries=[entry],
            providers=[provider],
            solver=solver,
            dump_dir=tmp_path,
        )
        executor2.execute()
        history_after_second: int = len(algo.get_call_history())

        assert history_after_second == history_after_first

    def test_cached_trace_matches_original(self, tmp_path: Path) -> None:
        """Trace loaded from cache should have identical detected_change_points."""
        algo = MockOnlineAlgorithm[float](name="A", return_sequence=[0.0, 0.0, 5.0])
        entry = AlgorithmEntry(algorithm=algo, thresholds=[3.0])
        provider: LabeledData[float] = _make_provider(6, name="data")
        solver: OnlineCpdSolver = OnlineCpdSolver()

        executor1: BenchmarkExecutor[float] = BenchmarkExecutor(
            entries=[entry],
            providers=[provider],
            solver=solver,
            dump_dir=tmp_path,
        )
        results1 = executor1.execute()

        executor2: BenchmarkExecutor[float] = BenchmarkExecutor(
            entries=[entry],
            providers=[provider],
            solver=solver,
            dump_dir=tmp_path,
        )
        results2 = executor2.execute()

        trace1: OnlineDetectionTrace[Any] = results1[0][1]
        trace2: OnlineDetectionTrace[Any] = results2[0][1]

        assert list(trace1.detected_change_points) == list(trace2.detected_change_points)
        assert trace1.algorithm_name == trace2.algorithm_name
        np.testing.assert_array_almost_equal(trace1.detection_function, trace2.detection_function)

    def test_registry_csv_has_correct_structure(self, tmp_path: Path) -> None:
        """Registry CSV should have expected columns and matching row data."""
        algo = MockOnlineAlgorithm[float](name="CsvAlgo", return_sequence=[0.0])
        entry = AlgorithmEntry(algorithm=algo, thresholds=[2.5])
        provider: LabeledData[float] = _make_provider(5, name="csv_data")
        solver: OnlineCpdSolver = OnlineCpdSolver()

        executor: BenchmarkExecutor[float] = BenchmarkExecutor(
            entries=[entry],
            providers=[provider],
            solver=solver,
            dump_dir=tmp_path,
        )
        executor.execute()

        registry_path: Path = tmp_path / "benchmark_registry.csv"
        with open(registry_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows: list[dict[str, str]] = list(reader)

        assert len(rows) == 1
        row: dict[str, str] = rows[0]

        expected_columns: set[str] = {
            "algorithm",
            "configuration_hash",
            "data",
            "threshold",
            "trace_path",
        }
        assert set(row.keys()) == expected_columns
        assert row["algorithm"] == entry.full_name
        assert row["data"] == "csv_data"
        assert float(row["threshold"]) == 2.5
        assert row["trace_path"] != ""

    def test_inf_threshold_in_pickle_filename(self, tmp_path: Path) -> None:
        """Pickle filename for infinite threshold should contain 'inf'."""
        algo = MockOnlineAlgorithm[float](name="A", return_sequence=[0.0])
        entry = AlgorithmEntry(algorithm=algo, thresholds=[float("inf")])
        provider: LabeledData[float] = _make_provider(5, name="data")
        solver: OnlineCpdSolver = OnlineCpdSolver()

        executor: BenchmarkExecutor[float] = BenchmarkExecutor(
            entries=[entry],
            providers=[provider],
            solver=solver,
            dump_dir=tmp_path,
        )
        executor.execute()

        pkl_files: list[Path] = list(tmp_path.glob("*.pkl"))
        assert len(pkl_files) == 1
        assert "inf" in pkl_files[0].name

    def test_multiple_thresholds_create_separate_pickle_files(self, tmp_path: Path) -> None:
        """Each threshold should produce its own pickle file."""
        algo = MockOnlineAlgorithm[float](name="A", return_sequence=[0.0])
        entry = AlgorithmEntry(algorithm=algo, thresholds=[1.0, 2.0, 3.0])
        provider: LabeledData[float] = _make_provider(5, name="data")
        solver: OnlineCpdSolver = OnlineCpdSolver()

        executor: BenchmarkExecutor[float] = BenchmarkExecutor(
            entries=[entry],
            providers=[provider],
            solver=solver,
            dump_dir=tmp_path,
        )
        executor.execute()

        pkl_files: list[Path] = list(tmp_path.glob("*.pkl"))
        assert len(pkl_files) == 3

        registry_path: Path = tmp_path / "benchmark_registry.csv"
        with open(registry_path, encoding="utf-8") as f:
            rows: list[dict[str, str]] = list(csv.DictReader(f))
        assert len(rows) == 3


# ---------------------------------------------------------------------------
# 6. Data Transformers
# ---------------------------------------------------------------------------
class TestBenchmarkExecutorTransformers:
    """Tests for the DataTransformer integration in BenchmarkExecutor."""

    def test_transformer_modifies_data_passed_to_algorithm(self) -> None:
        """Executor should pass transformed data, not raw data, to the solver."""
        algo = MockOnlineAlgorithm[float](name="A", return_sequence=[0.0])
        transformer = MockDataTransformer(name="T1", add_value=5.0)
        entry = AlgorithmEntry(algorithm=algo, thresholds=[1.0], transformer=transformer)

        # Original provider with zeros
        provider: LabeledData[float] = LabeledData(raw_data=[0.0, 0.0, 0.0], change_points=[], name="data")
        solver: OnlineCpdSolver = OnlineCpdSolver()

        executor: BenchmarkExecutor[float] = BenchmarkExecutor(
            entries=[entry],
            providers=[provider],
            solver=solver,
        )
        executor.execute()

        # The algorithm should have received [5.0, 5.0, 5.0]
        history: list[float] = algo.get_call_history()
        assert history == [5.0, 5.0, 5.0]

    def test_record_metadata_uses_transformer_name_and_hash(self) -> None:
        """Benchmark record should inherit the full name and hash from the Entry."""
        algo = MockOnlineAlgorithm[float](name="BaseAlgo", return_sequence=[0.0])
        transformer = MockDataTransformer(name="MyTF", add_value=1.0)
        entry = AlgorithmEntry(algorithm=algo, thresholds=[1.0], transformer=transformer)

        provider: LabeledData[float] = _make_provider(3, name="d1")
        solver: OnlineCpdSolver = OnlineCpdSolver()

        executor: BenchmarkExecutor[float] = BenchmarkExecutor(
            entries=[entry],
            providers=[provider],
            solver=solver,
        )
        results = executor.execute()
        record: BenchmarkRecord = results[0][0]

        # Name should be combined
        assert record.algorithm == "BaseAlgo_MyTF"
        assert record.algorithm == entry.full_name

        # Hash should match the entry's composite hash
        assert record.configuration_hash == entry.full_hash

    def test_caching_separates_different_transformers(self, tmp_path: Path) -> None:
        """Using the same algorithm but different transformers should create separate cache records."""
        algo = MockOnlineAlgorithm[float](name="A", return_sequence=[0.0])

        entry_clean = AlgorithmEntry(algorithm=algo, thresholds=[1.0], transformer=None)
        entry_transformed = AlgorithmEntry(
            algorithm=algo, thresholds=[1.0], transformer=MockDataTransformer(name="T1", add_value=2.0)
        )

        provider: LabeledData[float] = _make_provider(3, name="data")
        solver: OnlineCpdSolver = OnlineCpdSolver()

        executor: BenchmarkExecutor[float] = BenchmarkExecutor(
            entries=[entry_clean, entry_transformed],
            providers=[provider],
            solver=solver,
            dump_dir=tmp_path,
        )
        executor.execute()

        # Should produce two distinct pickle files
        pkl_files: list[Path] = list(tmp_path.glob("*.pkl"))
        assert len(pkl_files) == 2

        # Names of the files should reflect the different algorithm representations
        file_names: str = " ".join(f.name for f in pkl_files)
        assert "A_" in file_names
        assert "A_T1_" in file_names

    def test_transformer_is_called_even_on_cache_hit(self, tmp_path: Path) -> None:
        """Transformer should be applied before checking cache, incrementing its call count."""
        transformer = MockDataTransformer(name="T1", add_value=1.0)
        provider: LabeledData[float] = _make_provider(3, name="data")
        solver: OnlineCpdSolver = OnlineCpdSolver()

        # First run to populate cache
        algo1 = MockOnlineAlgorithm[float](name="A", return_sequence=[0.0])
        entry1 = AlgorithmEntry(algorithm=algo1, thresholds=[1.0], transformer=transformer)
        exec1: BenchmarkExecutor[float] = BenchmarkExecutor([entry1], [provider], solver, tmp_path)
        exec1.execute()

        assert transformer.call_count == 1
        assert len(algo1.get_call_history()) == 3

        # Second run with cache hit (using a fresh algorithm instance to verify it doesn't run)
        algo2 = MockOnlineAlgorithm[float](name="A", return_sequence=[0.0])
        entry2 = AlgorithmEntry(algorithm=algo2, thresholds=[1.0], transformer=transformer)
        exec2: BenchmarkExecutor[float] = BenchmarkExecutor([entry2], [provider], solver, tmp_path)
        exec2.execute()

        # Transformer is still called during iteration
        assert transformer.call_count == 2

        # But the solver/algorithm was skipped due to cache hit
        assert len(algo2.get_call_history()) == 0

    def test_multiple_entries_mixed_transformers(self) -> None:
        """Executor should properly route data when processing mixed transformer configurations."""
        algo1 = MockOnlineAlgorithm[float](name="A", return_sequence=[0.0])
        algo2 = MockOnlineAlgorithm[float](name="A", return_sequence=[0.0])
        algo3 = MockOnlineAlgorithm[float](name="A", return_sequence=[0.0])

        entry_none = AlgorithmEntry(algorithm=algo1, thresholds=[1.0])
        entry_t1 = AlgorithmEntry(algorithm=algo2, thresholds=[1.0], transformer=MockDataTransformer("T1", 10.0))
        entry_t2 = AlgorithmEntry(algorithm=algo3, thresholds=[1.0], transformer=MockDataTransformer("T2", 20.0))

        # Provider yields [1.0, 1.0]
        provider: LabeledData[float] = _make_provider(2, name="data")
        solver: OnlineCpdSolver = OnlineCpdSolver()

        executor: BenchmarkExecutor[float] = BenchmarkExecutor(
            entries=[entry_none, entry_t1, entry_t2],
            providers=[provider],
            solver=solver,
        )
        results = executor.execute()

        assert len(results) == 3

        # Verify specific algorithm histories to ensure they received correct streams
        assert algo1.get_call_history() == [1.0, 1.0]  # No transformation
        assert algo2.get_call_history() == [11.0, 11.0]  # 1.0 + 10.0
        assert algo3.get_call_history() == [21.0, 21.0]  # 1.0 + 20.0
