from collections.abc import Sequence
from typing import Any

from pysatl_cpd.analysis.labeled_data import LabeledData
from pysatl_cpd.benchmark.metrics.aggregation_metric import AggregationMetric
from pysatl_cpd.core.detection_trace import DetectionTrace
from tests.mocks.analysis.metrics.run_metric import MockRunMetric


class MockAggregationMetric[TraceT: DetectionTrace, ProviderT: LabeledData[Any]](
    AggregationMetric[TraceT, ProviderT, float, float]
):
    """
    Minimal mock AggregationMetric.

    aggregate() sums all per-run results so tests can verify what was
    collected.
    """

    def __init__(self, base: MockRunMetric[TraceT, ProviderT]) -> None:
        self._base = base
        self.aggregate_calls: list[Sequence[float]] = []

    @property
    def base_metric(self) -> MockRunMetric[TraceT, ProviderT]:
        return self._base

    def aggregate(self, results: Sequence[float]) -> float:
        self.aggregate_calls.append(list(results))
        return sum(results)
