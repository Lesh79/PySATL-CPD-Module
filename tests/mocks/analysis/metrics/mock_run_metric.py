from collections.abc import Sequence
from typing import Any

from pysatl_cpd.analysis.labeled_data import LabeledData
from pysatl_cpd.analysis.metrics.single_run_metric import SingleRunMetric as SingleRunMetric
from pysatl_cpd.core.detection_trace import DetectionTrace


class MockRunMetric[TraceT: DetectionTrace, ProviderT: LabeledData[Any]](SingleRunMetric[TraceT, ProviderT, float]):
    """
    Minimal SingleRunMetric that returns values from a pre-configured sequence.

    Records every (trace, data) pair that was passed to evaluate() so that
    tests can assert call order and arguments.
    """

    def __init__(self, return_values: Sequence[float]) -> None:
        self._return_values = list(return_values)
        self._call_index = 0
        self.calls: list[tuple[DetectionTrace, LabeledData[Any]]] = []

    def evaluate(self, trace: TraceT, data: ProviderT) -> float:
        self.calls.append((trace, data))
        value = self._return_values[self._call_index % len(self._return_values)]
        self._call_index += 1
        return value
