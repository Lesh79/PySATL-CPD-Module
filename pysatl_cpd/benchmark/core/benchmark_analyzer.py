from typing import Any

from pysatl_cpd.analysis.labeled_data import LabeledData
from pysatl_cpd.benchmark.metrics.multiple_run_metric import MultipleRunMetric
from pysatl_cpd.core.online.ionline_algorithm import OnlineAlgorithmState
from pysatl_cpd.core.online.online_detection_trace import OnlineDetectionTrace


class BenchmarkAnalyzer[TraceT: OnlineDetectionTrace[OnlineAlgorithmState], ProviderT: LabeledData[Any]]:
    def __init__(
        self,
        metrics: dict[str, MultipleRunMetric[TraceT, ProviderT, Any]],
    ) -> None:
        return

    def analyze(
        self,
        runs: list[tuple[TraceT, ProviderT]],
    ) -> dict[str, Any]:
        raise NotImplementedError("Method `analyze` is not implemented yet.")
