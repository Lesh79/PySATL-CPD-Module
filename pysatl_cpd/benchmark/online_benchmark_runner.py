# online_runner.py
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from pysatl_cpd.analysis.labeled_data import LabeledData
from pysatl_cpd.benchmark.metrics.multiple_run_metric import MultipleRunMetric
from pysatl_cpd.core.online.ionline_algorithm import OnlineAlgorithm, OnlineAlgorithmConfiguration
from pysatl_cpd.core.online.online_cpd_solver import OnlineCpdSolver
from pysatl_cpd.core.online.online_detection_trace import OnlineDetectionTrace


class OnlineBenchmarkRunner[TraceT: OnlineDetectionTrace[Any], ProviderT: LabeledData[Any]](ABC):
    def __init__(
        self,
        algorithms: list[tuple[OnlineAlgorithm[Any, Any, Any], list[float]]],
        providers: list[ProviderT],
        metrics: dict[str, MultipleRunMetric[TraceT, ProviderT, Any]],
        solver: OnlineCpdSolver,
        dump_dir: Path | None = None,
    ) -> None:
        return

    @abstractmethod
    def _collect_runs(
        self,
        algorithm: OnlineAlgorithm[Any, Any, Any],
        threshold: float,
        providers: list[ProviderT],
    ) -> list[tuple[TraceT, ProviderT]]:
        raise NotImplementedError("Method `_collect_runs` is not implemented yet.")

    def run(
        self,
    ) -> dict[tuple[str, OnlineAlgorithmConfiguration], list[tuple[float, dict[str, Any]]]]:
        raise NotImplementedError("Method `run` is not implemented yet.")
