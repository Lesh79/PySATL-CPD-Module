from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pysatl_cpd.analysis.labeled_data import LabeledData
from pysatl_cpd.benchmark.metrics.multiple_run_metric import MultipleRunMetric
from pysatl_cpd.benchmark.noreset.noreset_detection_trace import NoResetDetectionTrace
from pysatl_cpd.benchmark.noreset.threshold_policy import ThresholdPolicy
from pysatl_cpd.benchmark.online_benchmark_runner import OnlineBenchmarkRunner
from pysatl_cpd.core.online.ionline_algorithm import OnlineAlgorithm
from pysatl_cpd.core.online.online_cpd_solver import OnlineCpdSolver
from pysatl_cpd.core.online.online_detection_trace import OnlineDetectionTrace


class NoResetBenchmarkRunner[ProviderT: LabeledData[Any]](OnlineBenchmarkRunner[NoResetDetectionTrace[Any], ProviderT]):
    def __init__(
        self,
        algorithms: Sequence[tuple[OnlineAlgorithm[Any, Any, Any], Sequence[float]]],
        providers: Sequence[ProviderT],
        metrics: dict[str, MultipleRunMetric[NoResetDetectionTrace[Any], ProviderT, Any]],
        solver: OnlineCpdSolver,
        policy: ThresholdPolicy,
        dump_dir: Path | None = None,
    ) -> None:
        return

    def _collect_runs(
        self,
        algorithm: OnlineAlgorithm[Any, Any, Any],
        threshold: float,
        providers: Sequence[ProviderT],
    ) -> list[tuple[NoResetDetectionTrace[Any], ProviderT]]:
        raise NotImplementedError("Method '_collect_runs' is not implemented yet.")

    def _get_inf_trace(
        self,
        algorithm: OnlineAlgorithm[Any, Any, Any],
        provider: ProviderT,
    ) -> OnlineDetectionTrace[Any]:
        raise NotImplementedError("Method '_get_inf_trace' is not implemented yet.")
