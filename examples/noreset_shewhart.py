"""
Example: Shewhart Control Chart benchmark on Normal Distribution data
using NoResetBenchmarkRunner with ClassificationReport & Delay metrics,
and ARLBenchmarkRunner for Average Run Length evaluation.
"""

import numpy as np

from pysatl_cpd.algorithms.online.shewhart_control_chart import ShewhartControlChart
from pysatl_cpd.analysis.labeled_data import LabeledData
from pysatl_cpd.benchmark.arl_benchmark_runner import ARLBenchmarkRunner
from pysatl_cpd.benchmark.metrics.classification.classification_report import ClassificationReport
from pysatl_cpd.benchmark.metrics.online.delay_metric import MeanDelayMetric, MedianDelayMetric
from pysatl_cpd.benchmark.noreset.noreset_benchmark_runner import NoResetBenchmarkRunner
from pysatl_cpd.benchmark.noreset.threshold_policy import EventBasedPolicy
from pysatl_cpd.core.algorithm_entry import AlgorithmEntry
from pysatl_cpd.core.online.online_cpd_solver import OnlineCpdSolver

# ---------------------------------------------------------------------------
# 1. Labeled data providers
# ---------------------------------------------------------------------------


class NormalShiftProvider(LabeledData[float]):
    """Provider for a single time series WITH one change point."""

    def __init__(self, name: str, data: list[float], change_point: int) -> None:
        self._name = name
        self._data = data
        self._change_point = change_point

    @property
    def name(self) -> str:
        return self._name

    @property
    def change_points(self) -> list[int]:
        return [self._change_point]

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)


class NormalNullProvider(LabeledData[float]):
    """Provider for a single time series WITHOUT change points (for ARL)."""

    def __init__(self, name: str, data: list[float]) -> None:
        self._name = name
        self._data = data

    @property
    def name(self) -> str:
        return self._name

    @property
    def change_points(self) -> list[int]:
        return []

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)


# ---------------------------------------------------------------------------
# 2. Dataset generation
# ---------------------------------------------------------------------------


def generate_dataset(
    n: int,
    series_length: int = 200,
    change_point: int = 100,
    mu_before: float = 0.0,
    mu_after: float = 3.0,
    sigma: float = 1.0,
    seed: int = 42,
) -> list[NormalShiftProvider]:
    """Generate n time series, each with one change point."""
    rng = np.random.default_rng(seed)
    providers = []

    for i in range(n):
        n_before = change_point - 1
        n_after = series_length - n_before

        before = rng.normal(mu_before, sigma, size=n_before).tolist()
        after = rng.normal(mu_after, sigma, size=n_after).tolist()

        provider = NormalShiftProvider(
            name=f"series_{i:04d}",
            data=before + after,
            change_point=change_point,
        )
        providers.append(provider)

    return providers


def generate_arl_dataset(
    n: int,
    series_length: int = 200,
    mu: float = 0.0,
    sigma: float = 1.0,
    seed: int = 42,
) -> list[NormalNullProvider]:
    """Generate n stationary time series without change points for ARL."""
    rng = np.random.default_rng(seed)
    providers = []

    for i in range(n):
        data = rng.normal(mu, sigma, size=series_length).tolist()
        provider = NormalNullProvider(
            name=f"arl_series_{i:04d}",
            data=data,
        )
        providers.append(provider)

    return providers


# ---------------------------------------------------------------------------
# 3. Main benchmark
# ---------------------------------------------------------------------------


def main() -> None:
    # --- Parameters ---
    N_SERIES = 25
    SERIES_LENGTH = 10100
    CHANGE_POINT = 10000
    MU_BEFORE = 0.0
    MU_AFTER = 0.5
    SIGMA = 1.0

    # Shewhart parameters
    LEARNING_PERIOD = 1000
    WINDOW_SIZE = 50

    # Thresholds to evaluate
    THRESHOLDS = np.linspace(0, 7, 3000)

    # Error margin for TP/FP/FN matching & Delays
    ERROR_MARGIN = (0, 100)

    # --- Generate datasets ---
    # 1. Dataset with change points for Quality and Delays
    providers = generate_dataset(
        n=N_SERIES,
        series_length=SERIES_LENGTH,
        change_point=CHANGE_POINT,
        mu_before=MU_BEFORE,
        mu_after=MU_AFTER,
        sigma=SIGMA,
        seed=42,
    )
    # 2. Dataset without change points for ARL
    arl_providers = generate_arl_dataset(
        n=N_SERIES,
        series_length=SERIES_LENGTH,
        mu=MU_BEFORE,
        sigma=SIGMA,
        seed=42,
    )

    print(f"Algorithm: ShewhartControlChart(learning_period={LEARNING_PERIOD}, window={WINDOW_SIZE})")
    print(
        f"Dataset (NoReset): {N_SERIES} series, length={SERIES_LENGTH}, change_point={CHANGE_POINT},"
        f"shift={MU_AFTER - MU_BEFORE:.1f}*sigma"
    )
    print(f"Dataset (ARL):     {N_SERIES} series, length={SERIES_LENGTH}, no change points")
    print(f"Error margin: {ERROR_MARGIN}")
    print("-" * 115)

    algorithm = ShewhartControlChart(
        learning_period_size=LEARNING_PERIOD,
        window_size=WINDOW_SIZE,
    )
    solver = OnlineCpdSolver()

    # ==========================================
    # RUN 1: Classification & Delays (NoReset)
    # ==========================================
    metrics = {
        "classification_report": ClassificationReport(error_margin=ERROR_MARGIN),
        "mean_delay": MeanDelayMetric(max_delay=ERROR_MARGIN[1]),
        "median_delay": MedianDelayMetric(max_delay=ERROR_MARGIN[1]),
    }
    policy = EventBasedPolicy(ERROR_MARGIN[1], strict_edge=False)

    runner = NoResetBenchmarkRunner(
        entries=[AlgorithmEntry(algorithm, THRESHOLDS)],
        providers=providers,
        metrics=metrics,
        solver=solver,
        policy=policy,
        dump_dir="benchmark_cache/noreset",
        verbose=True,
    )
    noreset_results = runner.run()

    # ==========================================
    # RUN 2: Average Run Length (ARL)
    # ==========================================
    arl_runner = ARLBenchmarkRunner(
        entries=[AlgorithmEntry(algorithm, THRESHOLDS)],
        providers=arl_providers,
        solver=solver,
        mode="noreset",  # uses rapid point-based extraction behind the scenes
        dump_dir="benchmark_cache/arl",
        verbose=True,
    )
    arl_results = arl_runner.run()

    # ==========================================
    # Combine and Print Results
    # ==========================================

    # Structure to hold merged metrics: {threshold: {"metric_name": value}}
    combined_results = {}

    # 1. Parse ARL
    for (_algo_name, _config), threshold_results in arl_results.items():
        for threshold, metric_values in threshold_results:
            combined_results.setdefault(threshold, {})["arl"] = metric_values["arl"]

    # 2. Parse Quality & Delays
    for (_algo_name, _config), threshold_results in noreset_results.items():
        for threshold, metric_values in threshold_results:
            rep = metric_values["classification_report"]
            combined_results.setdefault(threshold, {}).update(
                {
                    "tp": rep["tp"],
                    "fp": rep["fp"],
                    "fn": rep["fn"],
                    "precision": rep["precision"],
                    "recall": rep["recall"],
                    "f1": rep["f1"],
                    "mean_delay": metric_values["mean_delay"],
                    "median_delay": metric_values["median_delay"],
                }
            )

    # 3. Print unified table
    print(
        f"\n{'Threshold':>10} | {'ARL':>10} | {'TP':>4} | {'FP':>4} | {'FN':>4} | "
        f"{'Precision':>9} | {'Recall':>9} | {'F1':>9} | "
        f"{'Mean Delay':>8} | {'Med Delay':>8}"
    )
    print("-" * 115)

    for threshold in sorted(combined_results.keys()):
        res = combined_results[threshold]
        print(
            f"{threshold:>10.1f} | "
            f"{res.get('arl', float('inf')):>10.1f} | "
            f"{res.get('tp', 0):>4.0f} | "
            f"{res.get('fp', 0):>4.0f} | "
            f"{res.get('fn', 0):>4.0f} | "
            f"{res.get('precision', 0):>9.4f} | "
            f"{res.get('recall', 0):>9.4f} | "
            f"{res.get('f1', 0):>9.4f} | "
            f"{res.get('mean_delay', 0):>8.1f} | "
            f"{res.get('median_delay', 0):>8.1f}"
        )


if __name__ == "__main__":
    main()
