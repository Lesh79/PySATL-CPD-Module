"""
Example: Shewhart Control Chart benchmark on Normal Distribution data
using NoResetBenchmarkRunner with ClassificationReport metric.

Dataset structure:
- n rows (labeled data providers)
- Each row contains one change point
- Before change point: N(0, 1)
- After change point: N(mu_shift, 1)
"""

import numpy as np

from pysatl_cpd.analysis.labeled_data import LabeledData
from pysatl_cpd.benchmark.metrics.classification.classification_report import ClassificationReport
from pysatl_cpd.benchmark.noreset.noreset_benchmark_runner import NoResetBenchmarkRunner
from pysatl_cpd.benchmark.noreset.threshold_policy import EventBasedPolicy, PointBasedPolicy
from pysatl_cpd.core.online.online_cpd_solver import OnlineCpdSolver
from pysatl_cpd.algorithms.online.shewhart_control_chart import ShewhartControlChart


# ---------------------------------------------------------------------------
# 1. Labeled data provider
# ---------------------------------------------------------------------------

class NormalShiftProvider(LabeledData[float]):
    """
    Labeled data provider for a single time series with one change point.

    Before change point: N(mu_before, sigma)
    After change point:  N(mu_after,  sigma)

    Parameters
    ----------
    name : str
        Unique identifier for this provider.
    data : list[float]
        Pre-generated time series.
    change_point : int
        1-based index of the true change point.
    """

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
    """
    Generate n time series, each with one change point.

    Parameters
    ----------
    n : int
        Number of series (rows).
    series_length : int
        Total length of each series.
    change_point : int
        1-based index where the mean shifts.
    mu_before : float
        Mean before the change point.
    mu_after : float
        Mean after the change point.
    sigma : float
        Standard deviation (constant throughout).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    list[NormalShiftProvider]
        List of n labeled data providers.
    """
    rng = np.random.default_rng(seed)
    providers = []

    for i in range(n):
        # Segment before change point (1-based: indices 1..change_point-1)
        n_before = change_point - 1
        n_after = series_length - n_before

        before = rng.normal(mu_before, sigma, size=n_before).tolist()
        after = rng.normal(mu_after, sigma, size=n_after).tolist()

        data = before + after
        provider = NormalShiftProvider(
            name=f"series_{i:04d}",
            data=data,
            change_point=change_point,
        )
        providers.append(provider)

    return providers

# ---------------------------------------------------------------------------
# 4. Main benchmark
# ---------------------------------------------------------------------------

def main() -> None:
    # --- Parameters ---
    N_SERIES = 25          # number of rows
    SERIES_LENGTH = 10100    # length of each series
    CHANGE_POINT = 10000     # 1-based change point position
    MU_BEFORE = 0.0
    MU_AFTER = 0.5         # mean shift magnitude
    SIGMA = 1.0

    # Shewhart parameters
    LEARNING_PERIOD = 1000
    WINDOW_SIZE = 50

    # Thresholds to evaluate
    THRESHOLDS = np.linspace(0, 7, 30)

    # Error margin for TP/FP/FN matching
    ERROR_MARGIN = (0, 100)  # +/- 5 samples around true change point

    # --- Generate dataset ---
    providers = generate_dataset(
        n=N_SERIES,
        series_length=SERIES_LENGTH,
        change_point=CHANGE_POINT,
        mu_before=MU_BEFORE,
        mu_after=MU_AFTER,
        sigma=SIGMA,
        seed=42,
    )

    print(f"Dataset: {N_SERIES} series, length={SERIES_LENGTH}, "
          f"change_point={CHANGE_POINT}, shift={MU_AFTER - MU_BEFORE:.1f}σ")
    print(f"Algorithm: ShewhartControlChart("
          f"learning_period={LEARNING_PERIOD}, window={WINDOW_SIZE})")
    print(f"Thresholds: {THRESHOLDS}")
    print(f"Error margin: {ERROR_MARGIN}")
    print("-" * 60)

    # --- Algorithm ---
    algorithm = ShewhartControlChart(
        learning_period_size=LEARNING_PERIOD,
        window_size=WINDOW_SIZE,
    )

    # --- Metrics ---
    metrics = {
        "classification_report": ClassificationReport(error_margin=ERROR_MARGIN),
    }

    # --- Policy ---
    policy = EventBasedPolicy(ERROR_MARGIN[1], strict_edge=False)

    # --- Solver ---
    solver = OnlineCpdSolver()

    # --- Runner ---
    runner = NoResetBenchmarkRunner(
        algorithms=[(algorithm, THRESHOLDS)],
        providers=providers,
        metrics=metrics,
        solver=solver,
        policy=policy,
        dump_dir="benchmark_cache/",  # no caching
    )

    # --- Run ---
    results = runner.run()

    # --- Print results ---
    print(f"\n{'Threshold':>10} | {'TP':>6} | {'FP':>6} | {'FN':>6} | "
          f"{'Precision':>10} | {'Recall':>10} | {'F1':>10}")
    print("-" * 70)

    for (algo_name, config), threshold_results in results.items():
        for threshold, metric_values in threshold_results:
            report = metric_values["classification_report"]
            print(
                f"{threshold:>10.1f} | "
                f"{report['tp']:>6.0f} | "
                f"{report['fp']:>6.0f} | "
                f"{report['fn']:>6.0f} | "
                f"{report['precision']:>10.4f} | "
                f"{report['recall']:>10.4f} | "
                f"{report['f1']:>10.4f}"
            )


if __name__ == "__main__":
    main()
