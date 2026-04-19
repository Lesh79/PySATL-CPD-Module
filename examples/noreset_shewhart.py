"""
Example: Shewhart Control Chart benchmark on Normal Distribution data
using NoResetBenchmark with ClassificationReport & Delay metrics,
and Average Run Length (ARL) evaluation.
"""

import numpy as np
import pandas as pd

from pysatl_cpd.algorithms.online.shewhart_control_chart import ShewhartControlChart
from pysatl_cpd.benchmark.metrics.classification.classification_report import ClassificationReport
from pysatl_cpd.benchmark.metrics.online.arl_metric import ARLMetric
from pysatl_cpd.benchmark.metrics.online.delay_metric import MeanDelayMetric, MedianDelayMetric
from pysatl_cpd.benchmark.noreset.noreset_benchmark_runner import (
    LinspaceThresholds,
    NoResetBenchmark,
    OnlineBenchmarkEntry,
)
from pysatl_cpd.benchmark.noreset.threshold_policy import EventBasedPolicy, PointBasedPolicy
from pysatl_cpd.core.data_providers.dataset import Annotation, PandasLabeledDataProvider
from pysatl_cpd.core.online.online_cpd_solver import OnlineCpdSolver

# ---------------------------------------------------------------------------
# 1. Dataset generation
# ---------------------------------------------------------------------------

def generate_dataset(
    n: int,
    series_length: int = 200,
    change_point: int = 100,
    mu_before: float = 0.0,
    mu_after: float = 3.0,
    sigma: float = 1.0,
    seed: int = 42,
) -> list[PandasLabeledDataProvider]:
    """Generate n time series, each with one change point using PandasLabeledDataProvider."""
    rng = np.random.default_rng(seed)
    providers = []

    for i in range(n):
        before = rng.normal(mu_before, sigma, size=change_point)
        after = rng.normal(mu_after, sigma, size=series_length - change_point)
        data = np.concatenate([before, after])

        segments = np.zeros(series_length, dtype=int)
        segments[change_point:] = 1

        df = pd.DataFrame({"value": data, "segment": segments})

        seg_info = pd.DataFrame({
            "segment": [0, 1],
            "start": [0, change_point],
            "end": [change_point - 1, series_length - 1],
        })

        provider = PandasLabeledDataProvider(
            dataset=df,
            segment_info=seg_info,
            annotation=Annotation(scenario="shift"),
            name=f"series_{i:04d}",
        )
        providers.append(provider)

    return providers


def generate_arl_dataset(
    n: int,
    series_length: int = 200,
    mu: float = 0.0,
    sigma: float = 1.0,
    seed: int = 42,
) -> list[PandasLabeledDataProvider]:
    """Generate n stationary time series without change points for ARL."""
    rng = np.random.default_rng(seed)
    providers = []

    for i in range(n):
        data = rng.normal(mu, sigma, size=series_length)

        df = pd.DataFrame({"value": data, "segment": 0})
        seg_info = pd.DataFrame({
            "segment": [0],
            "start": [0],
            "end": [series_length - 1],
        })

        provider = PandasLabeledDataProvider(
            dataset=df,
            segment_info=seg_info,
            annotation=Annotation(scenario="null"),
            name=f"arl_series_{i:04d}",
        )
        providers.append(provider)

    return providers


# ---------------------------------------------------------------------------
# 2. Main benchmark
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

    # Error margin for TP/FP/FN matching & Delays
    ERROR_MARGIN = (0, 100)

    # --- Generate datasets ---
    providers = generate_dataset(
        n=N_SERIES, series_length=SERIES_LENGTH, change_point=CHANGE_POINT,
        mu_before=MU_BEFORE, mu_after=MU_AFTER, sigma=SIGMA, seed=42,
    )
    arl_providers = generate_arl_dataset(
        n=N_SERIES, series_length=SERIES_LENGTH,
        mu=MU_BEFORE, sigma=SIGMA, seed=42,
    )

    print(f"Algorithm: ShewhartControlChart(learning_period={LEARNING_PERIOD}, window={WINDOW_SIZE})")
    print(f"Dataset (NoReset): {N_SERIES} series, length={SERIES_LENGTH}, cp={CHANGE_POINT}, shift={MU_AFTER - MU_BEFORE:.1f}σ")
    print(f"Dataset (ARL):     {N_SERIES} series, length={SERIES_LENGTH}, no change points")
    print(f"Error margin: {ERROR_MARGIN}")
    print("-" * 115)

    algorithm = ShewhartControlChart(
        learning_period_size=LEARNING_PERIOD,
        window_size=WINDOW_SIZE,
    )
    solver = OnlineCpdSolver()

    entry = OnlineBenchmarkEntry(
        algorithm=algorithm,
        thresholds=LinspaceThresholds(start=0, stop=7, num=30),
        entry_name="Shewhart"
    )

    # ==========================================
    # RUN 1: Classification & Delays (NoReset)
    # ==========================================
    metrics_q = {
        "classification_report": ClassificationReport(error_margin=ERROR_MARGIN),
        "mean_delay": MeanDelayMetric(max_delay=ERROR_MARGIN[1]),
        "median_delay": MedianDelayMetric(max_delay=ERROR_MARGIN[1]),
    }

    runner_q = NoResetBenchmark(
        solver=solver,
        policy=EventBasedPolicy(ERROR_MARGIN[1], strict_edge=False),
        metrics=metrics_q,
        dump_dir="benchmark_cache/noreset",
        verbose=True,
    )

    results_q = runner_q.run(entries=[entry], providers=providers)
    df_quality = results_q["Shewhart"]

    report_df = df_quality["classification_report"].apply(pd.Series)
    df_quality = pd.concat([df_quality.drop(columns=["classification_report"]), report_df], axis=1)

    # ==========================================
    # RUN 2: Average Run Length (ARL)
    # ==========================================
    runner_arl = NoResetBenchmark(
        solver=solver,
        policy=PointBasedPolicy(strict=True), # Быстрая поточечная экстракция
        metrics={"arl": ARLMetric()},
        dump_dir="benchmark_cache/arl",
        verbose=True,
    )

    results_arl = runner_arl.run(entries=[entry], providers=arl_providers)
    df_arl = results_arl["Shewhart"]

    # ==========================================
    # Combine and Print Results
    # ==========================================
    df_final = pd.merge(df_quality, df_arl, on="threshold", how="outer").sort_values("threshold")

    print(
        f"\n{'Threshold':>10} | {'ARL':>10} | {'TP':>4} | {'FP':>4} | {'FN':>4} | "
        f"{'Precision':>9} | {'Recall':>9} | {'F1':>9} | "
        f"{'Mean Delay':>8} | {'Med Delay':>8}"
    )
    print("-" * 115)

    for _, res in df_final.iterrows():
        print(
            f"{res['threshold']:>10.1f} | "
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
