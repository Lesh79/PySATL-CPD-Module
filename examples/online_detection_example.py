"""
Example: Online Change-Point Detection with Shewhart Control Chart
"""

import matplotlib.pyplot as plt
import numpy as np

from pysatl_cpd._typing import NumericArray
from pysatl_cpd.data_providers import NDArrayUnivariateProvider
from pysatl_cpd.online.online_cpd_solver import OnlineCpdSolver
from pysatl_cpd.online.online_detection_trace import OnlineDetectionTrace
from pysatl_cpd.online.shewhart_control_chart import ShewhartControlChart


def generate_data_with_changes(n_points: int = 200) -> NumericArray:
    """Generate synthetic data with multiple change points."""
    np.random.seed(42)

    data: list[float] = []
    segments = [
        (0, 50, 0.0, 1.0),  # mean=0, std=1
        (50, 100, 3.0, 1.0),  # mean=3, std=1
        (100, 150, 0.0, 1.0),  # mean=0, std=1
        (150, 200, -2.0, 1.0),  # mean=-2, std=1
    ]

    for start, end, mean, std in segments:
        data.extend(np.random.normal(mean, std, end - start))

    return np.array(data)


def main() -> None:
    # Generate data
    print("Generating data with change points at indices 50, 100, 150...")
    data = generate_data_with_changes(200)

    # Create data provider
    data_provider = NDArrayUnivariateProvider(data)

    # Create algorithm with configuration
    algorithm = ShewhartControlChart(
        learning_period_size=20,
        window_size=10,
    )

    # Create solver with threshold 2.5
    solver = OnlineCpdSolver(
        data_provider=data_provider,
        algorithm=algorithm,
        threshold=2.5,
        skip_period=10,  # Skip 10 observations after detection
        collect_states=True,
    )

    # Run detection and collect results
    print("\nRunning online change-point detection...")
    results = list(solver.run())

    # Extract results
    detection_scores = [r.detection_function for r in results]
    change_points = [r.step_num for r in results if r.is_change_point]
    forced_changes = [r.step_num for r in results if r.is_force_change_point]
    skip_periods = [r.step_num for r in results if r.is_in_skip_period]

    # Print results
    print("\nResults:")
    print(f"  Total observations: {len(results)}")
    print(f"  Detected change points: {change_points}")
    print(f"  Forced changes: {forced_changes}")
    print(f"  Skip period observations: {len(skip_periods)}")

    trace = OnlineDetectionTrace.from_online_detection_steps(threshold=2.5, steps=results)

    print("\nTrace Summary:")
    print(f"  Number of change points: {len(trace.detected_changes)}")
    print(f"  Max detection score: {trace.observation_scores.max():.3f}")
    print(f"  Avg processing time: {trace.processing_time.mean():.6f}s")

    # Plot results
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Plot original data
    axes[0].plot(data, "b-", alpha=0.7, label="Data")
    axes[0].axhline(y=0, color="k", linestyle="--", alpha=0.3)
    axes[0].axhline(y=3, color="k", linestyle="--", alpha=0.3)
    axes[0].axhline(y=-2, color="k", linestyle="--", alpha=0.3)

    # Mark detected change points
    for cp in change_points:
        axes[0].axvline(x=cp, color="r", linestyle="--", alpha=0.7, linewidth=1.5)

    axes[0].set_title("Data with Detected Change Points")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Value")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot detection scores
    axes[1].plot(detection_scores, "g-", label="Detection Score")
    axes[1].axhline(y=2.5, color="r", linestyle="--", label="Threshold")
    axes[1].fill_between(
        range(len(detection_scores)),
        2.5,
        detection_scores,
        where=np.array(detection_scores) > 2.5,
        color="red",
        alpha=0.3,
        label="Detections",
    )

    axes[1].set_title("Detection Scores")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Score")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("online_detection_example.png")

    # Print algorithm state at end
    if algorithm.state:
        print("\nFinal Algorithm State:")
        print(f"  Learning period: {algorithm.state.is_in_learning_period}")
        print(f"  Current mean: {algorithm.state.mean:.3f}")
        print(f"  Current variance: {algorithm.state.variance:.3f}")


if __name__ == "__main__":
    main()
