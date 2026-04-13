import os
import pickle
import warnings
from pathlib import Path

import git  # type: ignore
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from pysatl_cpd.algorithms.online.shewhart_control_chart import ShewhartControlChart
from pysatl_cpd.analysis.labeled_data import LabeledData
from pysatl_cpd.core.online import OnlineCpdSolver, OnlineDetectionTrace
from pysatl_cpd.core.typedefs import UnivariateNumericArray as NumericArray

# TODO: write comment about path setting
REPO_ROOT_DIR = git.Repo(".", search_parent_directories=True).working_tree_dir
if REPO_ROOT_DIR is None:
    warnings.warn("Could not find locate root of repository!.Script will use current working directory", stacklevel=3)
    REPO_ROOT_DIR = os.getcwd()
    IMG_SAVE_PATH = Path(REPO_ROOT_DIR) / "online_detection_example.png"
    TRACE_DUMP_DIR = None
else:
    IMG_SAVE_PATH = Path(REPO_ROOT_DIR) / "assets" / "online_detection_example.png"
    TRACE_DUMP_DIR = Path(REPO_ROOT_DIR) / "assets" / "data"


def generate_labeled_data(
    data_len: int, means: list[float], change_point_index: list[int], var: float = 1.0
) -> LabeledData[np.float64]:
    """
    Generate synthetic labeled time series data with change points.

    Parameters
    ----------
    means : list[float]
        Mean values for each segment.
    lengths : list[int]
        Lengths of each segment.
    var : float, default=1.0
        Variance of the normal distribution.

    Returns
    -------
    LabeledData[np.float64]
        Labeled dataset with generated observations and change point indices.
    """
    idxs = [0] + change_point_index + [data_len]
    lengths = [idxs[i + 1] - idxs[i] for i in range(len(change_point_index) + 1)]
    if len(means) != len(lengths):
        raise ValueError("Length of means and lengths mismatch")

    raw_data: NumericArray = np.empty(shape=(0,))
    change_points: list[int] = [0]
    for mean, length in zip(means, lengths, strict=True):
        raw_data = np.concatenate(
            (raw_data, np.array(norm.rvs(size=length, loc=mean, scale=np.sqrt(var))).reshape((-1,))), axis=0
        )
        change_points.append(change_points[-1] + length)
    return LabeledData(raw_data=raw_data, change_points=change_points[1:-1])


data_len = 12_000
change_point_index = [1_000, 7_500, 10_500]
change_point_means = [0.9, 4.0, -2.0, 2.0]
data = generate_labeled_data(data_len, change_point_means, change_point_index)

idxs = [0] + change_point_index + [data_len]
mean = [change_point_means[i] for i in range(len(change_point_index) + 1) for _ in range(idxs[i + 1] - idxs[i])]


# Create Shewhart control chart algorithm
learning_period_size = 100
window_size = 50
algorithm = ShewhartControlChart(learning_period_size=100, window_size=50)

# Run online detection solver
solver = OnlineCpdSolver(
    skip_period=40,
    max_runlength=1000,
    collect_states=True,  # Enable state collection for learning periods
)

# Collect detection steps
steps = list(solver.run(algorithm, data, 2.5))

# Build detection trace from steps
trace = OnlineDetectionTrace.from_run(
    algorithm_name=algorithm.name, configuration_hash=hash(algorithm.configuration), threshold=2.5, steps=steps
)


# -----------------------------------#
#      Visualization (Naive)        #
# -----------------------------------#
# For advanced visualization examples see examples/visualization
# or checkout notebooks/tutorial_visualization.ipynb notebook


# Create figure with 3 subplots
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
fig.suptitle("Manual Visualization of Online Change-Point Detection", fontsize=16, fontweight="bold")

# Get data
time_points = np.arange(len(data))
values = np.array(list(data))

ax1, ax2, ax3 = axes[0], axes[1], axes[2]


# Subplot 1: Time series with change points
ax1.plot(time_points, values, "k-", linewidth=1, alpha=0.7, label="Time Series")

for i, cp in enumerate(data.change_points):
    # Add ground truth change points
    ax1.axvline(x=cp, color="red", linestyle="-", linewidth=2, alpha=0.8, label="Ground Truth" if i == 0 else "")
    # Add ground truth margins (window around truth)
    ax1.axvspan(cp, cp + 50, alpha=0.1, color="red", label="Margin Window" if i == 0 else "")

## Add detected change points
for i, cp in enumerate(trace.signal_change_points):
    ax1.axvline(x=cp, color="green", linestyle="--", linewidth=2, alpha=0.8, label="Detected CP" if i == 0 else "")

## Add forced change points
for i, cp in enumerate(trace.forced_change_points):
    ax1.axvline(x=cp, color="orange", linestyle="--", linewidth=2, alpha=0.8, label="Forced CP" if i == 0 else "")

ax1.set_ylabel("Value")
ax1.set_title("Time Series with Change Points")
ax1.legend(loc="upper left", fontsize=9)
ax1.grid(True, alpha=0.3)


# Subplot 2: Detection function with threshold
detection_scores = trace.detection_function
time_scores = np.arange(len(detection_scores))

ax2.plot(time_scores, detection_scores, "b-", linewidth=1, alpha=0.7, label="Detection Function")
ax2.axhline(y=2.5, color="red", linestyle="--", linewidth=2, alpha=0.8, label="Threshold (2.5)")
ax2.set_ylabel("Detection Statistic")
ax2.set_title("Detection Function")
ax2.legend(loc="upper left", fontsize=9)
ax2.grid(True, alpha=0.3)


# Subplot 3: Processing time
processing_times = trace.processing_time
time_proc = np.arange(len(processing_times))

ax3.plot(time_proc, processing_times, "purple", linewidth=1, alpha=0.7, label="Processing Time")
ax3.fill_between(time_proc, 0, processing_times, alpha=0.3, color="purple")

ax3.set_xlabel("Time Index")
ax3.set_ylabel("Time (seconds)")
ax3.set_title("Processing Time per Step")
ax3.legend(loc="upper left", fontsize=9)
ax3.grid(True, alpha=0.3)

# Commons: skip and learning periods

## Add skip periods (visualize as shaded regions)
for i, (s, e) in enumerate(trace.skip_periods):
    ax1.axvspan(s, e, alpha=0.2, color="brown", label="Skip Period" if i == 0 else "")
    ax2.axvspan(s, e, alpha=0.2, color="brown", label="Skip Period" if i == 0 else "")
    ax3.axvspan(s, e, alpha=0.2, color="brown", label="Skip Period" if i == 0 else "")

## Add learning periods (visualize as shaded regions)
for i, (s, e) in enumerate(trace.learning_periods):
    ax1.axvspan(s, e, alpha=0.2, color="green", label="Learning Period" if i == 0 else "")
    ax2.axvspan(s, e, alpha=0.2, color="green", label="Learning Period" if i == 0 else "")
    ax3.axvspan(s, e, alpha=0.2, color="green", label="Learning Period" if i == 0 else "")

# Adjust layout
plt.tight_layout()

if IMG_SAVE_PATH is not None:
    plt.savefig(str(IMG_SAVE_PATH))


if TRACE_DUMP_DIR is not None:
    with open(TRACE_DUMP_DIR / "trace.pcl", "wb") as f:
        pickle.dump(trace, f)
    with open(TRACE_DUMP_DIR / "data.pcl", "wb") as f:
        pickle.dump(data, f)
