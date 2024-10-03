from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

from CPDShell.Core.algorithms.bayesian_algorithm import BayesianAlgorithm
from CPDShell.Core.algorithms.BayesianCPD.detectors.simple_detector import SimpleDetector
from CPDShell.Core.algorithms.BayesianCPD.hazards.constant_hazard import ConstantHazard
from CPDShell.Core.algorithms.BayesianCPD.likelihoods.gaussian_unknown_mean_and_variance import (
    GaussianUnknownMeanAndVariance,
)
from CPDShell.Core.algorithms.BayesianCPD.localizers.simple_localizer import SimpleLocalizer
from CPDShell.labeled_data import LabeledCPData
from CPDShell.shell import CPDShell

THRESHOLD = 0.0
NUM_OF_SAMPLES = 1000
SAMPLE_SIZE = 500
BERNOULLI_PROB = 1.0 - 0.5 ** (1.0 / SAMPLE_SIZE)
HAZARD_RATE = 1 / BERNOULLI_PROB
LEARNING_SAMPLE_SIZE = 50

WORKING_DIR = Path()


def plot_algorithm_output(
    size, data, change_points, bayesian_algorithm, distributions, num, result_dir, with_save=False
):
    run_lengths = bayesian_algorithm.run_lengths
    gap_sizes = bayesian_algorithm.gap_sizes

    fig, axes = plt.subplots(4, 1, figsize=(20, 10))

    ax1, ax2, ax3, ax4 = axes

    ax1.set_title(f"Data: {distributions} №{num}")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Value")
    ax1.scatter(range(0, size), data)
    ax1.plot(range(0, size), data)
    ax1.set_xlim([0, size])
    ax1.margins(0)

    ax2.set_title("Run lengths distributions (log-normalized colors; white is 0.0 and black is 1.0)")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Run lengths distribution")
    ax2.imshow(np.rot90(run_lengths), aspect="auto", cmap="gray_r", norm=LogNorm(vmin=0.0001, vmax=1))
    ax2.set_xlim([0, size])
    ax2.margins(0)

    ax3.set_title("Maximal run length's probability")
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Probability")
    ax3.plot(
        run_lengths[np.arange(SAMPLE_SIZE), gap_sizes.astype(int)],
        color="orange",
        label="Probability of the maximal run length (a.k.a. gap size)",
    )
    ax3.set_xlim([0, size])
    ax3.margins(0)
    ax3.grid()
    ax3.legend()

    ax4.set_title("The most probable run length")
    ax4.set_xlabel("Time")
    ax4.set_ylabel("Run length")
    ax4.plot(np.argmax(run_lengths, axis=1), color="blue", label="The most probable run length")
    ax4.set_xlim([0, size])
    ax4.margins(0)

    for ax in axes:
        for cp in change_points:
            ax.axvline(cp, c="red", ls="dotted")

    plt.tight_layout()

    if with_save:
        if not Path(result_dir).exists():
            Path(result_dir).mkdir()
        plt.savefig(result_dir / "plot.png")
    else:
        plt.show()

    plt.close()


def process_samples(distributions, start, end, experiment_base_dir, results_base_dir):
    for sample_num in range(start, end):
        print(distributions, sample_num)
        reader = LabeledCPData.read_generated_datasets(experiment_base_dir / f"{distributions}\\sample_{sample_num}")

        gaussian_likelihood = GaussianUnknownMeanAndVariance()
        constant_hazard = ConstantHazard(HAZARD_RATE)
        simple_detector = SimpleDetector(THRESHOLD)
        simple_localizer = SimpleLocalizer()
        bayesian_algorithm = BayesianAlgorithm(
            learning_steps=LEARNING_SAMPLE_SIZE,
            likelihood=gaussian_likelihood,
            hazard=constant_hazard,
            detector=simple_detector,
            localizer=simple_localizer,
        )

        data = reader[f"{distributions}"].raw_data
        cpd = CPDShell(reader[f"{distributions}"], cpd_algorithm=bayesian_algorithm)
        cpd.scrubber.window_length = SAMPLE_SIZE
        cpd.scrubber.movement_k = 1.0
        bayesian_algorithm.localize(data)

        result_dir = results_base_dir / f"{distributions}\\sample_{sample_num}"
        result_dir.mkdir(parents=True, exist_ok=True)

        np.save(result_dir / "run_lengths", bayesian_algorithm.run_lengths)
        np.save(result_dir / "gap_sizes", bayesian_algorithm.gap_sizes)

        plot_algorithm_output(
            size=SAMPLE_SIZE,
            data=data,
            change_points=reader[distributions].change_points,
            bayesian_algorithm=bayesian_algorithm,
            distributions=distributions,
            num=sample_num,
            result_dir=result_dir,
            with_save=False,
        )


def process_datasets():
    # File paths to datasets and results directories.
    experiment_base_dir = Path("D:\\Alexey\\PyCharmProjects\\PySATL-CPD-Module\\CPDShell\\experiment\\stage_1\\")
    results_base_dir = Path("D:\\Alexey\\PyCharmProjects\\PySATL-CPD-Module\\CPDShell\\result\\stage_1\\")

    experiment_description = pd.read_csv(experiment_base_dir / "experiment_description.csv")
    names = experiment_description["name"].tolist()

    for name in names:
        process_samples(name, 0, 1, experiment_base_dir, results_base_dir)


if __name__ == "__main__":
    process_datasets()
