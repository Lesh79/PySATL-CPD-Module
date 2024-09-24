from CPDShell.Core.algorithms.knn_algorithm import KNNAlgorithm
from CPDShell.generator.distributions import (
    Distribution,
)
from CPDShell.generator.generator import ScipyDatasetGenerator
from CPDShell.shell import CPDShell


def metric(obs1: float, obs2: float) -> float:
    return abs(obs1 - obs2)


for _ in range(20):
    dist1 = Distribution.from_str("normal", {"mean": 0.0, "variance": 1.0})
    dist2 = Distribution.from_str("normal", {"mean": -2.0, "variance": 0.5})
    dataset = ScipyDatasetGenerator().generate_sample([dist1, dist2], [100, 100])

    shell = CPDShell(dataset)

    # specify CPD algorithm with parameters
    shell.CPDalgorithm = KNNAlgorithm(metric, k=5, threshold=4.7)

    shell.scrubber.window_length = 32
    shell.scrubber.movement_k = 0.5

    # then run algorithm
    change_points = shell.run_cpd()

    # print the results
    print(change_points)
    change_points.visualize()
