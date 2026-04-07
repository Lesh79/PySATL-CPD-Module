import numpy as np
import pytest

from pysatl_cpd.core.algorithms.ssa.decomposition import BasicSVD
from pysatl_cpd.core.algorithms.ssa.detectors import DistanceThreshold
from pysatl_cpd.core.algorithms.ssa.embedding import BasicEmbedding
from pysatl_cpd.core.algorithms.ssa.grouping import ConstantGrouping
from pysatl_cpd.core.algorithms.ssa.ssa import SSA
from pysatl_cpd.core.algorithms.ssa_online_algorithm import SSAOnline


@pytest.fixture
def experimental_params(distribution_type):
    params = {
        "size": 500,
        "change_point": 250,
        "tolerable_deviation": 25,
    }
    return params


@pytest.fixture
def confiure_algorithm(distribution_type):
    match distribution_type:
        case "normal":
            m = 12
        case "uniform":
            m = 2
        case _:
            raise ValueError("Unsupported likelihood")

    ssa = SSA(
        embedding_step=BasicEmbedding(),
        decomposition_step=BasicSVD(),
        grouping_step=ConstantGrouping(m),
    )
    detector = DistanceThreshold(threshold=0.75)

    ssa_online = SSAOnline(
        ssa=ssa,
        detector=detector,
        N=40,
    )
    return ssa_online


@pytest.fixture(scope="function")
def generate_data(distribution_type, experimental_params):
    def _generate():
        np.random.seed(42)
        cp = experimental_params["change_point"]
        size = experimental_params["size"]

        match distribution_type:
            case "normal":
                return np.concatenate([np.random.normal(0, 1, cp), np.random.normal(5, 2, size - cp)])
            case "uniform":
                return np.concatenate(
                    [
                        np.random.uniform(0.0, 0.1, cp),
                        np.random.uniform(2.0, 2.1, size - cp)
                    ]
                )
            case _:
                raise ValueError("Unsupported likelihood")

    return _generate


@pytest.mark.parametrize("distribution_type", ["normal", "uniform"])
class TestSSAOnlineAlgorithm:
    def test_consecutive_detection(self, generate_data, confiure_algorithm, experimental_params):
        online_ssa = confiure_algorithm
        data = generate_data()
        was_change_point = False
        for value in data:
            result = online_ssa.detect(value)
            if result:
                was_change_point = True

        assert was_change_point, "There was undetected change point in data"
        online_ssa.clear()

    def test_consecutive_localization(self, generate_data, confiure_algorithm, experimental_params):
        online_ssa = confiure_algorithm
        data = generate_data()
        was_change_point = False
        for value in data:
            result = online_ssa.localize(value)
            if result:
                was_change_point = True
                assert (
                    experimental_params["change_point"] - experimental_params["tolerable_deviation"]
                    <= result
                    <= experimental_params["change_point"] + experimental_params["tolerable_deviation"]
                ), "Incorrect change point localization"

        assert was_change_point, "There was undetected change point in data"
        online_ssa.clear()

    def test_online_localization_correctness(self, generate_data, confiure_algorithm, experimental_params):
        online_ssa = confiure_algorithm
        data = generate_data()
        for time, value in np.ndenumerate(data):
            result = online_ssa.detect(value)
            if result:
                assert experimental_params["change_point"] <= time[0], "Change point cannot be detected beforehand"

        online_ssa.clear()
