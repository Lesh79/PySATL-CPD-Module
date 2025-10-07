import numpy as np
import pytest

import pysatl_cpd.generator.distributions as dstr
from pysatl_cpd.core.algorithms.arima_online_algorithm import ArimaCusumAlgorithm


@pytest.fixture(scope="module")
def set_seed():
    np.random.seed(1)


@pytest.fixture
def algorithm_factory():
    def _factory():
        return ArimaCusumAlgorithm(training_size=5, h_coefficient=5, ema_alpha=0.05, p=0, d=0, q=0)

    return _factory


@pytest.fixture(scope="function")
def data_params():
    base_params = {
        "num_of_tests": 10,
        "size": 500,
        "change_point": 250,
    }
    return base_params


@pytest.fixture()
def generate_data(data_params):
    np.random.seed(1)
    cp = data_params["change_point"]
    size = data_params["size"]

    left_distr = dstr.Distribution.from_str(str(dstr.Distributions.BETA), {"alpha": "0.5", "beta": "0.5"})
    right_distr = dstr.Distribution.from_str(str(dstr.Distributions.NORMAL), {"mean": "1.0", "variance": "1.0"})
    return np.concatenate([left_distr.scipy_sample(cp), right_distr.scipy_sample(size - cp)])


class TestArimaCusumAlgorithm:
    @pytest.fixture(autouse=True)
    def setup(self, algorithm_factory):
        self.algorithm_factory = algorithm_factory

    def test_consecutive_detection(self, generate_data, data_params):
        for _ in range(data_params["num_of_tests"]):
            algorithm = self.algorithm_factory()
            was_change_point = False
            for value in generate_data:
                result = algorithm.detect(value)
                if result:
                    was_change_point = True

            assert was_change_point, "There was undetected change point in data"

    def test_correctness_of_consecutive_detection(self, generate_data, data_params):
        outer_algorithm = self.algorithm_factory()
        for _ in range(data_params["num_of_tests"]):
            inner_algorithm = self.algorithm_factory()
            outer_result = []
            inner_result = []

            for value in generate_data:
                outer_result.append(outer_algorithm.detect(value))
                inner_result.append(inner_algorithm.detect(value))

            outer_algorithm.clear()
            assert outer_result == inner_result, "Consecutive and independent detection should give same results"

    def test_correctness_of_consecutive_localization(self, generate_data, data_params):
        outer_algorithm = self.algorithm_factory()
        for _ in range(data_params["num_of_tests"]):
            inner_algorithm = self.algorithm_factory()

            for value in generate_data:
                outer_result = outer_algorithm.localize(value)
                inner_result = inner_algorithm.localize(value)
                assert outer_result == inner_result, "Consecutive and independent localization should give same results"

            outer_algorithm.clear()

    def test_online_localization_correctness(self, generate_data, data_params):
        for _ in range(data_params["num_of_tests"]):
            algorithm = self.algorithm_factory()
            for time, value in np.ndenumerate(generate_data):
                result = algorithm.localize(value)
                if result:
                    assert result <= time[0] + 1, "Change point cannot be in future"
