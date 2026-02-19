import numpy as np
import pytest

from pysatl_cpd.core.algorithms.entropies.shannon_entropy import (
    ShannonEntropyAlgorithm,
)


def set_seed():
    np.random.seed(1)


def construct_shannon_entropy_algorithm():
    return ShannonEntropyAlgorithm(window_size=40, bins=10, threshold=0.3, anomaly_threshold=2.0, std_threshold=3.0)


@pytest.fixture(scope="function")
def data_params():
    return {
        "num_of_tests": 10,
        "size": 500,
        "change_point": 250,
        "tolerable_deviation": 30,
    }


@pytest.fixture
def generate_data(data_params):
    def _generate_data():
        set_seed()
        return np.concatenate(
            [
                np.random.normal(loc=0, scale=1, size=data_params["change_point"]),
                np.random.normal(
                    loc=5,
                    scale=2,
                    size=data_params["size"] - data_params["change_point"],
                ),
            ]
        )

    return _generate_data


@pytest.fixture(scope="function")
def outer_shannon_algorithm():
    return construct_shannon_entropy_algorithm()


@pytest.fixture
def inner_algorithm_factory():
    def _factory():
        return construct_shannon_entropy_algorithm()

    return _factory


def test_online_detection(outer_shannon_algorithm, generate_data, data_params):
    for _ in range(data_params["num_of_tests"]):
        data = generate_data()
        change_detected = False

        for point in data:
            if outer_shannon_algorithm.detect(point):
                change_detected = True
                break

        assert change_detected


def test_online_localization(outer_shannon_algorithm, generate_data, data_params):
    for _ in range(data_params["num_of_tests"]):
        data = generate_data()
        change_points = []

        algorithm = construct_shannon_entropy_algorithm()

        for point_idx, point in enumerate(data):
            cp = algorithm.localize(point)
            if cp is not None:
                change_points.append(cp)

        assert len(change_points) > 0

        closest_point = min(change_points, key=lambda x: abs(x - data_params["change_point"]))
        assert (
            data_params["change_point"] - data_params["tolerable_deviation"]
            <= closest_point
            <= data_params["change_point"] + data_params["tolerable_deviation"]
        )


def test_entropy_calculation():
    the_absolute_deviation_between_the_calculated_entropy_and_the_expected_entropy_value = 0.01
    algorithm = construct_shannon_entropy_algorithm()

    uniform_probs = np.ones(8) / 8
    computed_entropy = algorithm._compute_entropy(uniform_probs)
    expected_entropy = 3.0
    assert (
        abs(computed_entropy - expected_entropy)
        < the_absolute_deviation_between_the_calculated_entropy_and_the_expected_entropy_value
    )

    certain_probs = np.zeros(8)
    certain_probs[0] = 1.0
    computed_entropy = algorithm._compute_entropy(certain_probs)
    expected_entropy = 0.0
    assert (
        abs(computed_entropy - expected_entropy)
        < the_absolute_deviation_between_the_calculated_entropy_and_the_expected_entropy_value
    )


def test_change_point_detection():
    algorithm = construct_shannon_entropy_algorithm()

    data1 = np.zeros(60)
    for i in range(len(data1)):
        algorithm.detect(data1[i])

    change_detected = False
    data2 = np.ones(60)
    for i in range(len(data2)):
        if algorithm.detect(data2[i]):
            change_detected = True
            break

    assert change_detected


def test_edge_cases():
    algorithm = construct_shannon_entropy_algorithm()

    changes_detected = False
    for i in range(30):
        point = float(i) * 0.01 + np.random.normal(0, 0.0001)
        if algorithm.detect(point):
            changes_detected = True

    assert not changes_detected

    algorithm.reset()
    for _ in range(40):
        algorithm.detect(1.0)

    change_detected = False
    for _ in range(10):
        if algorithm.detect(10.0):
            change_detected = True
            break
    assert change_detected

    algorithm.reset()
    for i in range(20):
        algorithm.detect(float(i))
    constant_detected = False
    for _ in range(15):
        if algorithm.detect(5.0):
            constant_detected = True
            break

    assert constant_detected
