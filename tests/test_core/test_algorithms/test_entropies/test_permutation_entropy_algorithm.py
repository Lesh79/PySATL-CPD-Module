import numpy as np
import pytest

from pysatl_cpd.core.algorithms.entropies.permutation_entropy import (
    PermutationEntropyAlgorithm,
)


def set_seed():
    np.random.seed(1)


def construct_permutation_entropy_algorithm():
    return PermutationEntropyAlgorithm(window_size=40, embedding_dimension=3, time_delay=1, threshold=0.2)


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
        t1 = np.linspace(0, 10 * np.pi, data_params["change_point"])
        data1 = np.sin(t1) + np.random.normal(0, 0.1, size=len(t1))
        data2 = np.random.normal(0, 1, size=data_params["size"] - data_params["change_point"])
        return np.concatenate([data1, data2])

    return _generate_data


@pytest.fixture(scope="function")
def outer_permutation_algorithm():
    return construct_permutation_entropy_algorithm()


@pytest.fixture
def inner_algorithm_factory():
    def _factory():
        return construct_permutation_entropy_algorithm()

    return _factory


def test_online_detection(outer_permutation_algorithm, generate_data, data_params):
    for _ in range(data_params["num_of_tests"]):
        data = generate_data()
        change_detected = False

        for point in data:
            if outer_permutation_algorithm.detect(point):
                change_detected = True
                break

        assert change_detected


def test_online_localization(outer_permutation_algorithm, generate_data, data_params):
    for _ in range(data_params["num_of_tests"]):
        data = generate_data()
        change_points = []

        algorithm = construct_permutation_entropy_algorithm()

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


def test_embedding_dimension_effect():
    t = np.linspace(0, 10 * np.pi, 200)
    signal = np.sin(t) + 0.1 * np.random.normal(0, 1, 200)

    algorithm1 = PermutationEntropyAlgorithm(
        window_size=40,
        embedding_dimension=2,
        time_delay=1,
        threshold=0.2,
    )

    algorithm2 = PermutationEntropyAlgorithm(
        window_size=40,
        embedding_dimension=5,
        time_delay=1,
        threshold=0.2,
    )

    change_points1 = []
    change_points2 = []

    for point in signal:
        if algorithm1.detect(point):
            change_points1.append(algorithm1._position)

        if algorithm2.detect(point):
            change_points2.append(algorithm2._position)

    assert len(change_points1) <= len(change_points2)


def test_time_delay_effect():
    t = np.linspace(0, 20 * np.pi, 400)

    algorithm1 = PermutationEntropyAlgorithm(window_size=40, embedding_dimension=3, time_delay=1, threshold=0.2)

    algorithm2 = PermutationEntropyAlgorithm(window_size=40, embedding_dimension=3, time_delay=1000, threshold=0.2)

    signal = np.concatenate([np.sin(t[:200]), np.random.normal(0, 1, 200)])

    change_detected1 = False
    for point in signal:
        if algorithm1.detect(point):
            change_detected1 = True
            break

    change_detected2 = False
    for point in signal:
        if algorithm2.detect(point):
            change_detected2 = True
            break

    assert change_detected1 != change_detected2


def test_edge_cases():
    algorithm = construct_permutation_entropy_algorithm()

    changes_detected = False
    for i in range(20):
        if algorithm.detect(float(i)):
            changes_detected = True

    assert not changes_detected

    algorithm = construct_permutation_entropy_algorithm()

    for _ in range(50):
        algorithm.detect(0.0)

    change_detected = False
    for _ in range(50):
        if algorithm.detect(5.0):
            change_detected = True
            break

    assert change_detected

    algorithm = construct_permutation_entropy_algorithm()
    constant_signal = np.ones(100)

    change_detected = False
    for point in constant_signal:
        if algorithm.detect(point):
            change_detected = True
            break

    assert not change_detected
