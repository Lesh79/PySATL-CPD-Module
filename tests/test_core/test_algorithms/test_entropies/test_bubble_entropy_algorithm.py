import numpy as np
import pytest

from pysatl_cpd.core.algorithms.entropies.bubble_entropy import (
    BubbleEntropyAlgorithm,
)


def set_seed():
    np.random.seed(1)


def construct_bubble_entropy_algorithm():
    return BubbleEntropyAlgorithm(window_size=40, embedding_dimension=3, time_delay=1, threshold=0.2)


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
        data1 = np.zeros(data_params["change_point"])
        for i in range(5, data_params["change_point"]):
            data1[i] = 0.6 * data1[i - 1] - 0.3 * data1[i - 2] + 0.1 * data1[i - 3] + np.random.normal(0, 0.1)

        data2 = np.zeros(data_params["size"] - data_params["change_point"])
        for i in range(5, len(data2)):
            data2[i] = 0.2 * data2[i - 1] + 0.7 * data2[i - 2] + np.random.normal(0, 0.5)

        return np.concatenate([data1, data2])

    return _generate_data


@pytest.fixture(scope="function")
def outer_bubble_algorithm():
    return construct_bubble_entropy_algorithm()


@pytest.fixture
def inner_algorithm_factory():
    def _factory():
        return construct_bubble_entropy_algorithm()

    return _factory


def test_online_detection(outer_bubble_algorithm, generate_data, data_params):
    for _ in range(data_params["num_of_tests"]):
        data = generate_data()
        change_detected = False

        for point in data:
            if outer_bubble_algorithm.detect(point):
                change_detected = True
                break

        assert change_detected


def test_online_localization(outer_bubble_algorithm, generate_data, data_params):
    for _ in range(data_params["num_of_tests"]):
        data = generate_data()
        change_points = []

        algorithm = construct_bubble_entropy_algorithm()

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


def test_online_vs_batch_comparison():
    min_distances = 50
    size = 500
    change_point = 250

    np.random.seed(42)
    data1 = np.random.normal(0, 0.5, change_point)
    data2 = np.random.normal(3, 0.5, size - change_point)
    data = np.concatenate([data1, data2])

    online_algorithm = construct_bubble_entropy_algorithm()
    online_changes = []

    for idx, point in enumerate(data):
        cp = online_algorithm.localize(point)
        if cp is not None:
            online_changes.append(cp)

    assert len(online_changes) > 0

    min_distance = min([abs(cp - change_point) for cp in online_changes])
    assert min_distance <= min_distances


def test_bubble_entropy_calculation():
    algorithm = construct_bubble_entropy_algorithm()

    t = np.linspace(0, 4 * np.pi, 100)
    deterministic_signal = np.sin(t)

    for point in deterministic_signal:
        algorithm.detect(point)

    algorithm = construct_bubble_entropy_algorithm()
    np.random.seed(42)
    random_signal = np.random.normal(0, 1, 100)

    for point in random_signal:
        algorithm.detect(point)

    algorithm = construct_bubble_entropy_algorithm()
    constant_signal = np.ones(100)

    changes_detected = False
    for point in constant_signal:
        if algorithm.detect(point):
            changes_detected = True

    assert not changes_detected


def test_edge_cases():
    algorithm = construct_bubble_entropy_algorithm()

    changes_detected = False
    for i in range(20):
        if algorithm.detect(float(i)):
            changes_detected = True

    assert not changes_detected

    algorithm = construct_bubble_entropy_algorithm()
    for i in range(40):
        algorithm.detect(float(i))

    algorithm = construct_bubble_entropy_algorithm()

    for _ in range(50):
        algorithm.detect(0.0)

    change_detected = False
    for _ in range(50):
        if algorithm.detect(5.0):
            change_detected = True
            break

    assert change_detected
