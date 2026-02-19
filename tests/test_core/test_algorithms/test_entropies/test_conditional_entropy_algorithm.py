import numpy as np
import pytest

from pysatl_cpd.core.algorithms.entropies.conditional_entropy import (
    ConditionalEntropyAlgorithm,
)


def set_seed():
    np.random.seed(1)


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
        data_x = np.concatenate(
            [
                np.random.normal(loc=0, scale=1, size=data_params["change_point"]),
                np.random.normal(
                    loc=5,
                    scale=2,
                    size=data_params["size"] - data_params["change_point"],
                ),
            ]
        )
        data_y = np.concatenate(
            [
                np.random.normal(loc=2, scale=1, size=data_params["change_point"]),
                np.random.normal(
                    loc=-2,
                    scale=1,
                    size=data_params["size"] - data_params["change_point"],
                ),
            ]
        )
        return data_x, data_y

    return _generate_data


@pytest.fixture
def conditional_algorithm_factory():
    def _factory():
        return ConditionalEntropyAlgorithm(
            window_size=40,
            bins=10,
            threshold=0.3,
        )

    return _factory


def test_online_detection(conditional_algorithm_factory, generate_data, data_params):
    for _ in range(data_params["num_of_tests"]):
        data_x, data_y = generate_data()
        algorithm = conditional_algorithm_factory()
        change_detected = False

        for i in range(len(data_x)):
            observation = np.array([data_x[i], data_y[i]])
            if algorithm.detect(observation):
                change_detected = True
                break

        assert change_detected


def test_online_localization(conditional_algorithm_factory, generate_data, data_params):
    for _ in range(data_params["num_of_tests"]):
        data_x, data_y = generate_data()
        algorithm = conditional_algorithm_factory()
        change_points = []

        for i in range(len(data_x)):
            observation = np.array([data_x[i], data_y[i]])
            cp = algorithm.localize(observation)
            if cp is not None:
                change_points.append(cp)

        assert len(change_points) > 0
        closest_point = min(change_points, key=lambda x: abs(x - data_params["change_point"]))
        assert (
            data_params["change_point"] - data_params["tolerable_deviation"]
            <= closest_point
            <= data_params["change_point"] + data_params["tolerable_deviation"]
        )


def test_conditional_entropy():
    size = 200
    algorithm = ConditionalEntropyAlgorithm(window_size=40, bins=10, threshold=0.3)

    data_x = np.random.normal(0, 1, size)
    data_y1 = data_x + np.random.normal(0, 0.1, size)

    for i in range(size):
        algorithm.detect(np.array([data_x[i], data_y1[i]]))

    algorithm = ConditionalEntropyAlgorithm(window_size=40, bins=10, threshold=0.3)

    data_x = np.random.normal(0, 1, size)
    data_y2 = np.random.normal(0, 1, size)

    change_detected = False
    for i in range(size - 40):
        observation = np.array([data_x[i], data_y2[i]])
        if algorithm.detect(observation):
            change_detected = True
            break

    for i in range(size - 40, size):
        observation = np.array([data_x[i], data_y1[i]])
        if algorithm.detect(observation):
            change_detected = True
            break

    assert change_detected


def test_edge_cases():
    algorithm = ConditionalEntropyAlgorithm(window_size=10, bins=5, threshold=0.3)

    with pytest.raises(ValueError):
        algorithm.detect(np.float64(1.0))

    algorithm = ConditionalEntropyAlgorithm(window_size=40, bins=10, threshold=0.3)

    for i in range(20):
        observation = np.array([float(i), float(i + 1)])
        assert not algorithm.detect(observation)

    algorithm = ConditionalEntropyAlgorithm(window_size=40, bins=10, threshold=0.3)

    for i in range(50):
        observation = np.array([0.0, 0.0])
        algorithm.detect(observation)

    change_detected = False
    for i in range(50):
        observation = np.array([5.0, -5.0])
        if algorithm.detect(observation):
            change_detected = True
            break

    assert change_detected
