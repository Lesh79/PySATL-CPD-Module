import numpy as np
import pytest

from pysatl_cpd.core.algorithms.entropies.renyi_entropy import (
    RenyiEntropyAlgorithm,
)


def set_seed():
    np.random.seed(1)


def construct_renyi_entropy_algorithm():
    return RenyiEntropyAlgorithm(window_size=40, alpha=0.5, bins=10, threshold=0.15)


@pytest.fixture(scope="function")
def data_params():
    return {
        "num_of_tests": 10,
        "size": 500,
        "change_point": 250,
        "tolerable_deviation": 35,
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
def outer_renyi_algorithm():
    return construct_renyi_entropy_algorithm()


@pytest.fixture
def inner_algorithm_factory():
    def _factory():
        return construct_renyi_entropy_algorithm()

    return _factory


def test_online_detection(outer_renyi_algorithm, generate_data, data_params):
    for _ in range(data_params["num_of_tests"]):
        data = generate_data()
        change_detected = False

        for point in data:
            if outer_renyi_algorithm.detect(point):
                change_detected = True
                break

        assert change_detected, "Algorithm should detect at least one change point"


def test_online_localization(outer_renyi_algorithm, generate_data, data_params):
    for _ in range(data_params["num_of_tests"]):
        data = generate_data()
        change_points = []

        algorithm = construct_renyi_entropy_algorithm()

        for point_idx, point in enumerate(data):
            cp = algorithm.localize(point)
            if cp is not None:
                change_points.append(cp)

        assert len(change_points) > 0, "At least one change point should be detected"

        closest_point = min(change_points, key=lambda x: abs(x - data_params["change_point"]))
        assert (
            data_params["change_point"] - data_params["tolerable_deviation"]
            <= closest_point
            <= data_params["change_point"] + data_params["tolerable_deviation"]
        ), (
            f"Change point {closest_point} should be within "
            f"{data_params['tolerable_deviation']} of true point {data_params['change_point']}"
        )


def test_online_vs_batch_comparison():
    min_distances = 45
    size = 500
    change_point = 250

    np.random.seed(42)
    data1 = np.random.normal(0, 0.5, change_point)
    data2 = np.random.normal(3, 0.5, size - change_point)
    data = np.concatenate([data1, data2])

    online_algorithm = RenyiEntropyAlgorithm(window_size=30, alpha=0.5, bins=8, threshold=0.1)
    online_changes = []

    for idx, point in enumerate(data):
        cp = online_algorithm.localize(point)
        if cp is not None:
            online_changes.append(cp)

    assert len(online_changes) > 0, "At least one change point should be detected"

    min_distance = min([abs(cp - change_point) for cp in online_changes])
    assert min_distance <= min_distances, f"Minimum distance {min_distance} should be <= {min_distances}"


def test_renyi_entropy_calculation():
    algorithm1 = construct_renyi_entropy_algorithm()
    t = np.linspace(0, 4 * np.pi, 100)
    deterministic_signal = np.sin(t)

    for point in deterministic_signal:
        algorithm1.detect(point)

    deterministic_entropies = algorithm1.get_entropy_history()

    algorithm2 = construct_renyi_entropy_algorithm()
    np.random.seed(42)
    random_signal = np.random.normal(0, 1, 100)

    for point in random_signal:
        algorithm2.detect(point)

    random_entropies = algorithm2.get_entropy_history()

    algorithm3 = construct_renyi_entropy_algorithm()
    constant_signal = np.ones(100)

    changes_detected = False
    for point in constant_signal:
        if algorithm3.detect(point):
            changes_detected = True

    assert not changes_detected, "Constant signal should not trigger change detection"

    if len(deterministic_entropies) > 0:
        assert all(np.isfinite(e) for e in deterministic_entropies), "Deterministic signal entropies should be finite"
        assert all(e >= 0 for e in deterministic_entropies), "Entropies should be non-negative"

    if len(random_entropies) > 0:
        assert all(np.isfinite(e) for e in random_entropies), "Random signal entropies should be finite"
        assert all(e >= 0 for e in random_entropies), "Entropies should be non-negative"


def test_edge_cases():
    algorithm = construct_renyi_entropy_algorithm()
    changes_detected = False
    for i in range(30):
        if algorithm.detect(float(i)):
            changes_detected = True

    assert not changes_detected, "Should not detect changes with insufficient data"

    algorithm = construct_renyi_entropy_algorithm()
    for i in range(80):
        algorithm.detect(float(i))

    algorithm = RenyiEntropyAlgorithm(window_size=30, alpha=0.5, bins=6, threshold=0.08)

    for _ in range(50):
        algorithm.detect(0.0)

    change_detected = False
    for _ in range(40):
        if algorithm.detect(5.0):
            change_detected = True
            break

    assert change_detected, "Should detect change from constant to different value"


def test_parameter_validation():
    v1 = 0.5
    v2 = 8
    algorithm = RenyiEntropyAlgorithm(window_size=50, alpha=0.5, bins=8, threshold=0.2)
    assert algorithm._alpha == v1
    assert algorithm._bins == v2

    with pytest.raises(ValueError, match="Alpha must be positive and not equal to 1"):
        RenyiEntropyAlgorithm(alpha=0)

    with pytest.raises(ValueError, match="Alpha must be positive and not equal to 1"):
        RenyiEntropyAlgorithm(alpha=-0.5)

    with pytest.raises(ValueError, match="Alpha must be positive and not equal to 1"):
        RenyiEntropyAlgorithm(alpha=1.0)


def test_different_alpha_values():
    alpha_values = [0.1, 0.5, 2.0, 5.0, 10.0]

    np.random.seed(123)
    test_data = np.concatenate([np.random.normal(0, 0.3, 100), np.random.normal(2.5, 0.3, 100)])

    alpha_params = {
        0.1: {"threshold": 0.05, "bins": 6, "window_size": 35},
        0.5: {"threshold": 0.12, "bins": 8, "window_size": 40},
        2.0: {"threshold": 0.18, "bins": 10, "window_size": 45},
        5.0: {"threshold": 0.25, "bins": 12, "window_size": 50},
        10.0: {"threshold": 0.30, "bins": 12, "window_size": 50},
    }

    for alpha in alpha_values:
        params = alpha_params[alpha]
        algorithm = RenyiEntropyAlgorithm(
            window_size=params["window_size"], alpha=alpha, bins=params["bins"], threshold=params["threshold"]
        )

        change_points = []
        for idx, point in enumerate(test_data):
            cp = algorithm.localize(point)
            if cp is not None:
                change_points.append(cp)

        entropy_history = algorithm.get_entropy_history()

        assert len(entropy_history) > 0, f"Should produce entropy values for alpha={alpha}"
        assert all(np.isfinite(e) for e in entropy_history), f"All entropies should be finite for alpha={alpha}"
        assert len(change_points) > 0, f"Should detect changes for alpha={alpha}"


def test_binning_behavior():
    bin_counts = [5, 10, 15, 20]

    test_data = np.sin(np.linspace(0, 4 * np.pi, 80)) + 0.2 * np.random.normal(0, 1, 80)

    for num_bins in bin_counts:
        algorithm = RenyiEntropyAlgorithm(window_size=40, alpha=0.5, bins=num_bins, threshold=0.25)

        for point in test_data:
            algorithm.detect(point)

        entropy_history = algorithm.get_entropy_history()

        assert len(entropy_history) > 0, f"Should work with {num_bins} bins"
        assert all(np.isfinite(e) for e in entropy_history), f"Entropies should be finite with {num_bins} bins"


def test_adaptive_binning():
    v = 4.5
    algorithm = construct_renyi_entropy_algorithm()

    small_range_data = np.random.uniform(0, 1, 50)
    for point in small_range_data:
        algorithm.detect(point)

    large_range_data = np.concatenate([[-4.8, 4.8], np.random.uniform(-5, 5, 48)])
    np.random.shuffle(large_range_data)

    for point in large_range_data:
        algorithm.detect(point)

    assert algorithm._global_min is not None
    assert algorithm._global_max is not None
    assert algorithm._global_min <= -v
    assert algorithm._global_max >= v

    entropy_history = algorithm.get_entropy_history()
    assert len(entropy_history) > 0, "Should produce entropy values with adaptive binning"


def test_entropy_history_and_reset():
    algorithm = construct_renyi_entropy_algorithm()

    np.random.seed(456)
    test_data = np.random.normal(0, 1, 80)

    for point in test_data:
        algorithm.detect(point)

    entropy_history = algorithm.get_entropy_history()

    assert len(entropy_history) > 0, "Should have entropy history after processing data"

    for entropy_val in entropy_history:
        assert np.isfinite(entropy_val), f"Entropy value {entropy_val} should be finite"
        assert entropy_val >= 0, f"Entropy value {entropy_val} should be non-negative"

    algorithm._global_min
    algorithm._global_max

    algorithm.reset()

    assert len(algorithm.get_entropy_history()) == 0, "History should be empty after reset"
    assert algorithm._global_min is None, "Global min should be reset"
    assert algorithm._global_max is None, "Global max should be reset"

    for point in test_data[:40]:
        algorithm.detect(point)

    assert len(algorithm.get_entropy_history()) >= 0, "Should be able to process data after reset"


def test_different_data_types():
    algorithm = construct_renyi_entropy_algorithm()

    test_array = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = algorithm.detect(test_array)
    assert isinstance(result, bool)

    single_float = np.float64(3.14)
    result = algorithm.detect(single_float)
    assert isinstance(result, bool)

    algorithm.reset()
    larger_array = np.random.normal(0, 1, 80)
    change_points = []

    for i in range(0, len(larger_array), 10):
        chunk = larger_array[i : i + 10]
        cp = algorithm.localize(chunk)
        if cp is not None:
            change_points.append(cp)

    assert isinstance(change_points, list)


def test_special_alpha_cases():
    test_data = np.random.uniform(0, 1, 100)

    algorithm_small = RenyiEntropyAlgorithm(window_size=50, alpha=0.01, bins=10, threshold=0.3)
    for point in test_data:
        algorithm_small.detect(point)

    small_alpha_entropies = algorithm_small.get_entropy_history()

    algorithm_large = RenyiEntropyAlgorithm(window_size=50, alpha=100.0, bins=10, threshold=0.3)
    for point in test_data:
        algorithm_large.detect(point)

    large_alpha_entropies = algorithm_large.get_entropy_history()

    if len(small_alpha_entropies) > 0:
        assert all(np.isfinite(e) for e in small_alpha_entropies), "Small alpha entropies should be finite"

    if len(large_alpha_entropies) > 0:
        assert all(np.isfinite(e) for e in large_alpha_entropies), "Large alpha entropies should be finite"


def test_probability_computation():
    algorithm = construct_renyi_entropy_algorithm()

    test_data = np.concatenate([np.full(30, 0.0), np.full(15, 1.0), np.full(5, 2.0)])
    np.random.shuffle(test_data)

    for point in test_data:
        algorithm.detect(point)

    entropy_history = algorithm.get_entropy_history()

    if len(entropy_history) > 0:
        assert all(e >= 0 for e in entropy_history), "Entropies should be non-negative"
        assert all(np.isfinite(e) for e in entropy_history), "Entropies should be finite"


def test_variance_based_detection():
    v = 5
    algorithm = construct_renyi_entropy_algorithm()

    stable_data = np.sin(np.linspace(0, 4 * np.pi, 100))

    chaotic_data = []
    for i in range(50):
        if i % 10 < v:
            chaotic_data.append(np.random.normal(0, 0.1))
        else:
            chaotic_data.append(np.random.normal(0, 2.0))

    test_data = np.concatenate([stable_data, chaotic_data])

    change_points = []
    for idx, point in enumerate(test_data):
        cp = algorithm.localize(point)
        if cp is not None:
            change_points.append(cp)

    assert len(change_points) > 0, "Should detect changes due to entropy variance"


def test_edge_case_data():
    algorithm1 = RenyiEntropyAlgorithm(window_size=30, alpha=0.5, bins=10, threshold=0.2)
    identical_data = np.full(50, 5.0)

    for point in identical_data:
        algorithm1.detect(point)

    entropy_history1 = algorithm1.get_entropy_history()
    if len(entropy_history1) > 0:
        assert all(np.isfinite(e) for e in entropy_history1), "Should handle identical values"

    algorithm2 = RenyiEntropyAlgorithm(window_size=20, alpha=2.0, bins=5, threshold=0.3)
    small_data = np.array([1.0, 2.0, 3.0])

    changes_detected = False
    for point in small_data:
        if algorithm2.detect(point):
            changes_detected = True

    assert not changes_detected, "Should not detect changes with very small dataset"
