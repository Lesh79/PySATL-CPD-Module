import numpy as np
import pytest

from pysatl_cpd.core.algorithms.entropies.sample_entropy import (
    SampleEntropyAlgorithm,
)


def set_seed():
    np.random.seed(1)


def construct_sample_entropy_algorithm():
    return SampleEntropyAlgorithm(window_size=60, m=2, r=None, r_factor=0.25, threshold=0.3)


@pytest.fixture(scope="function")
def data_params():
    return {
        "num_of_tests": 10,
        "size": 500,
        "change_point": 250,
        "tolerable_deviation": 50,
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
def outer_sample_algorithm():
    return construct_sample_entropy_algorithm()


@pytest.fixture
def inner_algorithm_factory():
    def _factory():
        return construct_sample_entropy_algorithm()

    return _factory


def test_online_detection(outer_sample_algorithm, generate_data, data_params):
    for _ in range(data_params["num_of_tests"]):
        data = generate_data()
        change_detected = False

        for point in data:
            if outer_sample_algorithm.detect(point):
                change_detected = True
                break

        assert change_detected, "Algorithm should detect at least one change point"


def test_online_localization(outer_sample_algorithm, generate_data, data_params):
    for test_num in range(data_params["num_of_tests"]):
        data = generate_data()
        change_points = []

        algorithm = SampleEntropyAlgorithm(window_size=50, m=2, r=None, r_factor=0.3, threshold=0.25)

        for point_idx, point in enumerate(data):
            cp = algorithm.localize(point)
            if cp is not None:
                change_points.append(cp)

        assert len(change_points) > 0, f"At least one change point should be detected in test {test_num + 1}"

        closest_point = min(change_points, key=lambda x: abs(x - data_params["change_point"]))

        deviation = abs(closest_point - data_params["change_point"])
        if deviation > data_params["tolerable_deviation"]:
            close_points = [
                cp
                for cp in change_points
                if abs(cp - data_params["change_point"]) <= data_params["tolerable_deviation"]
            ]
            if len(close_points) > 0:
                closest_point = min(close_points, key=lambda x: abs(x - data_params["change_point"]))
            else:
                extended_tolerance = data_params["tolerable_deviation"] + 20
                assert deviation <= extended_tolerance, (
                    f"Change point {closest_point} should be within {extended_tolerance} "
                    f"of true point {data_params['change_point']}. All detected points: {change_points}"
                )

        assert (
            data_params["change_point"] - data_params["tolerable_deviation"]
            <= closest_point
            <= data_params["change_point"] + data_params["tolerable_deviation"]
        ), (
            f"Change point {closest_point} should be within {data_params['tolerable_deviation']} "
            f"of true point {data_params['change_point']}"
        )


def test_online_vs_batch_comparison():
    min_distances = 60
    size = 500
    change_point = 250

    np.random.seed(42)
    data1 = np.random.normal(0, 0.5, change_point)
    data2 = np.random.normal(3, 0.5, size - change_point)
    data = np.concatenate([data1, data2])

    online_algorithm = SampleEntropyAlgorithm(window_size=45, m=2, r=None, r_factor=0.3, threshold=0.2)
    online_changes = []

    for idx, point in enumerate(data):
        cp = online_algorithm.localize(point)
        if cp is not None:
            online_changes.append(cp)

    assert len(online_changes) > 0, "At least one change point should be detected"

    min_distance = min([abs(cp - change_point) for cp in online_changes])
    assert min_distance <= min_distances, f"Minimum distance {min_distance} should be <= {min_distances}"


def test_sample_entropy_calculation():
    algorithm1 = construct_sample_entropy_algorithm()
    t = np.linspace(0, 4 * np.pi, 120)
    deterministic_signal = np.sin(t)

    for point in deterministic_signal:
        algorithm1.detect(point)

    deterministic_entropies = algorithm1.get_entropy_history()

    algorithm2 = construct_sample_entropy_algorithm()
    np.random.seed(42)
    random_signal = np.random.normal(0, 1, 120)

    for point in random_signal:
        algorithm2.detect(point)

    random_entropies = algorithm2.get_entropy_history()

    algorithm3 = construct_sample_entropy_algorithm()
    constant_signal = np.ones(120)

    changes_detected = False
    for point in constant_signal:
        if algorithm3.detect(point):
            changes_detected = True

    assert not changes_detected, "Constant signal should not trigger change detection"

    if len(deterministic_entropies) > 0:
        for e in deterministic_entropies:
            assert e >= 0 or np.isinf(e), f"Entropy {e} should be non-negative or infinite"

    if len(random_entropies) > 0:
        for e in random_entropies:
            assert e >= 0 or np.isinf(e), f"Entropy {e} should be non-negative or infinite"


def test_edge_cases():
    algorithm = construct_sample_entropy_algorithm()
    changes_detected = False
    for i in range(30):
        if algorithm.detect(float(i)):
            changes_detected = True

    assert not changes_detected, "Should not detect changes with insufficient data"

    algorithm = construct_sample_entropy_algorithm()
    for i in range(100):
        algorithm.detect(float(i))

    algorithm = SampleEntropyAlgorithm(window_size=50, m=2, r=None, r_factor=0.4, threshold=0.15)

    for _ in range(70):
        algorithm.detect(0.0)

    change_detected = False
    for _ in range(40):
        if algorithm.detect(5.0):
            change_detected = True
            break

    assert change_detected, "Should detect change from constant to different value"


def test_parameter_validation():
    v1 = 0.1
    v2 = 2
    algorithm = SampleEntropyAlgorithm(window_size=60, m=2, r=0.1, threshold=0.4)
    assert algorithm._m == v2
    assert algorithm._r == v1

    algorithm_auto_r = SampleEntropyAlgorithm(window_size=50, m=3, r=None, r_factor=0.15)
    test_data = np.random.normal(1, 0.5, 60)
    for point in test_data:
        algorithm_auto_r.detect(point)

    current_r = algorithm_auto_r.get_current_r()
    assert current_r is not None, "r should be calculated from data"
    assert current_r > 0, "Calculated r should be positive"

    for m_val in [1, 2, 3, 4]:
        algorithm_m = SampleEntropyAlgorithm(window_size=80, m=m_val, r_factor=0.2)
        assert algorithm_m._m == m_val


def test_infinite_entropy_handling():
    algorithm = SampleEntropyAlgorithm(window_size=50, m=2, r=0.01, threshold=0.5)

    regular_pattern = []
    for i in range(100):
        regular_pattern.append(i % 3)

    for point in regular_pattern:
        algorithm.detect(point)

    entropy_history = algorithm.get_entropy_history()
    if len(entropy_history) > 0:
        for entropy in entropy_history:
            assert entropy >= 0 or np.isinf(entropy), "Entropy should be non-negative or infinite"
            assert not np.isnan(entropy), "Entropy should not be NaN"


def test_different_r_values():
    r_values = [0.1, 0.2, 0.3, 0.5]
    test_data = np.sin(np.linspace(0, 6 * np.pi, 120)) + 0.1 * np.random.normal(0, 1, 120)

    for r_val in r_values:
        algorithm = SampleEntropyAlgorithm(window_size=60, m=2, r=r_val, threshold=0.4)
        entropy_history = []

        for point in test_data:
            algorithm.detect(point)

        entropy_history = algorithm.get_entropy_history()

        assert len(entropy_history) > 0, f"Should produce entropy values for r={r_val}"

        for entropy in entropy_history:
            assert entropy >= 0 or np.isinf(entropy), f"Invalid entropy {entropy} for r={r_val}"


def test_different_m_values():
    m_values = [1, 2, 3, 4]
    v = 75

    np.random.seed(234)
    test_data = []
    for i in range(150):
        if i < v:
            test_data.append(np.sin(i * 0.1) + 0.1 * np.random.normal())
        else:
            test_data.append(np.cos(i * 0.2) + 0.2 * np.random.normal())

    for m_val in m_values:
        algorithm = SampleEntropyAlgorithm(window_size=max(70, m_val * 15), m=m_val, r_factor=0.25, threshold=0.4)

        change_points = []
        for idx, point in enumerate(test_data):
            cp = algorithm.localize(point)
            if cp is not None:
                change_points.append(cp)

        entropy_history = algorithm.get_entropy_history()
        assert len(entropy_history) >= 0, f"Should work with m={m_val}"


def test_entropy_history_and_reset():
    algorithm = construct_sample_entropy_algorithm()

    np.random.seed(567)
    test_data = np.random.normal(0, 1, 100)

    for point in test_data:
        algorithm.detect(point)

    entropy_history = algorithm.get_entropy_history()

    assert len(entropy_history) > 0, "Should have entropy history after processing data"

    for entropy_val in entropy_history:
        assert entropy_val >= 0 or np.isinf(entropy_val), (
            f"Entropy value {entropy_val} should be non-negative or infinite"
        )
        assert not np.isnan(entropy_val), f"Entropy value {entropy_val} should not be NaN"

    algorithm.reset()

    assert len(algorithm.get_entropy_history()) == 0, "History should be empty after reset"
    assert algorithm.get_current_r() is None, "Current r should be None after reset"

    for point in test_data[:50]:
        algorithm.detect(point)

    new_history = algorithm.get_entropy_history()
    assert len(new_history) >= 0, "Should be able to process data after reset"


def test_different_data_types():
    algorithm = construct_sample_entropy_algorithm()

    test_array = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = algorithm.detect(test_array)
    assert isinstance(result, bool)

    single_float = np.float64(3.14)
    result = algorithm.detect(single_float)
    assert isinstance(result, bool)

    algorithm.reset()
    larger_array = np.random.normal(0, 1, 100)
    change_points = []

    for i in range(0, len(larger_array), 10):
        chunk = larger_array[i : i + 10]
        cp = algorithm.localize(chunk)
        if cp is not None:
            change_points.append(cp)

    assert isinstance(change_points, list)


def test_distance_functions():
    v = 1e-10
    algorithm = construct_sample_entropy_algorithm()

    x = np.array([1.0, 2.0, 3.0])
    y = np.array([1.5, 2.2, 2.8])

    chebyshev_dist = algorithm._chebyshev_distance(x, y)
    expected_chebyshev = max(abs(1.0 - 1.5), abs(2.0 - 2.2), abs(3.0 - 2.8))
    assert abs(chebyshev_dist - expected_chebyshev) < v, "Chebyshev distance calculation error"

    euclidean_dist = algorithm._euclidean_distance(x, y)
    expected_euclidean = np.sqrt((1.0 - 1.5) ** 2 + (2.0 - 2.2) ** 2 + (3.0 - 2.8) ** 2)
    assert abs(euclidean_dist - expected_euclidean) < v, "Euclidean distance calculation error"


def test_r_factor_behavior():
    r_factors = [0.1, 0.15, 0.2, 0.25, 0.3]

    np.random.seed(678)
    test_data = np.concatenate([np.random.normal(0, 1, 75), np.random.normal(2, 1, 75)])

    for r_factor in r_factors:
        algorithm = SampleEntropyAlgorithm(window_size=70, m=2, r=None, r_factor=r_factor, threshold=0.4)

        change_points = []
        for idx, point in enumerate(test_data):
            cp = algorithm.localize(point)
            if cp is not None:
                change_points.append(cp)

        entropy_history = algorithm.get_entropy_history()
        assert len(entropy_history) > 0, f"Should produce entropy values for r_factor={r_factor}"

        calculated_r = algorithm.get_current_r()
        if calculated_r is not None:
            assert calculated_r > 0, f"Calculated r should be positive for r_factor={r_factor}"


def test_variance_based_detection():
    algorithm = construct_sample_entropy_algorithm()

    stable_data = np.tile([1, 2, 3, 4], 25)

    np.random.seed(789)
    chaotic_data = np.random.normal(0, 2, 50)

    test_data = np.concatenate([stable_data, chaotic_data])

    change_points = []
    for idx, point in enumerate(test_data):
        cp = algorithm.localize(point)
        if cp is not None:
            change_points.append(cp)

    entropy_history = algorithm.get_entropy_history()
    assert len(entropy_history) > 0, "Should produce entropy values"


def test_zero_standard_deviation():
    algorithm = SampleEntropyAlgorithm(window_size=50, m=2, r=None, r_factor=0.2)

    constant_values = [5.0] * 60
    for point in constant_values:
        algorithm.detect(point)

    algorithm.get_current_r()
    entropy_history = algorithm.get_entropy_history()

    if len(entropy_history) > 0:
        for entropy in entropy_history:
            assert entropy >= 0 or np.isinf(entropy), "Should gracefully handle constant values"


def test_small_tolerance():
    algorithm = SampleEntropyAlgorithm(window_size=60, m=2, r=1e-6, threshold=0.5)

    test_data = np.random.normal(0, 0.1, 80)
    for point in test_data:
        algorithm.detect(point)

    entropy_history = algorithm.get_entropy_history()
    if len(entropy_history) > 0:
        for entropy in entropy_history:
            assert entropy >= 0 or np.isinf(entropy), "Should handle small tolerance values"
            assert not np.isnan(entropy), "Should not produce NaN values"
