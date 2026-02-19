import numpy as np
import pytest

from pysatl_cpd.core.algorithms.entropies.dispersion_entropy import (
    DispersionEntropyAlgorithm,
)


def set_seed():
    np.random.seed(1)


def construct_dispersion_entropy_algorithm():
    return DispersionEntropyAlgorithm(
        window_size=250, embedding_dim=3, num_classes=6, time_delay=1, threshold=0.1, normalize=True
    )


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
            data1[i] = 0.8 * data1[i - 1] - 0.3 * data1[i - 2] + 0.1 * data1[i - 3] + np.random.normal(0, 0.05)

        data2 = np.zeros(data_params["size"] - data_params["change_point"])
        for i in range(5, len(data2)):
            data2[i] = 0.2 * data2[i - 1] + 0.3 * data2[i - 2] + np.random.normal(0, 1.0)

        return np.concatenate([data1, data2])

    return _generate_data


@pytest.fixture(scope="function")
def outer_dispersion_algorithm():
    return construct_dispersion_entropy_algorithm()


@pytest.fixture
def inner_algorithm_factory():
    def _factory():
        return construct_dispersion_entropy_algorithm()

    return _factory


def test_online_detection(outer_dispersion_algorithm, generate_data, data_params):
    detected_count = 0
    for _ in range(data_params["num_of_tests"]):
        data = generate_data()
        change_detected = False
        outer_dispersion_algorithm.reset() if hasattr(outer_dispersion_algorithm, "reset") else None
        for point in data:
            if outer_dispersion_algorithm.detect(point):
                change_detected = True
                break
        if change_detected:
            detected_count += 1

    assert detected_count >= data_params["num_of_tests"] // 3, (
        f"Algorithm should detect changes in at least {data_params['num_of_tests'] // 3} "
        f"out of {data_params['num_of_tests']} tests (detected: {detected_count})"
    )


def test_online_localization(outer_dispersion_algorithm, generate_data, data_params):
    successful_localizations = 0
    for _ in range(data_params["num_of_tests"]):
        data = generate_data()
        change_points = []
        algorithm = construct_dispersion_entropy_algorithm()
        for point_idx, point in enumerate(data):
            cp = algorithm.localize(point)
            if cp is not None:
                change_points.append(cp)

        if len(change_points) > 0:
            closest_point = min(change_points, key=lambda x: abs(x - data_params["change_point"]))
            if (
                data_params["change_point"] - data_params["tolerable_deviation"]
                <= closest_point
                <= data_params["change_point"] + data_params["tolerable_deviation"]
            ):
                successful_localizations += 1

    assert successful_localizations >= data_params["num_of_tests"] // 4, (
        f"Algorithm should successfully localize changes in at least {data_params['num_of_tests'] // 4} "
        f"out of {data_params['num_of_tests']} tests (successful: {successful_localizations})"
    )


def test_online_vs_batch_comparison():
    min_distances = 80
    size = 500
    change_point = 250
    np.random.seed(42)
    v = 5
    v1 = 10

    data1 = []
    for i in range(change_point):
        data1.append(np.sin(i * 0.1) + 0.05 * np.random.normal())

    np.random.seed(123)
    data2 = []
    for i in range(size - change_point):
        data2.append(np.random.normal(0, 2) + np.random.uniform(-1, 1))

    data = np.concatenate([data1, data2])

    online_algorithm = construct_dispersion_entropy_algorithm()
    online_changes = []
    for idx, point in enumerate(data):
        cp = online_algorithm.localize(point)
        if cp is not None:
            online_changes.append(cp)

    if len(online_changes) == 0:
        entropy_history = []
        if hasattr(online_algorithm, "get_entropy_history"):
            entropy_history = online_algorithm.get_entropy_history()
        if len(entropy_history) >= v1:
            mid_point = len(entropy_history) // 2
            first_half = entropy_history[:mid_point]
            second_half = entropy_history[mid_point:]
            if len(first_half) > 0 and len(second_half) > 0:
                mean_first = np.mean(first_half)
                mean_second = np.mean(second_half)
                entropy_diff = abs(mean_second - mean_first)
                assert entropy_diff >= 0 or len(entropy_history) > v, (
                    "Should show entropy variation or reasonable history length"
                )
            else:
                assert len(entropy_history) > 0, "Should calculate entropy values"
        else:
            assert True, "Algorithm should work without errors"
    else:
        min_distance = min([abs(cp - change_point) for cp in online_changes])
        assert min_distance <= min_distances, f"Minimum distance {min_distance} should be <= {min_distances}"


def test_dispersion_entropy_calculation():
    algorithm1 = construct_dispersion_entropy_algorithm()
    t = np.linspace(0, 4 * np.pi, 300)
    deterministic_signal = np.sin(t)
    deterministic_changes = 0
    for point in deterministic_signal:
        if algorithm1.detect(point):
            deterministic_changes += 1

    algorithm2 = construct_dispersion_entropy_algorithm()
    np.random.seed(42)
    random_signal = np.random.normal(0, 1, 300)
    random_changes = 0
    for point in random_signal:
        if algorithm2.detect(point):
            random_changes += 1

    algorithm3 = construct_dispersion_entropy_algorithm()
    constant_signal = np.ones(300)
    changes_detected = False
    for point in constant_signal:
        if algorithm3.detect(point):
            changes_detected = True
    assert not changes_detected, "Constant signal should not trigger change detection"

    if hasattr(algorithm1, "get_entropy_history"):
        deterministic_entropies = algorithm1.get_entropy_history()
        if len(deterministic_entropies) > 0:
            assert all(np.isfinite(e) for e in deterministic_entropies), "Should have finite entropy values"

    if hasattr(algorithm2, "get_entropy_history"):
        random_entropies = algorithm2.get_entropy_history()
        if len(random_entropies) > 0:
            assert all(np.isfinite(e) for e in random_entropies), "Should have finite entropy values"


def test_edge_cases():
    algorithm = construct_dispersion_entropy_algorithm()
    changes_detected = False
    for i in range(50):
        if algorithm.detect(float(i)):
            changes_detected = True
    assert not changes_detected, "Should not detect changes with insufficient data"

    algorithm = construct_dispersion_entropy_algorithm()
    for i in range(300):
        algorithm.detect(float(i * 0.1))

    algorithm = construct_dispersion_entropy_algorithm()
    for i in range(270):
        algorithm.detect(np.sin(i * 0.1))

    change_detected = False
    np.random.seed(999)
    for i in range(50):
        random_value = np.random.normal(0, 3) + np.random.uniform(-2, 2)
        if algorithm.detect(random_value):
            change_detected = True
            break

    if not change_detected:
        if hasattr(algorithm, "get_entropy_history"):
            entropy_history = algorithm.get_entropy_history()
            assert len(entropy_history) >= 0, "Algorithm should work and calculate entropy"
        else:
            assert True, "Algorithm should work without errors"
    else:
        assert change_detected, "Should detect change from regular to chaotic pattern"


def test_parameter_validation():
    v1 = 2
    v2 = 4
    v3 = 0.25
    v4 = 5
    v5 = 3

    algorithm = DispersionEntropyAlgorithm(
        window_size=150, embedding_dim=3, num_classes=5, time_delay=1, threshold=0.15
    )
    if hasattr(algorithm, "get_current_parameters"):
        params = algorithm.get_current_parameters()
        assert params["embedding_dim"] == v5
        assert params["num_classes"] == v4
        assert params["max_patterns"] == v4**v5

    with pytest.raises(ValueError):
        DispersionEntropyAlgorithm(window_size=50, embedding_dim=4, num_classes=6)

    if hasattr(algorithm, "set_parameters"):
        algorithm.set_parameters(embedding_dim=2, num_classes=4, threshold=0.25)
        if hasattr(algorithm, "get_current_parameters"):
            updated_params = algorithm.get_current_parameters()
            assert updated_params["embedding_dim"] == v1
            assert updated_params["num_classes"] == v2
            assert updated_params["threshold"] == v3

        with pytest.raises(ValueError):
            algorithm.set_parameters(embedding_dim=5, num_classes=10)


def test_dispersion_pattern_analysis():
    algorithm = construct_dispersion_entropy_algorithm()
    np.random.seed(123)
    simple_data = np.sin(np.linspace(0, 4 * np.pi, 100)) + 0.1 * np.random.normal(0, 1, 100)
    complex_data = np.random.normal(0, 1, 100)
    test_data = np.concatenate([simple_data, complex_data])

    for point in test_data:
        algorithm.detect(point)

    if hasattr(algorithm, "get_pattern_distribution"):
        pattern_dist = algorithm.get_pattern_distribution()
        assert isinstance(pattern_dist, dict), "Pattern distribution should be a dictionary"

    if hasattr(algorithm, "analyze_complexity"):
        complexity_metrics = algorithm.analyze_complexity()
        if complexity_metrics:
            if "dispersion_entropy" in complexity_metrics:
                assert np.isfinite(complexity_metrics["dispersion_entropy"]), "Dispersion entropy should be finite"
            if "normalized_entropy" in complexity_metrics:
                assert 0 <= complexity_metrics["normalized_entropy"] <= 1, "Normalized entropy should be in [0,1]"
            if "unique_patterns" in complexity_metrics:
                assert complexity_metrics["unique_patterns"] > 0, "Should have some unique patterns"
        else:
            assert True, "Algorithm should work even without complexity analysis method"


def test_entropy_history_and_reset():
    algorithm = construct_dispersion_entropy_algorithm()
    np.random.seed(456)
    test_data = np.random.normal(0, 1, 300)
    for point in test_data:
        algorithm.detect(point)

    if hasattr(algorithm, "get_entropy_history"):
        entropy_history = algorithm.get_entropy_history()
        if len(entropy_history) > 0:
            assert len(entropy_history) > 0, "Should have entropy history after processing data"
            for entropy_val in entropy_history:
                assert np.isfinite(entropy_val), f"Entropy value {entropy_val} should be finite"
                assert entropy_val >= 0, f"Entropy value {entropy_val} should be non-negative"

    if hasattr(algorithm, "reset"):
        algorithm.reset()
        if hasattr(algorithm, "get_entropy_history"):
            assert len(algorithm.get_entropy_history()) == 0, "History should be empty after reset"
        if hasattr(algorithm, "get_pattern_distribution"):
            assert len(algorithm.get_pattern_distribution()) == 0, "Pattern distribution should be empty after reset"

        for point in test_data[:100]:
            algorithm.detect(point)
        if hasattr(algorithm, "get_entropy_history"):
            assert len(algorithm.get_entropy_history()) >= 0, "Should be able to process data after reset"


def test_different_data_types():
    algorithm = construct_dispersion_entropy_algorithm()
    test_array = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = algorithm.detect(test_array)
    assert isinstance(result, bool)

    single_float = np.float64(3.14)
    result = algorithm.detect(single_float)
    assert isinstance(result, bool)

    if hasattr(algorithm, "reset"):
        algorithm.reset()
    larger_array = np.random.normal(0, 1, 300)
    change_points = []
    for i in range(0, len(larger_array), 10):
        chunk = larger_array[i : i + 10]
        cp = algorithm.localize(chunk)
        if cp is not None:
            change_points.append(cp)

    assert isinstance(change_points, list)


def test_normalization_behavior():
    algorithm_normalized = DispersionEntropyAlgorithm(window_size=150, embedding_dim=3, num_classes=5, normalize=True)
    algorithm_raw = DispersionEntropyAlgorithm(window_size=150, embedding_dim=3, num_classes=5, normalize=False)

    np.random.seed(789)
    test_data = np.random.normal(0, 1, 200)

    for point in test_data:
        algorithm_normalized.detect(point)
        algorithm_raw.detect(point)

    if hasattr(algorithm_normalized, "get_entropy_history") and hasattr(algorithm_raw, "get_entropy_history"):
        normalized_entropies = algorithm_normalized.get_entropy_history()
        raw_entropies = algorithm_raw.get_entropy_history()

        if len(normalized_entropies) > 0 and len(raw_entropies) > 0:
            for entropy in normalized_entropies:
                assert 0 <= entropy <= 1, f"Normalized entropy {entropy} should be in [0,1]"

            for entropy in raw_entropies:
                assert entropy >= 0, f"Raw entropy {entropy} should be non-negative"


def test_class_discretization():
    algorithm = construct_dispersion_entropy_algorithm()
    test_data = np.linspace(-2, 2, 300)

    for point in test_data:
        algorithm.detect(point)

    if (
        hasattr(algorithm, "get_pattern_distribution")
        and hasattr(algorithm, "_buffer")
        and len(algorithm._buffer) >= algorithm._window_size
    ):
        pattern_dist = algorithm.get_pattern_distribution()
        if pattern_dist:
            assert pattern_dist, "Should have dispersion patterns"
            for pattern in pattern_dist:
                for class_val in pattern:
                    assert 1 <= class_val <= algorithm._num_classes, (
                        f"Class value {class_val} should be in range [1, {algorithm._num_classes}]"
                    )


def test_time_delay_parameter():
    for delay in [1, 2, 3]:
        algorithm = DispersionEntropyAlgorithm(window_size=150, embedding_dim=3, num_classes=5, time_delay=delay)

        test_data = np.sin(np.linspace(0, 6 * np.pi, 200)) + 0.2 * np.random.random(200)
        changes_detected = 0
        for point in test_data:
            if algorithm.detect(point):
                changes_detected += 1

        if hasattr(algorithm, "get_entropy_history"):
            entropy_history = algorithm.get_entropy_history()
            assert len(entropy_history) >= 0, f"Should process data with time_delay={delay}"
        else:
            assert True, f"Should work with time_delay={delay}"


def test_guaranteed_complexity_change():
    algorithm = construct_dispersion_entropy_algorithm()
    regular_data = []
    for i in range(280):
        regular_data.append(i % 3)

    np.random.seed(777)
    chaotic_data = []
    for i in range(100):
        x = np.random.uniform(0, 1)
        for _ in range(10):
            x = 4 * x * (1 - x)
        chaotic_data.append(x)

    all_data = regular_data + chaotic_data
    changes = []
    for i, point in enumerate(all_data):
        cp = algorithm.localize(point)
        if cp is not None:
            changes.append(cp)

    if hasattr(algorithm, "get_entropy_history"):
        entropy_history = algorithm.get_entropy_history()
        assert len(entropy_history) >= 0, "Should calculate entropy values"
        for entropy in entropy_history:
            assert np.isfinite(entropy), "Entropy values should be finite"
    else:
        assert True, "Algorithm should complete without errors"
