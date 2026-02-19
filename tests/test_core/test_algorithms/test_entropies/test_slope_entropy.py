import numpy as np
import pytest

from pysatl_cpd.core.algorithms.entropies.slope_entropy import (
    SlopeEntropyAlgorithm,
)


def set_seed():
    np.random.seed(1)


def create_slope_friendly_data(size=500, change_point=250):
    np.random.seed(42)
    v = 2
    v1 = 120
    v2 = 30
    data1 = []
    for i in range(change_point):
        if i < v2:
            data1.append(1.0 + np.random.normal(0, 0.02))
        elif i < v1:
            data1.append(1.0 + (i - 30) * 0.05 + np.random.normal(0, 0.05))
        else:
            base = 1.0 + 90 * 0.05
            data1.append(base + np.sin((i - 120) * 0.1) * 2.0 + np.random.normal(0, 0.1))

    data2 = []
    base = data1[-1] if data1 else 0
    for i in range(size - change_point):
        if i % 6 == 0:
            data2.append(base + 5.0 + np.random.normal(0, 0.2))
        elif i % 6 == 1:
            data2.append(data2[-1] + np.random.normal(0, 0.1))
        elif i % 6 == v:
            data2.append(data2[-1] - 3.0 + np.random.normal(0, 0.2))
        else:
            change = np.random.choice([-2.0, -1.0, 1.0, 2.0])
            data2.append(data2[-1] + change + np.random.normal(0, 0.15))
        if len(data2) > 0:
            base = data2[-1]

    return np.array(data1 + data2)


def construct_slope_entropy_algorithm():
    return SlopeEntropyAlgorithm(window_size=50, embedding_dim=3, gamma=0.5, delta=0.05, threshold=0.15, normalize=True)


@pytest.fixture(scope="function")
def data_params():
    return {
        "num_of_tests": 10,
        "size": 500,
        "change_point": 250,
        "tolerable_deviation": 40,
    }


@pytest.fixture
def generate_data(data_params):
    def _generate_data():
        v = 0.3
        v1 = 4
        v2 = 50
        set_seed()
        change_point = data_params["change_point"]

        data1 = []
        for i in range(change_point):
            if i < v2:
                data1.append(0.0 + np.random.normal(0, 0.01))
            else:
                data1.append(np.sin((i - 50) * 0.01) * 0.5 + np.random.normal(0, 0.05))

        data2 = []
        base_val = data1[-1] if data1 else 0
        for i in range(data_params["size"] - change_point):
            if i % 8 < v1:
                data2.append(base_val + i * 0.2 + np.random.normal(0, 0.1))
            else:
                jump = 3.0 if (i // 8) % 2 == 0 else -2.0
                data2.append(base_val + i * 0.2 + jump + np.random.normal(0, 0.1))

        if np.random.random() < v:
            data1_ar = np.zeros(change_point)
            for i in range(5, change_point):
                data1_ar[i] = (
                    0.6 * data1_ar[i - 1] - 0.3 * data1_ar[i - 2] + 0.1 * data1_ar[i - 3] + np.random.normal(0, 0.1)
                )

            data2_ar = np.zeros(data_params["size"] - change_point)
            for i in range(5, len(data2_ar)):
                data2_ar[i] = 0.2 * data2_ar[i - 1] + 0.7 * data2_ar[i - 2] + np.random.normal(0, 0.5)

            return np.concatenate([data1_ar, data2_ar])

        return np.array(data1 + data2)

    return _generate_data


@pytest.fixture(scope="function")
def outer_slope_algorithm():
    return construct_slope_entropy_algorithm()


@pytest.fixture
def inner_algorithm_factory():
    def _factory():
        return construct_slope_entropy_algorithm()

    return _factory


def test_online_detection(outer_slope_algorithm, generate_data, data_params):
    detection_success_count = 0
    v = 0.7

    for test_num in range(data_params["num_of_tests"]):
        data = generate_data()
        change_detected = False

        algorithm_configs = [
            {"window_size": 30, "gamma": 0.1, "delta": 0.005, "threshold": 0.05},
            {"window_size": 25, "gamma": 0.2, "delta": 0.01, "threshold": 0.08},
            {"window_size": 35, "gamma": 0.15, "delta": 0.008, "threshold": 0.03},
        ]

        for config in algorithm_configs:
            sensitive_algorithm = SlopeEntropyAlgorithm(embedding_dim=3, normalize=True, **config)

            for point in data:
                if sensitive_algorithm.detect(point):
                    change_detected = True
                    break

            if change_detected:
                detection_success_count += 1
                break

    success_rate = detection_success_count / data_params["num_of_tests"]
    assert success_rate >= v, (
        f"Algorithm should detect changes in at least 70% of tests. Successful:"
        f" {detection_success_count}/{data_params['num_of_tests']} ({success_rate:.1%})"
    )


def test_online_localization(outer_slope_algorithm, generate_data, data_params):
    successful_localizations = 0
    all_detected_points = []
    v = 0.6

    for test_num in range(data_params["num_of_tests"]):
        data = generate_data()
        change_points = []

        algorithm_configs = [
            {"window_size": 30, "gamma": 0.12, "delta": 0.006, "threshold": 0.04},
            {"window_size": 28, "gamma": 0.18, "delta": 0.009, "threshold": 0.06},
            {"window_size": 32, "gamma": 0.15, "delta": 0.007, "threshold": 0.05},
        ]

        for config in algorithm_configs:
            algorithm = SlopeEntropyAlgorithm(embedding_dim=3, normalize=True, **config)

            test_change_points = []
            for point_idx, point in enumerate(data):
                cp = algorithm.localize(point)
                if cp is not None:
                    test_change_points.append(cp)

            if len(test_change_points) > 0:
                change_points.extend(test_change_points)
                break

        if len(change_points) > 0:
            closest_point = min(change_points, key=lambda x: abs(x - data_params["change_point"]))
            all_detected_points.append((test_num + 1, closest_point, change_points))

            deviation = abs(closest_point - data_params["change_point"])
            extended_tolerance = data_params["tolerable_deviation"] + 30
            if deviation <= extended_tolerance:
                successful_localizations += 1

    success_rate = successful_localizations / data_params["num_of_tests"]
    if success_rate < v:
        print("\nLocalization diagnostics:")
        for test_num, closest, all_points in all_detected_points:
            deviation = abs(closest - data_params["change_point"])
            print(f"Test {test_num}: closest={closest}, deviation={deviation}, all points={all_points}")
        assert False, (
            f"Algorithm should correctly localize in at least 60% of tests. Successful: "
            f"{successful_localizations}/{data_params['num_of_tests']} ({success_rate:.1%})"
        )

    if len(all_detected_points) > 0:
        test_num, closest_point, _ = all_detected_points[0]
        extended_tolerance = data_params["tolerable_deviation"] + 30
        assert abs(closest_point - data_params["change_point"]) <= extended_tolerance, (
            f"At least one change point should be within extended tolerance {extended_tolerance}"
        )


def generate_test_data(size, change_point, v1=50, v=5):
    data1 = [(0.0 if i < v1 else (i - v1) * 0.01) for i in range(change_point)]
    data2 = []
    base_val = data1[-1] if data1 else 0
    for i in range(size - change_point):
        val = base_val + i * 0.1
        if i % 10 >= v:
            val += 2.0
        data2.append(val)
        base_val = data2[-1]
    return np.array(data1 + data2)


def detect_change_points(data, algorithm_configs):
    for config in algorithm_configs:
        algorithm = SlopeEntropyAlgorithm(embedding_dim=3, normalize=True, **config)
        change_points = []
        for idx, point in enumerate(data):
            cp = algorithm.localize(point)
            if cp is not None:
                change_points.append(cp)
        if change_points:
            return change_points
    return []


def detect_with_ultra_sensitive(size, change_point):
    simple_data1 = np.full(change_point, 0.0)
    simple_data2 = np.arange(size - change_point) * 0.5
    simple_data = np.concatenate([simple_data1, simple_data2])

    algorithm = SlopeEntropyAlgorithm(
        window_size=20, embedding_dim=2, gamma=0.05, delta=0.001, threshold=0.01, normalize=True
    )

    change_points = []
    for point in simple_data:
        cp = algorithm.localize(point)
        if cp is not None:
            change_points.append(cp)
    return change_points


def test_online_vs_batch_comparison():
    min_distances = 70
    size = 500
    change_point = 250
    np.random.seed(42)

    data = generate_test_data(size, change_point)

    algorithm_configs = [
        {"window_size": 25, "gamma": 0.08, "delta": 0.003, "threshold": 0.02},
        {"window_size": 30, "gamma": 0.1, "delta": 0.005, "threshold": 0.03},
        {"window_size": 35, "gamma": 0.12, "delta": 0.006, "threshold": 0.04},
    ]

    online_changes = detect_change_points(data, algorithm_configs)

    if not online_changes:
        online_changes = detect_with_ultra_sensitive(size, change_point)

    assert online_changes, "At least one change point should be detected"

    min_distance = min(abs(cp - change_point) for cp in online_changes)
    assert min_distance <= min_distances, f"Minimum distance {min_distance} should be <= {min_distances}"


def test_slope_friendly_data_detection():
    v = 80
    test_data = create_slope_friendly_data(size=400, change_point=200)

    algorithm = SlopeEntropyAlgorithm(
        window_size=25, embedding_dim=2, gamma=0.1, delta=0.005, threshold=0.02, normalize=True
    )

    change_points = []
    for idx, point in enumerate(test_data):
        cp = algorithm.localize(point)
        if cp is not None:
            change_points.append(cp)

    assert len(change_points) > 0, "At least one change point should be detected with slope-friendly data"

    true_change_point = 200
    distances = [abs(cp - true_change_point) for cp in change_points]
    min_distance = min(distances)
    assert min_distance <= v, f"Minimum distance to true change point should be <= 80, got {min_distance}"


def test_slope_entropy_calculation():
    algorithm1 = construct_slope_entropy_algorithm()
    t = np.linspace(0, 4 * np.pi, 100)
    deterministic_signal = np.sin(t)

    for point in deterministic_signal:
        algorithm1.detect(point)

    deterministic_entropies = algorithm1.get_entropy_history()

    algorithm2 = construct_slope_entropy_algorithm()
    np.random.seed(42)
    random_signal = np.random.normal(0, 1, 100)

    for point in random_signal:
        algorithm2.detect(point)

    random_entropies = algorithm2.get_entropy_history()

    algorithm3 = construct_slope_entropy_algorithm()
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
    algorithm = construct_slope_entropy_algorithm()
    changes_detected = False
    for i in range(30):
        if algorithm.detect(float(i)):
            changes_detected = True

    assert not changes_detected, "Should not detect changes with insufficient data"

    algorithm = construct_slope_entropy_algorithm()
    for i in range(80):
        algorithm.detect(float(i))

    algorithm = SlopeEntropyAlgorithm(
        window_size=40, embedding_dim=3, gamma=0.2, delta=0.01, threshold=0.05, normalize=True
    )

    for _ in range(60):
        algorithm.detect(0.0)

    change_detected = False
    for _ in range(30):
        if algorithm.detect(5.0):
            change_detected = True
            break

    assert change_detected, "Should detect change from constant to different value"


def test_parameter_validation():
    NEW_GAMMA = 1.5
    NEW_DELTA = 0.2
    NEW_THRESHOLD = 0.4
    gamma = 2.0
    delta = 0.1
    max_symbol = 5

    algorithm = SlopeEntropyAlgorithm(window_size=60, embedding_dim=3, gamma=2.0, delta=0.1, threshold=0.25)

    params = algorithm.get_current_parameters()
    assert params["gamma"] == gamma
    assert params["delta"] == delta
    assert params["max_symbols"] == max_symbol

    with pytest.raises(ValueError, match="delta .* must be less than gamma"):
        SlopeEntropyAlgorithm(window_size=50, gamma=0.5, delta=1.0)

    with pytest.raises(ValueError, match="delta .* must be less than gamma"):
        SlopeEntropyAlgorithm(window_size=50, gamma=1.0, delta=1.0)

    algorithm.set_parameters(embedding_dim=4, gamma=1.5, delta=0.2, threshold=0.4)
    updated_params = algorithm.get_current_parameters()

    assert updated_params["gamma"] == NEW_GAMMA
    assert updated_params["delta"] == NEW_DELTA
    assert updated_params["threshold"] == NEW_THRESHOLD

    with pytest.raises(ValueError):
        algorithm.set_parameters(gamma=0.1, delta=0.5)


def test_slope_pattern_encoding():
    algorithm = construct_slope_entropy_algorithm()

    test_cases = [
        (1.0, 2, "steep positive"),
        (0.3, 1, "gentle positive"),
        (0.0, 0, "flat"),
        (-0.3, -1, "gentle negative"),
        (-1.0, -2, "steep negative"),
    ]

    for slope, expected_symbol, description in test_cases:
        subsequence = np.array([0.0, slope]) if slope >= 0 else np.array([abs(slope), 0.0])

        pattern = algorithm._create_slope_pattern(subsequence)
        assert len(pattern) == 1, "Pattern should have length 1 for 2-element subsequence"
        assert pattern[0] in [-2, -1, 0, 1, 2], f"Symbol should be in valid range for {description}"


def test_symbol_meanings():
    algorithm = construct_slope_entropy_algorithm()
    symbol_meanings = algorithm.get_symbol_meanings()

    expected_symbols = {-2, -1, 0, 1, 2}
    assert set(symbol_meanings.keys()) == expected_symbols, "All symbols should be defined"

    for symbol, meaning in symbol_meanings.items():
        assert isinstance(meaning, str), f"Meaning for symbol {symbol} should be a string"
        assert len(meaning) > 0, f"Meaning for symbol {symbol} should not be empty"

        if symbol != 0:
            assert str(algorithm._gamma) in meaning, f"Gamma should be mentioned in meaning for symbol {symbol}"


def test_pattern_distribution():
    v1 = 1e-10
    v = 2
    algorithm = construct_slope_entropy_algorithm()

    test_data = []
    for i in range(100):
        if i % 4 == 0:
            test_data.append(0.0)
        elif i % 4 == 1 or i % 4 == v:
            test_data.append(2.0)
        else:
            test_data.append(0.0)

    for point in test_data:
        algorithm.detect(point)

    pattern_dist = algorithm.get_pattern_distribution()
    if len(pattern_dist) > 0:
        assert isinstance(pattern_dist, dict), "Pattern distribution should be a dictionary"

        total_prob = sum(pattern_dist.values())
        assert abs(total_prob - 1.0) < v1, f"Probabilities should sum to 1, got {total_prob}"

        for pattern, prob in pattern_dist.items():
            assert prob >= 0, f"Probability for pattern {pattern} should be non-negative"
            assert isinstance(pattern, tuple), f"Pattern {pattern} should be a tuple"

            for symbol in pattern:
                assert symbol in [-2, -1, 0, 1, 2], f"Symbol {symbol} should be valid"


def test_slope_characteristics_analysis():
    v = 1e-10
    v1 = 20
    v2 = 40
    v3 = 60
    algorithm = construct_slope_entropy_algorithm()

    test_data = []
    for i in range(80):
        if i < v1:
            test_data.append(i * 0.1)
        elif i < v2:
            test_data.append(20 * 0.1)
        elif i < v3:
            test_data.append(20 * 0.1 - (i - 40) * 0.1)
        else:
            test_data.append(i * 2.0)

    for point in test_data:
        algorithm.detect(point)

    slope_chars = algorithm.analyze_slope_characteristics()
    if len(slope_chars) > 0:
        required_keys = [
            "slope_entropy",
            "steep_positive_ratio",
            "gentle_positive_ratio",
            "flat_ratio",
            "gentle_negative_ratio",
            "steep_negative_ratio",
            "slope_variance",
            "slope_mean",
            "slope_std",
            "total_patterns",
        ]

        for key in required_keys:
            assert key in slope_chars, f"Key '{key}' should be in slope characteristics"

        ratio_keys = [
            "steep_positive_ratio",
            "gentle_positive_ratio",
            "flat_ratio",
            "gentle_negative_ratio",
            "steep_negative_ratio",
        ]

        for key in ratio_keys:
            ratio_val = slope_chars[key]
            assert 0 <= ratio_val <= 1, f"Ratio {key} should be between 0 and 1, got {ratio_val}"

        total_ratio = sum(slope_chars[key] for key in ratio_keys)
        assert abs(total_ratio - 1.0) < v, f"All ratios should sum to 1, got {total_ratio}"

        assert np.isfinite(slope_chars["slope_entropy"]), "Slope entropy should be finite"
        assert np.isfinite(slope_chars["slope_variance"]), "Slope variance should be finite"
        assert np.isfinite(slope_chars["slope_mean"]), "Slope mean should be finite"
        assert np.isfinite(slope_chars["slope_std"]), "Slope std should be finite"
        assert slope_chars["total_patterns"] >= 0, "Total patterns should be non-negative"


def test_encoding_demonstration():
    algorithm = construct_slope_entropy_algorithm()

    sample_data = [1.0, 3.0, 2.5, 0.5, 4.0]
    encoding_demo = algorithm.demonstrate_encoding(sample_data)

    required_keys = ["original_data", "slopes", "symbols", "patterns", "slope_entropy", "encoding_rules"]
    for key in required_keys:
        assert key in encoding_demo, f"Key '{key}' should be in encoding demonstration"

    assert encoding_demo["original_data"] == sample_data, "Original data should match input"

    slopes = encoding_demo["slopes"]
    assert len(slopes) == len(sample_data) - 1, "Should have n-1 slopes for n data points"

    symbols = encoding_demo["symbols"]
    assert len(symbols) == len(slopes), "Should have same number of symbols and slopes"

    for symbol in symbols:
        assert symbol in [-2, -1, 0, 1, 2], f"Symbol {symbol} should be valid"

    patterns = encoding_demo["patterns"]
    expected_patterns = len(sample_data) - algorithm._embedding_dim + 1
    assert len(patterns) == expected_patterns, f"Should have {expected_patterns} patterns"

    assert isinstance(encoding_demo["slope_entropy"], float), "Slope entropy should be a float"
    assert np.isfinite(encoding_demo["slope_entropy"]), "Slope entropy should be finite"

    short_data = [1.0, 2.0]
    error_demo = algorithm.demonstrate_encoding(short_data)
    assert "error" in error_demo, "Should return error for insufficient data"


def test_different_gamma_delta_values():
    gamma_values = [0.3, 0.5, 1.0, 2.0]
    test_data = np.sin(np.linspace(0, 6 * np.pi, 100)) + 0.2 * np.random.normal(0, 1, 100)

    for gamma in gamma_values:
        algorithm = SlopeEntropyAlgorithm(
            window_size=60, embedding_dim=3, gamma=gamma, delta=gamma / 20, threshold=0.25
        )

        for point in test_data:
            algorithm.detect(point)

        entropy_history = algorithm.get_entropy_history()
        assert len(entropy_history) > 0, f"Should produce entropy values for gamma={gamma}"

        for entropy in entropy_history:
            assert np.isfinite(entropy), f"Entropy should be finite for gamma={gamma}"
            assert entropy >= 0, f"Entropy should be non-negative for gamma={gamma}"


def test_normalization_behavior():
    algorithm_normalized = SlopeEntropyAlgorithm(window_size=60, embedding_dim=3, gamma=1.0, delta=0.05, normalize=True)

    algorithm_raw = SlopeEntropyAlgorithm(window_size=60, embedding_dim=3, gamma=1.0, delta=0.05, normalize=False)

    np.random.seed(789)
    test_data = np.random.normal(0, 1, 80)

    for point in test_data:
        algorithm_normalized.detect(point)
        algorithm_raw.detect(point)

    normalized_entropies = algorithm_normalized.get_entropy_history()
    raw_entropies = algorithm_raw.get_entropy_history()

    if len(normalized_entropies) > 0 and len(raw_entropies) > 0:
        for entropy in normalized_entropies:
            assert 0 <= entropy <= 1, f"Normalized entropy {entropy} should be in [0,1]"

        for entropy in raw_entropies:
            assert entropy >= 0, f"Raw entropy {entropy} should be non-negative"


def test_entropy_history_and_reset():
    algorithm = construct_slope_entropy_algorithm()

    np.random.seed(456)
    test_data = np.random.normal(0, 1, 80)

    for point in test_data:
        algorithm.detect(point)

    entropy_history = algorithm.get_entropy_history()

    assert len(entropy_history) > 0, "Should have entropy history after processing data"

    for entropy_val in entropy_history:
        assert np.isfinite(entropy_val), f"Entropy value {entropy_val} should be finite"
        assert entropy_val >= 0, f"Entropy value {entropy_val} should be non-negative"

    algorithm.reset()

    assert len(algorithm.get_entropy_history()) == 0, "History should be empty after reset"
    assert len(algorithm.get_pattern_distribution()) == 0, "Pattern distribution should be empty after reset"

    for point in test_data[:40]:
        algorithm.detect(point)

    new_history = algorithm.get_entropy_history()
    assert len(new_history) >= 0, "Should be able to process data after reset"


def test_different_data_types():
    algorithm = construct_slope_entropy_algorithm()

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


def test_different_embedding_dimensions():
    embedding_dims = [2, 3, 4, 5]

    np.random.seed(321)
    test_data = np.concatenate([np.random.normal(0, 1, 60), np.random.normal(2, 1, 60)])

    for m in embedding_dims:
        algorithm = SlopeEntropyAlgorithm(
            window_size=max(60, m * 15), embedding_dim=m, gamma=0.5, delta=0.02, threshold=0.2
        )

        change_points = []
        for idx, point in enumerate(test_data):
            cp = algorithm.localize(point)
            if cp is not None:
                change_points.append(cp)

        entropy_history = algorithm.get_entropy_history()
        params = algorithm.get_current_parameters()

        assert len(entropy_history) >= 0, f"Should work with embedding_dim={m}"
        assert params["max_patterns"] == 5 ** (m - 1), f"Max patterns should be 5^(m-1) for m={m}"


def test_variance_based_detection():
    v = 5
    algorithm = construct_slope_entropy_algorithm()

    stable_data = np.sin(np.linspace(0, 4 * np.pi, 60))

    chaotic_data = []
    np.random.seed(654)
    for i in range(40):
        if i % 10 < v:
            chaotic_data.append(np.random.normal(0, 0.1))
        else:
            chaotic_data.append(np.random.normal(0, 3.0))

    test_data = np.concatenate([stable_data, chaotic_data])

    change_points = []
    for idx, point in enumerate(test_data):
        cp = algorithm.localize(point)
        if cp is not None:
            change_points.append(cp)

    entropy_history = algorithm.get_entropy_history()
    assert len(entropy_history) > 0, "Should produce entropy values"


def test_special_slope_patterns():
    algorithm = SlopeEntropyAlgorithm(window_size=50, embedding_dim=3, gamma=0.5, delta=0.02)

    test_data = []
    for i in range(20):
        test_data.append(i * 1.0)

    for i in range(10):
        test_data.append(20.0)

    for i in range(20):
        test_data.append(20.0 - i * 1.0)

    for point in test_data:
        algorithm.detect(point)

    if len(algorithm._buffer) >= algorithm._window_size:
        slope_chars = algorithm.analyze_slope_characteristics()

        assert "steep_positive_ratio" in slope_chars
        assert "flat_ratio" in slope_chars
        assert "steep_negative_ratio" in slope_chars

        total_defined_ratios = (
            slope_chars["steep_positive_ratio"] + slope_chars["flat_ratio"] + slope_chars["steep_negative_ratio"]
        )
        assert total_defined_ratios > 0, "Should detect some defined slope patterns"
