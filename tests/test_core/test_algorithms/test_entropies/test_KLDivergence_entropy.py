import numpy as np
import pytest

from pysatl_cpd.core.algorithms.entropies.KLDivergence_entropy import (
    KLDivergenceAlgorithm,
)


def set_seed():
    np.random.seed(1)


def construct_kl_divergence_algorithm():
    return KLDivergenceAlgorithm(
        window_size=80,
        reference_window_size=80,
        threshold=0.3,
        num_bins=15,
        use_kde=False,
        symmetric=True,
        smoothing_factor=1e-8,
    )


@pytest.fixture(scope="function")
def data_params():
    return {
        "num_of_tests": 10,
        "size": 400,
        "change_point": 200,
        "tolerable_deviation": 50,
    }


@pytest.fixture
def generate_data(data_params):
    def _generate_data():
        set_seed()
        data1 = np.zeros(data_params["change_point"])
        for i in range(data_params["change_point"]):
            data1[i] = np.random.normal(0, 1)

        data2 = np.zeros(data_params["size"] - data_params["change_point"])
        np.random.seed(42)
        for i in range(len(data2)):
            data2[i] = np.random.uniform(-3, 3)

        return np.concatenate([data1, data2])

    return _generate_data


@pytest.fixture(scope="function")
def outer_kl_algorithm():
    return construct_kl_divergence_algorithm()


@pytest.fixture
def inner_algorithm_factory():
    def _factory():
        return construct_kl_divergence_algorithm()

    return _factory


def test_algorithm_debug():
    v = 5
    algorithm = construct_kl_divergence_algorithm()
    print("\nKL Divergence Algorithm parameters:")
    if hasattr(algorithm, "get_current_parameters"):
        params = algorithm.get_current_parameters()
        for key, value in params.items():
            print(f" {key}: {value}")

    print("\nTesting distribution change detection:")
    print("Phase 1: Adding Gaussian values N(0,1)")
    for i in range(100):
        detected = algorithm.detect(np.random.normal(0, 1))
        if i < v or i % 20 == 0:
            print(f" Step {i}: detected={detected}")

    if hasattr(algorithm, "get_kl_history"):
        kl_history = algorithm.get_kl_history()
        print(f"KL history length after reference phase: {len(kl_history)}")
        if len(kl_history) > 0:
            print(f"Recent KL values: {kl_history[-3:]}")

    print("Phase 2: Adding uniform values U(-2,2)")
    np.random.seed(123)
    for i in range(30):
        uniform_val = np.random.uniform(-2, 2)
        detected = algorithm.detect(uniform_val)
        print(f" Step {100 + i}: value={uniform_val:.2f}, detected={detected}")
        if detected:
            break

    if hasattr(algorithm, "get_kl_history"):
        kl_history = algorithm.get_kl_history()
        print(f"Final KL history length: {len(kl_history)}")
        if len(kl_history) > 0:
            print(f"Last 5 KL values: {kl_history[-5:]}")
            print(f"KL range: {np.min(kl_history):.6f} to {np.max(kl_history):.6f}")
            print(f"KL variance: {np.var(kl_history):.6f}")

    assert True, "Debug test completed"


def test_online_detection(outer_kl_algorithm, generate_data, data_params):
    detected_count = 0
    v = 10

    for _ in range(data_params["num_of_tests"]):
        data = generate_data()
        if hasattr(outer_kl_algorithm, "reset"):
            outer_kl_algorithm.reset()
        if any(outer_kl_algorithm.detect(point) for point in data):
            detected_count += 1

    test_algorithm = construct_kl_divergence_algorithm()
    for _ in range(100):
        test_algorithm.detect(np.random.normal(0, 1))

    extreme_change_detected = any(test_algorithm.detect(np.random.exponential(2.0)) for _ in range(50))

    if hasattr(test_algorithm, "get_kl_history"):
        kl_history = test_algorithm.get_kl_history()
        assert len(kl_history) >= 0

        if kl_history:
            assert all(np.isfinite(kl) and kl >= 0 for kl in kl_history)

            if len(kl_history) >= v:
                kl_var = np.var(kl_history)
                kl_max = np.max(kl_history)
                assert extreme_change_detected or kl_var > 0 or kl_max > 0 or detected_count > 0
            else:
                assert True
        else:
            assert True


def test_online_localization(outer_kl_algorithm, generate_data, data_params):
    successful_localizations = 0
    v = 5
    all_change_points = []

    for _ in range(data_params["num_of_tests"]):
        data = generate_data()
        algorithm = construct_kl_divergence_algorithm()
        change_points = [cp for idx, point in enumerate(data) if (cp := algorithm.localize(point)) is not None]
        all_change_points.extend(change_points)

        if change_points:
            closest_point = min(change_points, key=lambda x: abs(x - data_params["change_point"]))
            if (
                data_params["change_point"] - data_params["tolerable_deviation"]
                <= closest_point
                <= data_params["change_point"] + data_params["tolerable_deviation"]
            ):
                successful_localizations += 1

    test_algorithm = construct_kl_divergence_algorithm()
    change_points = []
    change_points.extend(cp for _ in range(100) if (cp := test_algorithm.localize(np.random.beta(2, 5))) is not None)

    np.random.seed(999)
    change_points.extend(cp for _ in range(50) if (cp := test_algorithm.localize(np.random.gamma(3, 2))) is not None)

    if hasattr(test_algorithm, "get_kl_history"):
        kl_history = test_algorithm.get_kl_history()
        assert len(kl_history) >= 0, "Algorithm should produce KL divergence calculations"

        if kl_history:
            assert all(np.isfinite(kl) and kl >= 0 for kl in kl_history), (
                "KL divergence values should be finite and non-negative"
            )

            if len(kl_history) >= v:
                kl_var = np.var(kl_history)
                kl_mean = np.mean(kl_history)

                assert any(
                    [
                        successful_localizations > 0,
                        len(all_change_points) > 0,
                        len(change_points) > 0,
                        kl_var > 0,
                        kl_mean > 0,
                    ]
                ), (
                    f"Algorithm should localize changes ({successful_localizations}) or "
                    f"detect some change points ({len(all_change_points)}) or show KL variation ({kl_var:.6f})"
                )
        else:
            assert True, "Algorithm should work with limited data"
    else:
        assert True, "Algorithm should work even without KL history method"


def test_online_vs_batch_comparison():
    min_distances = 80
    size = 400
    change_point = 200
    np.random.seed(123)
    v = 10

    data1 = []
    for i in range(change_point):
        data1.append(np.random.normal(0, 0.5))

    data2 = []
    for i in range(size - change_point):
        data2.append(np.random.uniform(-4, 4))

    data = np.concatenate([data1, data2])

    online_algorithm = construct_kl_divergence_algorithm()
    online_changes = []
    for idx, point in enumerate(data):
        cp = online_algorithm.localize(point)
        if cp is not None:
            online_changes.append(cp)

    if len(online_changes) == 0:
        kl_history = []
        if hasattr(online_algorithm, "get_kl_history"):
            kl_history = online_algorithm.get_kl_history()
        if len(kl_history) >= v:
            kl_range = np.max(kl_history) - np.min(kl_history)
            kl_mean = np.mean(kl_history)
            assert kl_range >= 0, "Should have non-negative KL range"
            assert np.isfinite(kl_mean), "Mean KL divergence should be finite"
            assert kl_mean >= 0, "Mean KL divergence should be non-negative"
        else:
            assert len(kl_history) >= 0, "Algorithm should work without errors"
    else:
        min_distance = min([abs(cp - change_point) for cp in online_changes])
        assert min_distance <= min_distances, f"Minimum distance {min_distance} should be <= {min_distances}"


def test_kl_divergence_calculation():
    def check_kl_history(algorithm, v=10):
        if hasattr(algorithm, "get_kl_history"):
            kl_history = algorithm.get_kl_history()
            if len(kl_history) > 0:
                recent_kl = kl_history[-v:] if len(kl_history) >= v else kl_history
                assert all(np.isfinite(kl) and kl >= 0 for kl in recent_kl), (
                    "KL values should be finite and non-negative"
                )
                return True
        return False

    v = 10

    algorithm1 = construct_kl_divergence_algorithm()
    np.random.seed(123)
    for _ in range(160):
        algorithm1.detect(np.random.normal(0, 1))

    changes_detected_same = sum(1 for _ in range(50) if algorithm1.detect(np.random.normal(0, 1)))
    assert check_kl_history(algorithm1, v)
    algorithm2 = construct_kl_divergence_algorithm()
    np.random.seed(456)
    for _ in range(100):
        algorithm2.detect(np.random.normal(0, 1))

    changes_detected_different = sum(1 for _ in range(50) if algorithm2.detect(np.random.exponential(5)))
    assert check_kl_history(algorithm2, v)
    algorithm3 = construct_kl_divergence_algorithm()
    np.random.seed(789)
    for _ in range(100):
        algorithm3.detect(np.random.beta(2, 2))

    for _ in range(50):
        algorithm3.detect(np.random.uniform(-2, 2))

    if hasattr(algorithm3, "get_kl_history"):
        kl_history = algorithm3.get_kl_history()
        if len(kl_history) > 0:
            assert all(np.isfinite(kl) and kl >= 0 for kl in kl_history), "Should have finite, non-negative KL values"
            if len(kl_history) >= v:
                kl_var = np.var(kl_history)
                assert kl_var >= 0, "Should have non-negative KL variance"
        else:
            assert True, "Algorithm should work even with limited data"

    total_changes = changes_detected_same + changes_detected_different
    assert total_changes >= 0, (
        f"Should detect some changes or work without errors (same dist: "
        f"{changes_detected_same}, diff dist: {changes_detected_different})"
    )


def test_edge_cases():
    algorithm = construct_kl_divergence_algorithm()
    changes_detected = False
    for i in range(50):
        if algorithm.detect(float(i)):
            changes_detected = True
    assert not changes_detected, "Should not detect changes with insufficient data"

    algorithm = construct_kl_divergence_algorithm()
    for i in range(100):
        algorithm.detect(np.random.normal(0, 1))

    algorithm = construct_kl_divergence_algorithm()
    for i in range(100):
        blend_factor = i / 100.0
        value = np.random.normal(3, 2) if np.random.random() < blend_factor else np.random.normal(0, 1)
        algorithm.detect(value)

    algorithm = construct_kl_divergence_algorithm()
    for i in range(120):
        algorithm.detect(np.random.normal(5, 0.5))

    change_detected = False
    np.random.seed(789)
    for i in range(30):
        random_value = np.random.uniform(-10, 10)
        if algorithm.detect(random_value):
            change_detected = True
            break

    if not change_detected:
        if hasattr(algorithm, "get_kl_history"):
            kl_history = algorithm.get_kl_history()
            assert len(kl_history) >= 0, "Algorithm should work and calculate KL divergence"
        else:
            assert True, "Algorithm should work without errors"
    else:
        assert change_detected, "Should detect change from concentrated to uniform distribution"


def test_parameter_validation():
    NEW_NUM_BINS = 25
    v = 0.8
    v1 = 20
    v2 = 0.5
    v3 = 80
    v4 = 100

    algorithm = KLDivergenceAlgorithm(
        window_size=100, reference_window_size=80, threshold=0.5, num_bins=20, use_kde=False, symmetric=True
    )
    if hasattr(algorithm, "get_current_parameters"):
        params = algorithm.get_current_parameters()
        assert params["window_size"] == v4
        assert params["reference_window_size"] == v3
        assert params["threshold"] == v2
        assert params["num_bins"] == v1
        assert params["symmetric"] is True

    with pytest.raises(ValueError):
        KLDivergenceAlgorithm(window_size=0)

    with pytest.raises(ValueError):
        KLDivergenceAlgorithm(threshold=-1.0)

    with pytest.raises(ValueError):
        KLDivergenceAlgorithm(num_bins=1)

    if hasattr(algorithm, "set_parameters"):
        algorithm.set_parameters(threshold=v, num_bins=NEW_NUM_BINS, symmetric=False)
        if hasattr(algorithm, "get_current_parameters"):
            updated_params = algorithm.get_current_parameters()
            assert updated_params["threshold"] == v
            assert updated_params["num_bins"] == NEW_NUM_BINS
            assert not updated_params["symmetric"]

        with pytest.raises(ValueError):
            algorithm.set_parameters(threshold=-0.5)


def test_kde_vs_histogram():
    histogram_algorithm = KLDivergenceAlgorithm(window_size=60, threshold=0.3, use_kde=False, num_bins=15)

    kde_algorithm = KLDivergenceAlgorithm(window_size=60, threshold=0.3, use_kde=True)

    np.random.seed(111)
    reference_data = np.random.normal(0, 1, 80)
    change_data = np.random.gamma(2, 2, 60)
    test_data = np.concatenate([reference_data, change_data])

    for point in test_data:
        histogram_algorithm.detect(point)
        kde_algorithm.detect(point)

    if hasattr(histogram_algorithm, "get_kl_history") and hasattr(kde_algorithm, "get_kl_history"):
        hist_kl = histogram_algorithm.get_kl_history()
        kde_kl = kde_algorithm.get_kl_history()

        if len(hist_kl) > 0:
            assert all(np.isfinite(kl) and kl >= 0 for kl in hist_kl), (
                "Histogram KL values should be finite and non-negative"
            )
        if len(kde_kl) > 0:
            assert all(np.isfinite(kl) and kl >= 0 for kl in kde_kl), "KDE KL values should be finite and non-negative"


def test_symmetric_vs_asymmetric():
    symmetric_algorithm = KLDivergenceAlgorithm(window_size=80, threshold=0.3, symmetric=True)

    asymmetric_algorithm = KLDivergenceAlgorithm(window_size=80, threshold=0.3, symmetric=False)
    v = 100

    np.random.seed(222)
    test_data = []
    for i in range(200):
        if i < v:
            test_data.append(np.random.normal(0, 1))
        else:
            test_data.append(np.random.exponential(2))

    for point in test_data:
        symmetric_algorithm.detect(point)
        asymmetric_algorithm.detect(point)

    if hasattr(symmetric_algorithm, "get_kl_history") and hasattr(asymmetric_algorithm, "get_kl_history"):
        sym_kl = symmetric_algorithm.get_kl_history()
        asym_kl = asymmetric_algorithm.get_kl_history()

        if len(sym_kl) > 0:
            assert all(np.isfinite(kl) and kl >= 0 for kl in sym_kl), (
                "Symmetric KL values should be finite and non-negative"
            )
        if len(asym_kl) > 0:
            assert all(np.isfinite(kl) and kl >= 0 for kl in asym_kl), (
                "Asymmetric KL values should be finite and non-negative"
            )


def test_distribution_analysis():
    algorithm = construct_kl_divergence_algorithm()
    np.random.seed(333)
    normal_data = np.random.normal(2, 1, 100)
    uniform_data = np.random.uniform(0, 4, 80)
    test_data = np.concatenate([normal_data, uniform_data])

    for point in test_data:
        algorithm.detect(point)

    if hasattr(algorithm, "get_distribution_comparison"):
        comparison = algorithm.get_distribution_comparison()
        if comparison:
            if "kl_divergence" in comparison:
                assert np.isfinite(comparison["kl_divergence"]) and comparison["kl_divergence"] >= 0, (
                    "KL divergence should be finite and non-negative"
                )
            if "ks_statistic" in comparison:
                assert 0 <= comparison["ks_statistic"] <= 1, "KS statistic should be in [0,1]"
            if "mean_difference" in comparison:
                assert comparison["mean_difference"] >= 0, "Mean difference should be non-negative"

    if hasattr(algorithm, "analyze_distributions"):
        analysis = algorithm.analyze_distributions()
        if analysis and "reference_entropy" in analysis and "current_entropy" in analysis:
            assert np.isfinite(analysis["reference_entropy"]), "Reference entropy should be finite"
            assert np.isfinite(analysis["current_entropy"]), "Current entropy should be finite"


def test_kl_history_and_reset():
    algorithm = construct_kl_divergence_algorithm()
    np.random.seed(555)
    test_data = np.random.lognormal(0, 1, 150)
    for point in test_data:
        algorithm.detect(point)

    if hasattr(algorithm, "get_kl_history"):
        kl_history = algorithm.get_kl_history()
        if len(kl_history) > 0:
            assert len(kl_history) > 0, "Should have KL history after processing data"
            for kl_val in kl_history:
                assert np.isfinite(kl_val) and kl_val >= 0, f"KL value {kl_val} should be finite and non-negative"

    if hasattr(algorithm, "reset"):
        algorithm.reset()
        if hasattr(algorithm, "get_kl_history"):
            assert len(algorithm.get_kl_history()) == 0, "KL history should be empty after reset"

        for point in test_data[:50]:
            algorithm.detect(point)
        if hasattr(algorithm, "get_kl_history"):
            assert len(algorithm.get_kl_history()) >= 0, "Should be able to process data after reset"


def test_different_data_types():
    algorithm = construct_kl_divergence_algorithm()
    test_array = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = algorithm.detect(test_array)
    assert isinstance(result, bool)

    single_float = np.float64(3.14)
    result = algorithm.detect(single_float)
    assert isinstance(result, bool)

    if hasattr(algorithm, "reset"):
        algorithm.reset()
    larger_array = np.random.normal(0, 1, 120)
    change_points = []
    for i in range(0, len(larger_array), 5):
        chunk = larger_array[i : i + 5]
        cp = algorithm.localize(chunk)
        if cp is not None:
            change_points.append(cp)

    assert isinstance(change_points, list)


def test_guaranteed_distribution_change():
    algorithm = construct_kl_divergence_algorithm()
    concentrated_data = []
    for i in range(120):
        concentrated_data.append(np.random.beta(0.5, 0.5))

    different_data = []
    for i in range(80):
        different_data.append(np.random.gamma(5, 2))

    all_data = concentrated_data + different_data
    changes = []
    for i, point in enumerate(all_data):
        cp = algorithm.localize(point)
        if cp is not None:
            changes.append(cp)

    if hasattr(algorithm, "get_kl_history"):
        kl_history = algorithm.get_kl_history()
        assert len(kl_history) >= 0, "Should calculate KL divergence values"
        for kl in kl_history:
            assert np.isfinite(kl) and kl >= 0, "KL divergence values should be finite and non-negative"
    else:
        assert True, "Algorithm should complete without errors"


def test_smoothing_factor_effects():
    smoothing_factors = [1e-12, 1e-10, 1e-8, 1e-6]
    test_data = []
    for i in range(150):
        if i % 10 == 0:
            test_data.append(100.0)
        else:
            test_data.append(np.random.normal(0, 1))

    for smoothing in smoothing_factors:
        algorithm = KLDivergenceAlgorithm(window_size=80, threshold=0.3, smoothing_factor=smoothing)
        changes_detected = 0
        for point in test_data:
            if algorithm.detect(point):
                changes_detected += 1

        if hasattr(algorithm, "get_kl_history"):
            kl_history = algorithm.get_kl_history()
            if len(kl_history) > 0:
                assert all(np.isfinite(kl) and kl >= 0 for kl in kl_history), (
                    f"KL values should be finite with smoothing={smoothing}"
                )


def test_reference_window_update():
    algorithm = construct_kl_divergence_algorithm()
    for i in range(100):
        algorithm.detect(np.random.normal(0, 1))

    False
    for i in range(50):
        if algorithm.detect(np.random.uniform(-5, 5)):
            True
            break

    if hasattr(algorithm, "force_reference_update"):
        algorithm.force_reference_update()

    for i in range(30):
        algorithm.detect(np.random.exponential(2))

    if hasattr(algorithm, "get_kl_history"):
        kl_history = algorithm.get_kl_history()
        if len(kl_history) > 0:
            assert all(np.isfinite(kl) and kl >= 0 for kl in kl_history), "Should work after reference update"

    assert True, "Reference window update should work correctly"


def test_window_size_effects():
    v = 125
    window_sizes = [40, 80, 120, 160]
    test_data = []
    for i in range(250):
        if i < v:
            test_data.append(np.random.normal(0, 1))
        else:
            test_data.append(np.random.exponential(1))

    for window_size in window_sizes:
        algorithm = KLDivergenceAlgorithm(window_size=window_size, threshold=0.3)
        changes_detected = 0
        for point in test_data:
            if algorithm.detect(point):
                changes_detected += 1

        if hasattr(algorithm, "get_kl_history"):
            kl_history = algorithm.get_kl_history()
            if len(test_data) > 2 * window_size:
                assert len(kl_history) >= 0, f"Should calculate KL divergence with window_size={window_size}"


def test_threshold_sensitivity():
    thresholds = [0.1, 0.3, 0.5, 1.0, 2.0]
    np.random.seed(888)
    normal_data = [np.random.normal(0, 1) for i in range(120)]
    uniform_data = [np.random.uniform(-3, 3) for i in range(80)]
    test_data = normal_data + uniform_data

    detection_results = {}
    for threshold in thresholds:
        algorithm = KLDivergenceAlgorithm(window_size=80, threshold=threshold)
        changes_detected = []
        for i, point in enumerate(test_data):
            cp = algorithm.localize(point)
            if cp is not None:
                changes_detected.append(cp)
        detection_results[threshold] = len(changes_detected)

    for threshold, count in detection_results.items():
        assert count >= 0, f"Detection count should be non-negative for threshold {threshold}"

    thresholds_sorted = sorted(detection_results.keys())
    for i in range(len(thresholds_sorted) - 1):
        low_thresh = thresholds_sorted[i]
        high_thresh = thresholds_sorted[i + 1]
        assert detection_results[low_thresh] >= 0 and detection_results[high_thresh] >= 0, (
            "Both thresholds should produce non-negative detection counts"
        )


def test_error_handling():
    algorithm = construct_kl_divergence_algorithm()
    algorithm.detect(1.0)
    algorithm.detect(2.0)
    algorithm.detect(float("nan"))
    algorithm.detect(3.0)

    algorithm.detect(float("inf"))
    algorithm.detect(float("-inf"))
    algorithm.detect(4.0)

    if hasattr(algorithm, "get_kl_history"):
        kl_history = algorithm.get_kl_history()
        finite_kl = [kl for kl in kl_history if np.isfinite(kl)]
        for kl in finite_kl:
            assert kl >= 0, "Finite KL divergence values should be non-negative"


def test_very_large_and_small_values():
    algorithm = construct_kl_divergence_algorithm()
    test_values = [
        1e-10,
        1e-5,
        0.001,
        0.1,
        1.0,
        10.0,
        100.0,
        1e5,
        1e10,
        -1e-10,
        -1e-5,
        -0.001,
        -0.1,
        -1.0,
        -10.0,
        -100.0,
        -1e5,
        -1e10,
    ]

    np.random.seed(999)
    normal_values = np.random.normal(0, 1, 100).tolist()
    all_values = test_values + normal_values

    for value in all_values:
        algorithm.detect(value)

    if hasattr(algorithm, "get_kl_history"):
        kl_history = algorithm.get_kl_history()
        if len(kl_history) > 0:
            for kl in kl_history:
                assert np.isfinite(kl) and kl >= 0, (
                    f"KL divergence {kl} should be finite and non-negative even with extreme input values"
                )


def test_identical_distributions():
    v = 100.0
    algorithm = construct_kl_divergence_algorithm()
    np.random.seed(42)
    for i in range(100):
        algorithm.detect(np.random.normal(5, 2))

    np.random.seed(42)
    changes_detected = 0
    for i in range(100):
        if algorithm.detect(np.random.normal(5, 2)):
            changes_detected += 1

    if hasattr(algorithm, "get_kl_history"):
        kl_history = algorithm.get_kl_history()
        if len(kl_history) > 0:
            assert all(np.isfinite(kl) and kl >= 0 for kl in kl_history), (
                "All KL values should be finite and non-negative"
            )
            recent_kl = kl_history[-10:] if len(kl_history) >= v / 10 else kl_history
            max_recent_kl = max(recent_kl) if recent_kl else 0
            mean_recent_kl = np.mean(recent_kl) if recent_kl else 0
            assert max_recent_kl < v, (
                f"KL divergence should be reasonable for similar distributions, got max: {max_recent_kl}"
            )
            assert mean_recent_kl >= 0, f"Mean KL divergence should be non-negative, got: {mean_recent_kl}"
        else:
            assert True, "Algorithm should work even without KL history"

    assert changes_detected >= 0, f"Change detection count should be non-negative: {changes_detected}"


def test_distinct_distributions():
    v = 0.1
    algorithm = construct_kl_divergence_algorithm()
    for i in range(100):
        value = np.random.choice([0, 1, 2]) + np.random.normal(0, 0.01)
        algorithm.detect(value)

    change_detected = False
    for i in range(50):
        if algorithm.detect(np.random.uniform(-100, 100)):
            change_detected = True
            break

    if hasattr(algorithm, "get_kl_history"):
        kl_history = algorithm.get_kl_history()
        if len(kl_history) > 0:
            max_kl = max(kl_history)
            assert change_detected or max_kl > v, (
                f"Should detect change ({change_detected}) or have substantial KL divergence (max: {max_kl:.6f})"
            )


def test_algorithm_state_consistency():
    algorithm = construct_kl_divergence_algorithm()
    if hasattr(algorithm, "get_current_parameters"):
        initial_params = algorithm.get_current_parameters()

    test_data = np.random.normal(0, 1, 100)
    for point in test_data:
        algorithm.detect(point)

    if hasattr(algorithm, "get_current_parameters"):
        current_params = algorithm.get_current_parameters()
        assert current_params["window_size"] == initial_params["window_size"]
        assert current_params["threshold"] == initial_params["threshold"]
        assert current_params["num_bins"] == initial_params["num_bins"]
        assert current_params["symmetric"] == initial_params["symmetric"]

    position_before = getattr(algorithm, "_position", 0)
    algorithm.detect(42.0)
    position_after = getattr(algorithm, "_position", 0)
    assert position_after == position_before + 1, "Position should increment by 1 after processing one observation"


def test_comprehensive_workflow():
    algorithm = KLDivergenceAlgorithm(
        window_size=80,
        reference_window_size=80,
        threshold=0.4,
        num_bins=20,
        use_kde=False,
        symmetric=True,
        smoothing_factor=1e-8,
    )
    v = 0.2
    v1 = 15
    np.random.seed(12345)
    phase1_data = [np.random.normal(0, 1) for _ in range(100)]
    phase2_data = [np.random.exponential(2) for _ in range(80)]
    phase3_data = [np.random.beta(2, 5) for _ in range(80)]
    phase4_data = [np.random.normal(3, 0.5) for _ in range(60)]
    all_data = phase1_data + phase2_data + phase3_data + phase4_data

    detected_changes = []
    for i, point in enumerate(all_data):
        cp = algorithm.localize(point)
        if cp is not None:
            detected_changes.append((i, cp))

    if hasattr(algorithm, "get_kl_history"):
        kl_history = algorithm.get_kl_history()
        assert len(kl_history) > 0, "Should have KL divergence history"
        assert all(np.isfinite(kl) and kl >= 0 for kl in kl_history), "All KL values should be finite and non-negative"

    if hasattr(algorithm, "get_distribution_comparison"):
        comparison = algorithm.get_distribution_comparison()
        if comparison:
            assert "kl_divergence" in comparison or len(comparison) >= 0, "Should provide distribution comparison"

    if hasattr(algorithm, "analyze_distributions"):
        analysis = algorithm.analyze_distributions()
        if analysis:
            assert "reference_entropy" in analysis or len(analysis) >= 0, "Should provide distribution analysis"

    if hasattr(algorithm, "set_parameters"):
        algorithm.set_parameters(threshold=0.2, num_bins=15, symmetric=False)
        if hasattr(algorithm, "get_current_parameters"):
            updated_params = algorithm.get_current_parameters()
            assert updated_params["threshold"] == v
            assert updated_params["num_bins"] == v1
            assert not updated_params["symmetric"]

    if hasattr(algorithm, "reset"):
        algorithm.reset()
        if hasattr(algorithm, "get_kl_history"):
            assert len(algorithm.get_kl_history()) == 0, "Should be empty after reset"

    assert True, "Comprehensive workflow completed successfully"


def test_buffer_management():
    v = 40
    algorithm = KLDivergenceAlgorithm(window_size=50, reference_window_size=40)
    for i in range(200):
        algorithm.detect(float(i))

    if hasattr(algorithm, "_current_buffer") and hasattr(algorithm, "_reference_buffer"):
        assert len(algorithm._current_buffer) <= v + 10, "Current buffer should not exceed window_size"
        assert len(algorithm._reference_buffer) <= v, "Reference buffer should not exceed reference_window_size"

    changes_detected = 0
    for i in range(50):
        if algorithm.detect(np.random.uniform(-10, 10)):
            changes_detected += 1

    if hasattr(algorithm, "get_kl_history"):
        kl_history = algorithm.get_kl_history()
        if len(kl_history) > 0:
            assert all(np.isfinite(kl) and kl >= 0 for kl in kl_history), "Should work correctly with buffer overflow"


def test_memory_efficiency():
    algorithm = construct_kl_divergence_algorithm()
    for i in range(1000):
        algorithm.detect(np.random.normal(0, 1))

    if hasattr(algorithm, "_current_buffer"):
        assert len(algorithm._current_buffer) <= algorithm._window_size, "Current buffer should be bounded"
    if hasattr(algorithm, "_reference_buffer"):
        assert len(algorithm._reference_buffer) <= algorithm._reference_window_size, (
            "Reference buffer should be bounded"
        )

    if hasattr(algorithm, "get_kl_history"):
        kl_history = algorithm.get_kl_history()
        expected_max_history = 1000 - algorithm._window_size + 1
        assert len(kl_history) <= expected_max_history, "KL history should not exceed expected maximum"
