"""Tests for SLA (Service Level Agreement) bounds module."""

import numpy as np
import pytest

from ssbc import (
    ConformalSLAResult,
    OperationalRateBounds,
    compute_conformal_sla,
    compute_conformal_sla_mondrian,
    compute_pac_coverage,
    compute_transfer_cushion,
    cross_fit_cp_bounds,
    transfer_bounds_to_single_rule,
)

# ============================================================================
# Fixtures and Helpers
# ============================================================================


@pytest.fixture
def synthetic_classification_data():
    """Generate synthetic classification data for testing."""
    np.random.seed(42)
    n = 100
    n_classes = 3
    d = 5  # feature dimension

    features = np.random.randn(n, d)
    labels = np.random.randint(0, n_classes, n)

    return features, labels


@pytest.fixture
def simple_score_function():
    """Simple nonconformity score function for testing."""

    def score_fn(x, y):
        """Score based on distance from class center."""
        # Simulate class centers
        class_centers = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0]])
        if y < len(class_centers):
            center = class_centers[y]
        else:
            center = np.zeros(5)
        return float(np.linalg.norm(x - center))

    return score_fn


# ============================================================================
# Test PAC Coverage
# ============================================================================


def test_compute_pac_coverage_basic():
    """Test basic PAC coverage computation."""
    np.random.seed(42)
    scores = np.random.rand(100)
    alpha_target = 0.10
    delta_1 = 0.05

    alpha_adj, threshold, u_star = compute_pac_coverage(scores, alpha_target, delta_1)

    # Basic sanity checks
    assert 0.0 < alpha_adj <= alpha_target
    assert isinstance(threshold, float)
    assert isinstance(u_star, int)
    assert u_star >= 1


def test_compute_pac_coverage_with_small_n():
    """Test PAC coverage with small sample size."""
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    alpha_target = 0.20
    delta_1 = 0.10

    alpha_adj, threshold, u_star = compute_pac_coverage(scores, alpha_target, delta_1)

    assert 0.0 < alpha_adj <= alpha_target
    assert threshold in scores  # Should be one of the calibration scores


def test_compute_pac_coverage_threshold_order():
    """Test that threshold is correctly computed from order statistics."""
    scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    alpha_target = 0.30
    delta_1 = 0.10

    alpha_adj, threshold, u_star = compute_pac_coverage(scores, alpha_target, delta_1)

    # Threshold should be the u_star-th order statistic
    sorted_scores = np.sort(scores)
    assert threshold == sorted_scores[u_star - 1]


# ============================================================================
# Test Cross-Fit CP Bounds
# ============================================================================


def test_cross_fit_cp_bounds_basic(synthetic_classification_data, simple_score_function):
    """Test basic cross-fit CP bounds computation."""
    features, labels = synthetic_classification_data
    score_fn = simple_score_function

    alpha_adj = 0.10
    rate_types = ["singleton", "doublet", "abstention"]
    n_folds = 3
    delta_2 = 0.10

    results = cross_fit_cp_bounds(features, labels, score_fn, alpha_adj, rate_types, n_folds, delta_2, random_seed=42)

    # Check structure
    assert len(results) == len(rate_types)
    for rate_type in rate_types:
        assert "fold_results" in results[rate_type]
        assert "weights" in results[rate_type]
        assert "cf_lower" in results[rate_type]
        assert "cf_upper" in results[rate_type]

        # Check fold results
        assert len(results[rate_type]["fold_results"]) == n_folds
        assert len(results[rate_type]["weights"]) == n_folds

        # Check bounds are valid
        assert 0.0 <= results[rate_type]["cf_lower"] <= 1.0
        assert 0.0 <= results[rate_type]["cf_upper"] <= 1.0
        assert results[rate_type]["cf_lower"] <= results[rate_type]["cf_upper"]


def test_cross_fit_cp_bounds_weights_sum_to_one(synthetic_classification_data, simple_score_function):
    """Test that fold weights sum to 1."""
    features, labels = synthetic_classification_data
    score_fn = simple_score_function

    results = cross_fit_cp_bounds(
        features, labels, score_fn, 0.10, ["singleton"], n_folds=5, delta_2=0.05, random_seed=42
    )

    weights = results["singleton"]["weights"]
    assert abs(sum(weights) - 1.0) < 1e-10


def test_cross_fit_cp_bounds_reproducibility(synthetic_classification_data, simple_score_function):
    """Test that results are reproducible with same seed."""
    features, labels = synthetic_classification_data
    score_fn = simple_score_function

    results1 = cross_fit_cp_bounds(
        features, labels, score_fn, 0.10, ["singleton"], n_folds=3, delta_2=0.05, random_seed=42
    )

    results2 = cross_fit_cp_bounds(
        features, labels, score_fn, 0.10, ["singleton"], n_folds=3, delta_2=0.05, random_seed=42
    )

    # Results should be identical
    assert results1["singleton"]["cf_lower"] == results2["singleton"]["cf_lower"]
    assert results1["singleton"]["cf_upper"] == results2["singleton"]["cf_upper"]


def test_cross_fit_cp_bounds_all_rate_types(synthetic_classification_data, simple_score_function):
    """Test all supported rate types."""
    features, labels = synthetic_classification_data
    score_fn = simple_score_function

    rate_types = ["singleton", "doublet", "abstention", "error_in_singleton"]

    results = cross_fit_cp_bounds(features, labels, score_fn, 0.10, rate_types, n_folds=3, delta_2=0.10, random_seed=42)

    # All rate types should have results
    assert set(results.keys()) == set(rate_types)


# ============================================================================
# Test Transfer Cushion
# ============================================================================


def test_compute_transfer_cushion(synthetic_classification_data, simple_score_function):
    """Test transfer cushion computation."""
    features, labels = synthetic_classification_data
    score_fn = simple_score_function

    # First compute cross-fit results
    cf_results = cross_fit_cp_bounds(
        features, labels, score_fn, 0.10, ["singleton"], n_folds=3, delta_2=0.05, random_seed=42
    )

    # Then compute cushion
    cushion = compute_transfer_cushion(features, labels, score_fn, cf_results["singleton"], "singleton")

    # Cushion should be non-negative and reasonable
    assert 0.0 <= cushion <= 1.0


def test_compute_transfer_cushion_multiple_rates(synthetic_classification_data, simple_score_function):
    """Test cushion computation for multiple rate types."""
    features, labels = synthetic_classification_data
    score_fn = simple_score_function

    rate_types = ["singleton", "doublet", "abstention"]
    cf_results = cross_fit_cp_bounds(
        features, labels, score_fn, 0.10, rate_types, n_folds=3, delta_2=0.10, random_seed=42
    )

    cushions = {}
    for rate_type in rate_types:
        cushions[rate_type] = compute_transfer_cushion(features, labels, score_fn, cf_results[rate_type], rate_type)

    # All cushions should be valid
    for cushion in cushions.values():
        assert 0.0 <= cushion <= 1.0


# ============================================================================
# Test Transfer to Single Rule
# ============================================================================


def test_transfer_bounds_to_single_rule(synthetic_classification_data, simple_score_function):
    """Test transfer of bounds to single rule."""
    features, labels = synthetic_classification_data
    score_fn = simple_score_function

    rate_types = ["singleton", "doublet"]
    cf_results = cross_fit_cp_bounds(
        features, labels, score_fn, 0.10, rate_types, n_folds=3, delta_2=0.10, random_seed=42
    )

    cushions = {rt: compute_transfer_cushion(features, labels, score_fn, cf_results[rt], rt) for rt in rate_types}

    transferred = transfer_bounds_to_single_rule(cf_results, cushions)

    # Check structure
    assert len(transferred) == len(rate_types)
    for rate_type in rate_types:
        assert "single_lower" in transferred[rate_type]
        assert "single_upper" in transferred[rate_type]
        assert "cf_lower" in transferred[rate_type]
        assert "cf_upper" in transferred[rate_type]
        assert "cushion" in transferred[rate_type]

        # Single bounds should be wider than cross-fit bounds (or equal)
        assert transferred[rate_type]["single_lower"] <= transferred[rate_type]["cf_lower"]
        assert transferred[rate_type]["single_upper"] >= transferred[rate_type]["cf_upper"]

        # All bounds should be in [0, 1]
        assert 0.0 <= transferred[rate_type]["single_lower"] <= 1.0
        assert 0.0 <= transferred[rate_type]["single_upper"] <= 1.0


# ============================================================================
# Test Complete SLA Pipeline
# ============================================================================


def test_compute_conformal_sla_basic(synthetic_classification_data, simple_score_function):
    """Test complete SLA pipeline."""
    features, labels = synthetic_classification_data
    score_fn = simple_score_function

    sla_result = compute_conformal_sla(
        features, labels, score_fn, alpha_target=0.10, delta_1=0.05, delta_2=0.05, n_folds=3, random_seed=42
    )

    # Check result type
    assert isinstance(sla_result, ConformalSLAResult)

    # Check coverage parameters
    assert sla_result.alpha_target == 0.10
    assert 0.0 < sla_result.alpha_adjusted <= 0.10
    assert sla_result.coverage_guarantee == 0.90
    assert sla_result.coverage_confidence == 0.95

    # Check rate parameters
    assert sla_result.rate_confidence == 0.95
    assert sla_result.joint_confidence == 0.90

    # Check deployment info
    assert sla_result.n_calibration == len(labels)
    assert sla_result.n_folds == 3
    assert isinstance(sla_result.threshold, float)

    # Check rate bounds
    assert len(sla_result.rate_bounds) == 3  # default: singleton, doublet, abstention
    for rate_name, bounds in sla_result.rate_bounds.items():
        assert isinstance(bounds, OperationalRateBounds)
        assert bounds.rate_name == rate_name
        assert 0.0 <= bounds.lower_bound <= 1.0
        assert 0.0 <= bounds.upper_bound <= 1.0
        assert bounds.lower_bound <= bounds.upper_bound


def test_compute_conformal_sla_custom_rate_types(synthetic_classification_data, simple_score_function):
    """Test SLA with custom rate types."""
    features, labels = synthetic_classification_data
    score_fn = simple_score_function

    custom_rates = ["singleton", "error_in_singleton"]

    sla_result = compute_conformal_sla(
        features,
        labels,
        score_fn,
        alpha_target=0.10,
        delta_1=0.05,
        delta_2=0.05,
        rate_types=custom_rates,
        n_folds=3,
        random_seed=42,
    )

    # Should only have the requested rate types
    assert len(sla_result.rate_bounds) == len(custom_rates)
    assert set(sla_result.rate_bounds.keys()) == set(custom_rates)


def test_compute_conformal_sla_reproducibility(synthetic_classification_data, simple_score_function):
    """Test that SLA results are reproducible."""
    features, labels = synthetic_classification_data
    score_fn = simple_score_function

    sla1 = compute_conformal_sla(
        features, labels, score_fn, alpha_target=0.10, delta_1=0.05, delta_2=0.05, n_folds=3, random_seed=42
    )

    sla2 = compute_conformal_sla(
        features, labels, score_fn, alpha_target=0.10, delta_1=0.05, delta_2=0.05, n_folds=3, random_seed=42
    )

    # Key results should match
    assert sla1.alpha_adjusted == sla2.alpha_adjusted
    assert sla1.threshold == sla2.threshold

    for rate_type in sla1.rate_bounds:
        bounds1 = sla1.rate_bounds[rate_type]
        bounds2 = sla2.rate_bounds[rate_type]
        assert bounds1.lower_bound == bounds2.lower_bound
        assert bounds1.upper_bound == bounds2.upper_bound


# ============================================================================
# Test Mondrian SLA
# ============================================================================


def test_compute_conformal_sla_mondrian(synthetic_classification_data, simple_score_function):
    """Test Mondrian (class-conditional) SLA."""
    features, labels = synthetic_classification_data
    score_fn = simple_score_function

    mondrian_results = compute_conformal_sla_mondrian(
        features, labels, score_fn, alpha_target=0.10, delta_1=0.05, delta_2=0.10, n_folds=3, random_seed=42
    )

    # Should have results for each unique class
    unique_classes = np.unique(labels)
    assert len(mondrian_results) == len(unique_classes)

    # Check each class result
    for class_label in unique_classes:
        assert class_label in mondrian_results
        result = mondrian_results[class_label]
        assert isinstance(result, ConformalSLAResult)

        # Each class should have its own threshold and bounds
        assert isinstance(result.threshold, float)
        assert len(result.rate_bounds) > 0


def test_compute_conformal_sla_mondrian_risk_split(synthetic_classification_data, simple_score_function):
    """Test that Mondrian SLA splits risk appropriately."""
    features, labels = synthetic_classification_data
    score_fn = simple_score_function

    total_delta_2 = 0.15
    mondrian_results = compute_conformal_sla_mondrian(
        features, labels, score_fn, alpha_target=0.10, delta_1=0.05, delta_2=total_delta_2, n_folds=3, random_seed=42
    )

    n_classes = len(mondrian_results)

    # Each class should have rate_confidence = 1 - (total_delta_2 / n_classes)
    expected_confidence = 1 - (total_delta_2 / n_classes)

    for result in mondrian_results.values():
        # Allow small numerical error
        assert abs(result.rate_confidence - expected_confidence) < 1e-10


# ============================================================================
# Integration Tests
# ============================================================================


def test_sla_integration_small_dataset():
    """Integration test with very small dataset."""
    np.random.seed(123)
    n = 20
    features = np.random.randn(n, 3)
    labels = np.random.randint(0, 2, n)

    def score_fn(x, y):
        return float(np.sum(x**2) + y)

    # Should not crash with small dataset
    sla_result = compute_conformal_sla(
        features, labels, score_fn, alpha_target=0.20, delta_1=0.10, delta_2=0.10, n_folds=2, random_seed=123
    )

    assert isinstance(sla_result, ConformalSLAResult)
    assert sla_result.n_calibration == n


def test_sla_integration_different_alphas():
    """Test SLA with different alpha values."""
    np.random.seed(456)
    n = 50
    features = np.random.randn(n, 5)
    labels = np.random.randint(0, 3, n)

    def score_fn(x, y):
        return float(np.linalg.norm(x) - y)

    for alpha in [0.05, 0.10, 0.20]:
        sla_result = compute_conformal_sla(
            features, labels, score_fn, alpha_target=alpha, delta_1=0.05, delta_2=0.05, n_folds=3, random_seed=456
        )

        assert sla_result.alpha_target == alpha
        assert sla_result.coverage_guarantee == 1 - alpha


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


def test_sla_with_single_class():
    """Test SLA when all labels are the same class."""
    np.random.seed(789)
    n = 30
    features = np.random.randn(n, 4)
    labels = np.zeros(n, dtype=int)  # All same class

    def score_fn(x, y):
        return float(np.sum(x**2))

    sla_result = compute_conformal_sla(
        features, labels, score_fn, alpha_target=0.10, delta_1=0.05, delta_2=0.05, n_folds=2, random_seed=789
    )

    assert isinstance(sla_result, ConformalSLAResult)


def test_operational_rate_bounds_dataclass():
    """Test OperationalRateBounds dataclass."""
    bounds = OperationalRateBounds(
        rate_name="singleton",
        lower_bound=0.3,
        upper_bound=0.7,
        cross_fit_lower=0.35,
        cross_fit_upper=0.65,
        cushion=0.05,
        confidence_level=0.95,
        fold_results=[],
    )

    assert bounds.rate_name == "singleton"
    assert bounds.lower_bound == 0.3
    assert bounds.upper_bound == 0.7
    assert bounds.confidence_level == 0.95


def test_conformal_sla_result_dataclass():
    """Test ConformalSLAResult dataclass."""
    result = ConformalSLAResult(
        alpha_target=0.10,
        alpha_adjusted=0.08,
        coverage_guarantee=0.90,
        coverage_confidence=0.95,
        rate_bounds={},
        rate_confidence=0.95,
        joint_confidence=0.90,
        threshold=1.5,
        n_calibration=100,
        n_folds=5,
    )

    assert result.alpha_target == 0.10
    assert result.coverage_guarantee == 0.90
    assert result.joint_confidence == 0.90
    assert result.n_calibration == 100
