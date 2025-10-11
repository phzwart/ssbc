"""Tests for SLA (Service Level Agreement) operational bounds module."""

import numpy as np
import pytest

from ssbc import (
    BinaryClassifierSimulator,
    OperationalRateBounds,
    OperationalRateBoundsResult,
    compute_marginal_operational_bounds,
    compute_mondrian_operational_bounds,
    compute_transfer_cushion,
    cross_fit_cp_bounds,
    mondrian_conformal_calibrate,
    split_by_class,
    transfer_bounds_to_single_rule,
)

# ============================================================================
# Fixtures and Helpers
# ============================================================================


@pytest.fixture
def binary_classification_data():
    """Generate binary classification data with probabilities."""
    np.random.seed(42)
    sim = BinaryClassifierSimulator(p_class1=0.5, beta_params_class0=(3, 7), beta_params_class1=(7, 3), seed=42)
    labels, probs = sim.generate(n_samples=100)
    class_data = split_by_class(labels, probs)
    return labels, probs, class_data


# ============================================================================
# Test Cross-Fit CP Bounds
# ============================================================================


def test_cross_fit_cp_bounds_basic(binary_classification_data):
    """Test basic cross-fit CP bounds computation."""
    _, _, class_data = binary_classification_data

    # Test on class 0
    results = cross_fit_cp_bounds(
        class_data=class_data[0],
        true_class=0,
        alpha_target=0.10,
        delta_class=0.05,
        rate_types=["singleton", "doublet"],
        n_folds=5,
        delta=0.10,
        random_seed=42,
    )

    # Check structure
    assert "singleton" in results
    assert "doublet" in results

    for rate_type in ["singleton", "doublet"]:
        assert "fold_results" in results[rate_type]
        assert "weights" in results[rate_type]
        assert "cf_lower" in results[rate_type]
        assert "cf_upper" in results[rate_type]

        # Check bounds are valid
        assert 0.0 <= results[rate_type]["cf_lower"] <= results[rate_type]["cf_upper"] <= 1.0


def test_cross_fit_cp_bounds_weights_sum_to_one(binary_classification_data):
    """Test that fold weights sum to approximately 1."""
    _, _, class_data = binary_classification_data

    results = cross_fit_cp_bounds(class_data[0], 0, 0.10, 0.05, ["singleton"], 5, 0.10, random_seed=42)

    weights = results["singleton"]["weights"]
    assert np.isclose(sum(weights), 1.0, atol=1e-10)


def test_cross_fit_cp_bounds_reproducibility(binary_classification_data):
    """Test reproducibility with same random seed."""
    _, _, class_data = binary_classification_data

    results1 = cross_fit_cp_bounds(class_data[0], 0, 0.10, 0.05, ["singleton"], 5, 0.10, random_seed=42)

    results2 = cross_fit_cp_bounds(class_data[0], 0, 0.10, 0.05, ["singleton"], 5, 0.10, random_seed=42)

    assert np.isclose(results1["singleton"]["cf_lower"], results2["singleton"]["cf_lower"])
    assert np.isclose(results1["singleton"]["cf_upper"], results2["singleton"]["cf_upper"])


# ============================================================================
# Test Transfer Cushion
# ============================================================================


def test_compute_transfer_cushion(binary_classification_data):
    """Test transfer cushion computation."""
    _, _, class_data = binary_classification_data

    cf_results = cross_fit_cp_bounds(class_data[0], 0, 0.10, 0.05, ["singleton"], 3, 0.10, random_seed=42)

    cushion = compute_transfer_cushion(class_data[0], 0, cf_results["singleton"], "singleton")

    # Cushion should be non-negative and reasonable
    assert 0.0 <= cushion <= 1.0


def test_transfer_bounds_to_single_rule(binary_classification_data):
    """Test transferring bounds to single rule."""
    _, _, class_data = binary_classification_data

    cf_results = cross_fit_cp_bounds(class_data[0], 0, 0.10, 0.05, ["singleton", "doublet"], 3, 0.10, random_seed=42)

    cushions = {
        "singleton": compute_transfer_cushion(class_data[0], 0, cf_results["singleton"], "singleton"),
        "doublet": compute_transfer_cushion(class_data[0], 0, cf_results["doublet"], "doublet"),
    }

    transferred = transfer_bounds_to_single_rule(cf_results, cushions)

    # Check structure
    assert "singleton" in transferred
    assert "doublet" in transferred

    for rate_type in ["singleton", "doublet"]:
        assert "single_lower" in transferred[rate_type]
        assert "single_upper" in transferred[rate_type]
        assert "cf_lower" in transferred[rate_type]
        assert "cf_upper" in transferred[rate_type]
        assert "cushion" in transferred[rate_type]

        # Check bounds are widened appropriately
        assert transferred[rate_type]["single_lower"] <= transferred[rate_type]["cf_lower"]
        assert transferred[rate_type]["single_upper"] >= transferred[rate_type]["cf_upper"]


# ============================================================================
# Test Mondrian Operational Bounds
# ============================================================================


def test_compute_mondrian_operational_bounds(binary_classification_data):
    """Test Mondrian operational bounds computation."""
    labels, probs, class_data = binary_classification_data

    # First get Mondrian calibration
    cal_result, _ = mondrian_conformal_calibrate(class_data, alpha_target=0.10, delta=0.05, mode="beta")

    # Then compute operational bounds using labels, probs
    results = compute_mondrian_operational_bounds(
        calibration_result=cal_result,
        labels=labels,
        probs=probs,
        delta=0.10,
        rate_types=["singleton", "doublet"],
        n_folds=3,
        random_seed=42,
    )

    # Check structure
    assert isinstance(results, dict)
    assert 0 in results
    assert 1 in results

    for class_label in [0, 1]:
        result = results[class_label]
        assert isinstance(result, OperationalRateBoundsResult)
        assert "singleton" in result.rate_bounds
        assert "doublet" in result.rate_bounds


def test_compute_mondrian_operational_bounds_uses_mondrian_params(binary_classification_data):
    """Test that operational bounds use alpha/delta from calibration."""
    labels, probs, class_data = binary_classification_data

    # Mondrian calibration
    cal_result, _ = mondrian_conformal_calibrate(class_data, alpha_target=0.10, delta=0.05)

    # Operational bounds
    results = compute_mondrian_operational_bounds(cal_result, labels, probs, delta=0.10, n_folds=3, random_seed=42)

    # Check that thresholds match
    for class_label in [0, 1]:
        assert results[class_label].thresholds == cal_result[class_label]["threshold"]


# ============================================================================
# Test Integration
# ============================================================================


def test_operational_bounds_integration_small_dataset():
    """Test operational bounds with small dataset."""
    np.random.seed(123)
    sim = BinaryClassifierSimulator(p_class1=0.5, beta_params_class0=(2, 8), beta_params_class1=(8, 2), seed=123)
    labels, probs = sim.generate(n_samples=30)
    class_data = split_by_class(labels, probs)

    # Mondrian calibration
    cal_result, _ = mondrian_conformal_calibrate(class_data, alpha_target=0.10, delta=0.10)

    # Operational bounds
    results = compute_mondrian_operational_bounds(cal_result, labels, probs, delta=0.10, n_folds=3, random_seed=42)

    assert isinstance(results, dict)
    assert len(results) > 0  # At least one class should have results


def test_mondrian_integration_full_workflow():
    """Test complete Mondrian + operational bounds workflow."""
    np.random.seed(42)
    sim = BinaryClassifierSimulator(p_class1=0.5, beta_params_class0=(2, 8), beta_params_class1=(8, 2), seed=42)
    labels, probs = sim.generate(n_samples=100)

    # Step 1: Split data
    class_data = split_by_class(labels, probs)

    # Step 2: Mondrian calibration (PAC coverage)
    cal_result, pred_stats = mondrian_conformal_calibrate(class_data, alpha_target=0.10, delta=0.05)

    # Step 3: Operational bounds (marginal CV, then per-class)
    operational_bounds = compute_mondrian_operational_bounds(
        cal_result, labels, probs, delta=0.05, n_folds=5, random_seed=42
    )

    # Verify complete workflow
    assert len(operational_bounds) == 2  # Two classes
    for class_label in [0, 1]:
        # Check PAC coverage from Mondrian
        assert "alpha_corrected" in cal_result[class_label]
        assert "threshold" in cal_result[class_label]

        # Check operational bounds
        assert class_label in operational_bounds
        assert "singleton" in operational_bounds[class_label].rate_bounds


def test_different_alpha_per_class():
    """Test with different alpha targets per class."""
    np.random.seed(42)
    sim = BinaryClassifierSimulator(p_class1=0.5, beta_params_class0=(3, 7), beta_params_class1=(7, 3), seed=42)
    labels, probs = sim.generate(n_samples=100)
    class_data = split_by_class(labels, probs)

    # Different alpha per class
    cal_result, _ = mondrian_conformal_calibrate(class_data, alpha_target={0: 0.05, 1: 0.10}, delta=0.05)

    # Operational bounds
    operational_bounds = compute_mondrian_operational_bounds(
        cal_result, labels, probs, delta=0.05, n_folds=5, random_seed=42
    )

    assert len(operational_bounds) == 2
    # Verify different alphas were used
    assert cal_result[0]["alpha_target"] == 0.05
    assert cal_result[1]["alpha_target"] == 0.10


# ============================================================================
# Test Dataclasses
# ============================================================================


def test_operational_rate_bounds_dataclass():
    """Test OperationalRateBounds dataclass."""
    bounds = OperationalRateBounds(
        rate_name="singleton",
        lower_bound=0.6,
        upper_bound=0.85,
        cross_fit_lower=0.65,
        cross_fit_upper=0.80,
        cushion=0.05,
        ci_width=0.95,
        fold_results=[{"fold": 0, "m_f": 20}],
    )

    assert bounds.rate_name == "singleton"
    assert bounds.lower_bound == 0.6
    assert bounds.upper_bound == 0.85
    assert bounds.cushion == 0.05


def test_operational_bounds_result_dataclass():
    """Test OperationalRateBoundsResult dataclass."""
    singleton_bounds = OperationalRateBounds(
        rate_name="singleton",
        lower_bound=0.7,
        upper_bound=0.9,
        cross_fit_lower=0.75,
        cross_fit_upper=0.85,
        cushion=0.05,
        ci_width=0.95,
        fold_results=[],
    )

    result = OperationalRateBoundsResult(
        rate_bounds={"singleton": singleton_bounds},
        rate_confidence=0.95,
        thresholds=0.5,
        n_calibration=100,
        n_folds=5,
    )

    assert result.rate_confidence == 0.95
    assert result.thresholds == 0.5
    assert result.n_calibration == 100
    assert result.n_folds == 5
    assert "singleton" in result.rate_bounds


def test_custom_confidence_level():
    """Test that custom confidence_level parameter works."""
    np.random.seed(42)
    sim = BinaryClassifierSimulator(p_class1=0.5, beta_params_class0=(3, 7), beta_params_class1=(7, 3), seed=42)
    labels, probs = sim.generate(n_samples=100)

    # Marginal bounds with different confidence levels
    marginal_90 = compute_marginal_operational_bounds(
        labels, probs, 0.1, 0.05, delta=0.05, confidence_level=0.90, n_folds=3, random_seed=42
    )
    marginal_99 = compute_marginal_operational_bounds(
        labels, probs, 0.1, 0.05, delta=0.05, confidence_level=0.99, n_folds=3, random_seed=42
    )

    # Higher confidence should give wider intervals
    sing_90 = marginal_90.rate_bounds["singleton"]
    sing_99 = marginal_99.rate_bounds["singleton"]

    width_90 = sing_90.upper_bound - sing_90.lower_bound
    width_99 = sing_99.upper_bound - sing_99.lower_bound

    assert width_99 > width_90, "99% CI should be wider than 90% CI"


def test_rates_not_split_independently_confident():
    """Test that each rate gets the same confidence independently (NOT split)."""
    np.random.seed(42)
    sim = BinaryClassifierSimulator(p_class1=0.5, beta_params_class0=(3, 7), beta_params_class1=(7, 3), seed=42)
    labels, probs = sim.generate(n_samples=100)
    class_data = split_by_class(labels, probs)
    cal_result, _ = mondrian_conformal_calibrate(class_data, 0.1, 0.05)

    # Request 3 rates with delta=0.06 (94% confidence)
    op_bounds = compute_mondrian_operational_bounds(
        cal_result,
        labels,
        probs,
        delta=0.06,  # Per-class delta
        rate_types=["singleton", "doublet", "abstention"],
        n_folds=3,
        random_seed=42,
    )

    # Each class gets delta_per_class = 0.06/2 = 0.03 (97% PAC confidence)
    # All rates within each class should use DEFAULT CI width (95%)
    for class_label in [0, 1]:
        # Check that PAC confidence is correct
        assert np.isclose(op_bounds[class_label].rate_confidence, 1 - 0.06 / 2)

        # Check that CI width is 95% (default)
        for rate_name in ["singleton", "doublet", "abstention"]:
            if rate_name in op_bounds[class_label].rate_bounds:
                ci_width = op_bounds[class_label].rate_bounds[rate_name].ci_width
                assert np.isclose(ci_width, 0.95), f"{rate_name} should use 95% CI width (default)"
