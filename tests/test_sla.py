"""Tests for SLA (Service Level Agreement) operational bounds module."""

import numpy as np
import pytest

from ssbc import (
    BinaryClassifierSimulator,
    OperationalRateBounds,
    OperationalRateBoundsResult,
    compute_marginal_operational_bounds,
    compute_mondrian_operational_bounds,
    mondrian_conformal_calibrate,
    split_by_class,
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
# Test Mondrian Operational Bounds (Full Conformal LOO)
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
        rate_types=["singleton", "doublet"],
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
    results = compute_mondrian_operational_bounds(cal_result, labels, probs)

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
    results = compute_mondrian_operational_bounds(cal_result, labels, probs)

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
    operational_bounds = compute_mondrian_operational_bounds(cal_result, labels, probs)

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
    operational_bounds = compute_mondrian_operational_bounds(cal_result, labels, probs)

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
        ci_width=0.95,
        n_evaluations=100,
        n_successes=75,
    )

    assert bounds.rate_name == "singleton"
    assert bounds.lower_bound == 0.6
    assert bounds.upper_bound == 0.85
    assert bounds.ci_width == 0.95
    assert bounds.n_evaluations == 100
    assert bounds.n_successes == 75


def test_operational_bounds_result_dataclass():
    """Test OperationalRateBoundsResult dataclass."""
    singleton_bounds = OperationalRateBounds(
        rate_name="singleton",
        lower_bound=0.7,
        upper_bound=0.9,
        ci_width=0.95,
        n_evaluations=100,
        n_successes=80,
    )

    result = OperationalRateBoundsResult(
        rate_bounds={"singleton": singleton_bounds},
        ci_width=0.95,
        thresholds=0.5,
        n_calibration=100,
    )

    assert result.ci_width == 0.95
    assert result.thresholds == 0.5
    assert result.n_calibration == 100
    assert "singleton" in result.rate_bounds


def test_custom_ci_width():
    """Test that custom ci_width parameter works."""
    np.random.seed(42)
    sim = BinaryClassifierSimulator(p_class1=0.5, beta_params_class0=(3, 7), beta_params_class1=(7, 3), seed=42)
    labels, probs = sim.generate(n_samples=100)

    # Marginal bounds with different CI widths
    marginal_90 = compute_marginal_operational_bounds(labels, probs, 0.1, 0.05, ci_width=0.90)
    marginal_99 = compute_marginal_operational_bounds(labels, probs, 0.1, 0.05, ci_width=0.99)

    # Higher CI width should give wider intervals
    sing_90 = marginal_90.rate_bounds["singleton"]
    sing_99 = marginal_99.rate_bounds["singleton"]

    width_90 = sing_90.upper_bound - sing_90.lower_bound
    width_99 = sing_99.upper_bound - sing_99.lower_bound

    assert width_99 > width_90, "99% CI width should give wider intervals than 90%"

    # Both should store their CI widths correctly
    assert np.isclose(sing_90.ci_width, 0.90)
    assert np.isclose(sing_99.ci_width, 0.99)


def test_rates_not_split_independently_confident():
    """Test that each rate gets the same confidence independently (NOT split)."""
    np.random.seed(42)
    sim = BinaryClassifierSimulator(p_class1=0.5, beta_params_class0=(3, 7), beta_params_class1=(7, 3), seed=42)
    labels, probs = sim.generate(n_samples=100)
    class_data = split_by_class(labels, probs)
    cal_result, _ = mondrian_conformal_calibrate(class_data, 0.1, 0.05)

    # Request 3 rates with custom CI width
    op_bounds = compute_mondrian_operational_bounds(
        cal_result,
        labels,
        probs,
        rate_types=["singleton", "doublet", "abstention"],
        ci_width=0.90,
    )

    # All rates should have same CI width
    for class_label in [0, 1]:
        # Check that CI width is correct
        assert np.isclose(op_bounds[class_label].ci_width, 0.90)

        # All rates should have same CI width
        all_ci_widths = [
            op_bounds[class_label].rate_bounds[rt].ci_width for rt in ["singleton", "doublet", "abstention"]
        ]
        assert len(set(all_ci_widths)) == 1  # All same
        assert np.isclose(all_ci_widths[0], 0.90)
