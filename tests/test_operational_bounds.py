"""Tests for operational_bounds_simple module."""

import numpy as np
import pytest

from ssbc.core import ssbc_correct
from ssbc.operational_bounds_simple import (
    compute_pac_operational_bounds_marginal,
    compute_pac_operational_bounds_perclass,
)
from ssbc.simulation import BinaryClassifierSimulator


@pytest.fixture
def simple_data():
    """Generate simple test data."""
    sim = BinaryClassifierSimulator(p_class1=0.3, beta_params_class0=(2, 5), beta_params_class1=(6, 2), seed=42)
    labels, probs = sim.generate(100)
    return labels, probs


@pytest.fixture
def ssbc_results(simple_data):
    """Get SSBC results for test data."""
    labels, _ = simple_data
    n_0 = np.sum(labels == 0)
    n_1 = np.sum(labels == 1)

    ssbc_0 = ssbc_correct(alpha_target=0.10, n=n_0, delta=0.10)
    ssbc_1 = ssbc_correct(alpha_target=0.10, n=n_1, delta=0.10)

    return ssbc_0, ssbc_1


class TestPACOperationalBoundsPerClass:
    """Test per-class PAC operational bounds."""

    def test_basic_computation(self, simple_data, ssbc_results):
        """Test basic per-class bounds computation."""
        labels, probs = simple_data
        ssbc_0, ssbc_1 = ssbc_results

        result = compute_pac_operational_bounds_perclass(
            ssbc_result_0=ssbc_0,
            ssbc_result_1=ssbc_1,
            labels=labels,
            probs=probs,
            class_label=0,
            test_size=100,
            ci_level=0.95,
            pac_level=0.90,
            use_union_bound=True,
        )

        # Check structure
        assert "singleton_rate_bounds" in result
        assert "doublet_rate_bounds" in result
        assert "abstention_rate_bounds" in result
        assert "singleton_error_rate_bounds" in result
        assert "expected_singleton_rate" in result
        assert "expected_doublet_rate" in result
        assert "expected_abstention_rate" in result
        assert "expected_singleton_error_rate" in result

        # Check bounds are tuples
        assert isinstance(result["singleton_rate_bounds"], tuple)
        assert len(result["singleton_rate_bounds"]) == 2

        # Check bounds are valid (lower <= upper)
        lower, upper = result["singleton_rate_bounds"]
        assert lower <= upper
        assert 0 <= lower <= 1
        assert 0 <= upper <= 1

    def test_both_classes(self, simple_data, ssbc_results):
        """Test bounds for both classes."""
        labels, probs = simple_data
        ssbc_0, ssbc_1 = ssbc_results

        result_0 = compute_pac_operational_bounds_perclass(
            ssbc_result_0=ssbc_0,
            ssbc_result_1=ssbc_1,
            labels=labels,
            probs=probs,
            class_label=0,
            test_size=100,
            ci_level=0.95,
            pac_level=0.90,
        )

        result_1 = compute_pac_operational_bounds_perclass(
            ssbc_result_0=ssbc_0,
            ssbc_result_1=ssbc_1,
            labels=labels,
            probs=probs,
            class_label=1,
            test_size=100,
            ci_level=0.95,
            pac_level=0.90,
        )

        # Both should have valid results
        assert result_0["expected_singleton_rate"] >= 0
        assert result_1["expected_singleton_rate"] >= 0

    def test_union_bound_effect(self, simple_data, ssbc_results):
        """Test that union bound widens intervals."""
        labels, probs = simple_data
        ssbc_0, ssbc_1 = ssbc_results

        # Without union bound
        result_no_union = compute_pac_operational_bounds_perclass(
            ssbc_result_0=ssbc_0,
            ssbc_result_1=ssbc_1,
            labels=labels,
            probs=probs,
            class_label=0,
            test_size=100,
            ci_level=0.95,
            pac_level=0.90,
            use_union_bound=False,
        )

        # With union bound
        result_with_union = compute_pac_operational_bounds_perclass(
            ssbc_result_0=ssbc_0,
            ssbc_result_1=ssbc_1,
            labels=labels,
            probs=probs,
            class_label=0,
            test_size=100,
            ci_level=0.95,
            pac_level=0.90,
            use_union_bound=True,
        )

        # Union bound should give wider (or equal) intervals
        width_no_union = result_no_union["singleton_rate_bounds"][1] - result_no_union["singleton_rate_bounds"][0]
        width_with_union = result_with_union["singleton_rate_bounds"][1] - result_with_union["singleton_rate_bounds"][0]

        assert width_with_union >= width_no_union - 1e-10  # Allow small numerical errors

    def test_multiprocessing(self, simple_data, ssbc_results):
        """Test that multiprocessing gives same results as serial."""
        labels, probs = simple_data
        ssbc_0, ssbc_1 = ssbc_results

        # Serial
        result_serial = compute_pac_operational_bounds_perclass(
            ssbc_result_0=ssbc_0,
            ssbc_result_1=ssbc_1,
            labels=labels,
            probs=probs,
            class_label=0,
            test_size=100,
            n_jobs=1,
        )

        # Parallel
        result_parallel = compute_pac_operational_bounds_perclass(
            ssbc_result_0=ssbc_0,
            ssbc_result_1=ssbc_1,
            labels=labels,
            probs=probs,
            class_label=0,
            test_size=100,
            n_jobs=2,
        )

        # Results should be identical
        np.testing.assert_allclose(
            result_serial["singleton_rate_bounds"], result_parallel["singleton_rate_bounds"], rtol=1e-10
        )


class TestPACOperationalBoundsMarginal:
    """Test marginal PAC operational bounds."""

    def test_basic_computation(self, simple_data, ssbc_results):
        """Test basic marginal bounds computation."""
        labels, probs = simple_data
        ssbc_0, ssbc_1 = ssbc_results

        result = compute_pac_operational_bounds_marginal(
            ssbc_result_0=ssbc_0,
            ssbc_result_1=ssbc_1,
            labels=labels,
            probs=probs,
            test_size=100,
            ci_level=0.95,
            pac_level=0.90,
            use_union_bound=True,
        )

        # Check structure
        assert "singleton_rate_bounds" in result
        assert "doublet_rate_bounds" in result
        assert "abstention_rate_bounds" in result
        assert "singleton_error_rate_bounds" in result
        assert "n_grid_points" in result

        # Check bounds validity
        for key in ["singleton_rate_bounds", "doublet_rate_bounds", "abstention_rate_bounds"]:
            lower, upper = result[key]
            assert lower <= upper
            assert 0 <= lower <= 1
            assert 0 <= upper <= 1

    def test_rates_sum_to_one(self, simple_data, ssbc_results):
        """Test that rate bounds are consistent (sum ~1)."""
        labels, probs = simple_data
        ssbc_0, ssbc_1 = ssbc_results

        result = compute_pac_operational_bounds_marginal(
            ssbc_result_0=ssbc_0,
            ssbc_result_1=ssbc_1,
            labels=labels,
            probs=probs,
            test_size=100,
        )

        # Expected rates should sum to ~1
        expected_sum = (
            result["expected_singleton_rate"] + result["expected_doublet_rate"] + result["expected_abstention_rate"]
        )

        np.testing.assert_allclose(expected_sum, 1.0, rtol=1e-2)

    def test_ci_level_effect(self, simple_data, ssbc_results):
        """Test that higher CI level gives wider bounds."""
        labels, probs = simple_data
        ssbc_0, ssbc_1 = ssbc_results

        # 90% CI
        result_90 = compute_pac_operational_bounds_marginal(
            ssbc_result_0=ssbc_0,
            ssbc_result_1=ssbc_1,
            labels=labels,
            probs=probs,
            test_size=100,
            ci_level=0.90,
            pac_level=0.90,
        )

        # 95% CI
        result_95 = compute_pac_operational_bounds_marginal(
            ssbc_result_0=ssbc_0,
            ssbc_result_1=ssbc_1,
            labels=labels,
            probs=probs,
            test_size=100,
            ci_level=0.95,
            pac_level=0.90,
        )

        # 95% CI should be wider than 90% CI
        width_90 = result_90["singleton_rate_bounds"][1] - result_90["singleton_rate_bounds"][0]
        width_95 = result_95["singleton_rate_bounds"][1] - result_95["singleton_rate_bounds"][0]

        assert width_95 >= width_90 - 1e-10


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_small_sample_size(self):
        """Test with very small sample size."""
        sim = BinaryClassifierSimulator(p_class1=0.5, seed=42)
        labels, probs = sim.generate(20)  # Small sample

        n_0 = np.sum(labels == 0)
        n_1 = np.sum(labels == 1)

        ssbc_0 = ssbc_correct(alpha_target=0.10, n=n_0, delta=0.10)
        ssbc_1 = ssbc_correct(alpha_target=0.10, n=n_1, delta=0.10)

        # Should not crash
        result = compute_pac_operational_bounds_marginal(
            ssbc_result_0=ssbc_0,
            ssbc_result_1=ssbc_1,
            labels=labels,
            probs=probs,
            test_size=20,
        )

        # Should have valid bounds
        assert result["singleton_rate_bounds"][0] <= result["singleton_rate_bounds"][1]

    def test_extreme_alpha(self):
        """Test with extreme alpha values."""
        sim = BinaryClassifierSimulator(p_class1=0.5, seed=42)
        labels, probs = sim.generate(50)

        n_0 = np.sum(labels == 0)
        n_1 = np.sum(labels == 1)

        # Very small alpha
        ssbc_0 = ssbc_correct(alpha_target=0.01, n=n_0, delta=0.10)
        ssbc_1 = ssbc_correct(alpha_target=0.01, n=n_1, delta=0.10)

        result = compute_pac_operational_bounds_marginal(
            ssbc_result_0=ssbc_0,
            ssbc_result_1=ssbc_1,
            labels=labels,
            probs=probs,
            test_size=50,
        )

        # Should have valid bounds
        assert 0 <= result["singleton_rate_bounds"][0] <= result["singleton_rate_bounds"][1] <= 1
