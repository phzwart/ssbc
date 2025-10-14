"""Tests for utils module."""

import numpy as np

from ssbc.utils import compute_operational_rate


class TestComputeOperationalRate:
    """Test compute_operational_rate function."""

    def test_basic_computation(self):
        """Test basic operational rate computation."""
        # Simple case: 3 singletons, 2 doublets, 1 abstention out of 6 total
        prediction_set_sizes = np.array([1, 1, 1, 2, 2, 0])  # 3 singles, 2 doubles, 1 abstain

        rates = compute_operational_rate(prediction_set_sizes)

        # Check structure
        assert "singleton_rate" in rates
        assert "doublet_rate" in rates
        assert "abstention_rate" in rates

        # Check values
        assert rates["singleton_rate"] == 3 / 6
        assert rates["doublet_rate"] == 2 / 6
        assert rates["abstention_rate"] == 1 / 6

    def test_rates_sum_to_one(self):
        """Test that rates sum to 1."""
        prediction_set_sizes = np.array([0, 1, 1, 2, 2, 1, 0, 2])

        rates = compute_operational_rate(prediction_set_sizes)

        total = rates["singleton_rate"] + rates["doublet_rate"] + rates["abstention_rate"]

        np.testing.assert_allclose(total, 1.0, rtol=1e-10)

    def test_all_singletons(self):
        """Test case with only singletons."""
        prediction_set_sizes = np.array([1, 1, 1, 1, 1])

        rates = compute_operational_rate(prediction_set_sizes)

        assert rates["singleton_rate"] == 1.0
        assert rates["doublet_rate"] == 0.0
        assert rates["abstention_rate"] == 0.0

    def test_all_doublets(self):
        """Test case with only doublets."""
        prediction_set_sizes = np.array([2, 2, 2, 2])

        rates = compute_operational_rate(prediction_set_sizes)

        assert rates["singleton_rate"] == 0.0
        assert rates["doublet_rate"] == 1.0
        assert rates["abstention_rate"] == 0.0

    def test_all_abstentions(self):
        """Test case with only abstentions."""
        prediction_set_sizes = np.array([0, 0, 0, 0, 0])

        rates = compute_operational_rate(prediction_set_sizes)

        assert rates["singleton_rate"] == 0.0
        assert rates["doublet_rate"] == 0.0
        assert rates["abstention_rate"] == 1.0

    def test_empty_input(self):
        """Test with empty input."""
        prediction_set_sizes = np.array([])

        rates = compute_operational_rate(prediction_set_sizes)

        # Should return NaN or handle gracefully
        # Check what the actual behavior is
        assert "singleton_rate" in rates

    def test_single_sample(self):
        """Test with single sample."""
        # Single singleton
        rates_single = compute_operational_rate(np.array([1]))
        assert rates_single["singleton_rate"] == 1.0

        # Single doublet
        rates_double = compute_operational_rate(np.array([2]))
        assert rates_double["doublet_rate"] == 1.0

        # Single abstention
        rates_abstain = compute_operational_rate(np.array([0]))
        assert rates_abstain["abstention_rate"] == 1.0

    def test_large_array(self):
        """Test with large array."""
        # Create large array
        prediction_set_sizes = np.random.choice([0, 1, 2], size=10000)

        rates = compute_operational_rate(prediction_set_sizes)

        # Rates should sum to 1
        total = rates["singleton_rate"] + rates["doublet_rate"] + rates["abstention_rate"]
        np.testing.assert_allclose(total, 1.0, rtol=1e-10)

        # All rates should be between 0 and 1
        assert 0 <= rates["singleton_rate"] <= 1
        assert 0 <= rates["doublet_rate"] <= 1
        assert 0 <= rates["abstention_rate"] <= 1

    def test_input_validation(self):
        """Test input validation."""
        # Test with list (should work if function accepts it)
        prediction_set_sizes = [0, 1, 1, 2]
        rates = compute_operational_rate(np.array(prediction_set_sizes))

        assert "singleton_rate" in rates

    def test_binary_classification_values(self):
        """Test that values are only 0, 1, or 2 for binary classification."""
        # Valid inputs
        prediction_set_sizes = np.array([0, 1, 2, 0, 1, 2, 1, 1])

        rates = compute_operational_rate(prediction_set_sizes)

        # Should compute correctly
        n_abstain = np.sum(prediction_set_sizes == 0)
        n_single = np.sum(prediction_set_sizes == 1)
        n_double = np.sum(prediction_set_sizes == 2)
        n_total = len(prediction_set_sizes)

        assert rates["abstention_rate"] == n_abstain / n_total
        assert rates["singleton_rate"] == n_single / n_total
        assert rates["doublet_rate"] == n_double / n_total

    def test_integer_vs_float_input(self):
        """Test with integer vs float input."""
        sizes_int = np.array([0, 1, 1, 2], dtype=int)
        sizes_float = np.array([0.0, 1.0, 1.0, 2.0], dtype=float)

        rates_int = compute_operational_rate(sizes_int)
        rates_float = compute_operational_rate(sizes_float)

        # Results should be identical
        np.testing.assert_allclose(rates_int["singleton_rate"], rates_float["singleton_rate"])
        np.testing.assert_allclose(rates_int["doublet_rate"], rates_float["doublet_rate"])
        np.testing.assert_allclose(rates_int["abstention_rate"], rates_float["abstention_rate"])

    def test_return_type(self):
        """Test return type is dict."""
        prediction_set_sizes = np.array([0, 1, 2, 1])

        rates = compute_operational_rate(prediction_set_sizes)

        assert isinstance(rates, dict)
        assert all(isinstance(v, float | np.floating) for v in rates.values())

    def test_numerical_precision(self):
        """Test numerical precision."""
        # Create case where floating point errors might occur
        prediction_set_sizes = np.array([1] * 3 + [2] * 3 + [0] * 3)  # Exactly 1/3 each

        rates = compute_operational_rate(prediction_set_sizes)

        # Each rate should be exactly 1/3
        np.testing.assert_allclose(rates["singleton_rate"], 1 / 3, rtol=1e-10)
        np.testing.assert_allclose(rates["doublet_rate"], 1 / 3, rtol=1e-10)
        np.testing.assert_allclose(rates["abstention_rate"], 1 / 3, rtol=1e-10)
