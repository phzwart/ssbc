"""Edge case tests for SSBC modules.

This module contains tests for edge cases and boundary conditions
that might not be covered in the main test suites.
"""

import numpy as np
import pytest

from ssbc.bounds import (
    clopper_pearson_intervals,
    clopper_pearson_lower,
    clopper_pearson_upper,
    cp_interval,
    prediction_bounds,
)
from ssbc.calibration import mondrian_conformal_calibrate, split_by_class
from ssbc.core_pkg import ssbc_correct


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_labels(self):
        """Test with empty label array (returns zero counts)."""
        labels = np.array([], dtype=int)
        intervals = clopper_pearson_intervals(labels)
        assert intervals[0]["count"] == 0
        assert intervals[1]["count"] == 0

    def test_single_class_labels(self):
        """Test with labels containing only one class."""
        labels = np.array([0, 0, 0, 0, 0])
        intervals = clopper_pearson_intervals(labels)

        assert intervals[0]["count"] == 5
        assert intervals[0]["proportion"] == 1.0
        assert intervals[1]["count"] == 0
        assert intervals[1]["proportion"] == 0.0

    def test_extreme_confidence_levels(self):
        """Test with extreme confidence levels."""
        labels = np.array([0, 0, 1, 1, 1])

        # Very low confidence
        intervals_low = clopper_pearson_intervals(labels, confidence=0.01)
        assert intervals_low[0]["lower"] < intervals_low[0]["upper"]

        # Very high confidence
        intervals_high = clopper_pearson_intervals(labels, confidence=0.999)
        assert intervals_high[0]["lower"] < intervals_high[0]["upper"]

    def test_zero_successes_clopper_pearson(self):
        """Test Clopper-Pearson with zero successes."""
        lower = clopper_pearson_lower(k=0, n=10, confidence=0.95)
        upper = clopper_pearson_upper(k=0, n=10, confidence=0.95)

        assert lower == 0.0
        assert upper > 0.0
        assert upper < 1.0

    def test_all_successes_clopper_pearson(self):
        """Test Clopper-Pearson with all successes."""
        lower = clopper_pearson_lower(k=10, n=10, confidence=0.95)
        upper = clopper_pearson_upper(k=10, n=10, confidence=0.95)

        assert lower > 0.0
        assert upper == 1.0

    def test_ssbc_extreme_parameters(self):
        """Test SSBC with extreme parameters."""
        # Very small alpha
        result = ssbc_correct(alpha_target=0.001, n=50, delta=0.10, mode="beta")
        assert result.alpha_corrected <= result.alpha_target

        # Very large alpha
        result = ssbc_correct(alpha_target=0.5, n=50, delta=0.10, mode="beta")
        assert result.alpha_corrected <= result.alpha_target

        # Very small delta
        result = ssbc_correct(alpha_target=0.10, n=50, delta=0.001, mode="beta")
        assert result.alpha_corrected <= result.alpha_target

        # Very large delta
        result = ssbc_correct(alpha_target=0.10, n=50, delta=0.5, mode="beta")
        assert result.alpha_corrected <= result.alpha_target

    def test_ssbc_minimum_n(self):
        """Test SSBC with minimum possible n."""
        result = ssbc_correct(alpha_target=0.10, n=1, delta=0.10, mode="beta")
        assert result.n == 1
        assert result.u_star >= 0
        assert result.u_star <= 1

    def test_prediction_bounds_edge_cases(self):
        """Test prediction bounds with edge cases."""
        # Zero calibration successes
        lower, upper = prediction_bounds(k_cal=0, n_cal=10, n_test=5, confidence=0.95)
        assert lower >= 0.0
        assert upper <= 1.0

        # All calibration successes
        lower, upper = prediction_bounds(k_cal=10, n_cal=10, n_test=5, confidence=0.95)
        assert lower >= 0.0
        assert upper <= 1.0

        # Single test sample
        lower, upper = prediction_bounds(k_cal=5, n_cal=10, n_test=1, confidence=0.95)
        assert lower >= 0.0
        assert upper <= 1.0

    def test_cp_interval_edge_cases(self):
        """Test cp_interval with edge cases."""
        # Zero count
        result = cp_interval(count=0, total=10, confidence=0.95)
        assert result["count"] == 0
        assert result["proportion"] == 0.0
        assert result["lower"] == 0.0

        # Count equals total
        result = cp_interval(count=10, total=10, confidence=0.95)
        assert result["count"] == 10
        assert result["proportion"] == 1.0
        assert result["upper"] == 1.0

    def test_conformal_empty_data(self):
        """Test conformal prediction with empty data returns empty splits."""
        labels = np.array([], dtype=int)
        probs = np.array([])
        splits = split_by_class(labels, probs)
        assert splits[0]["n"] == 0
        assert splits[1]["n"] == 0

    def test_conformal_single_class(self):
        """Test conformal prediction with single class."""
        labels = np.array([0, 0, 0, 0, 0])
        probs = np.random.random(5)

        split_data = split_by_class(labels, probs)
        assert 0 in split_data
        # implementation returns both classes; class 1 may be empty
        assert 1 in split_data
        assert split_data[1]["n"] == 0
        assert len(split_data[0]["labels"]) == 5

    def test_conformal_extreme_probabilities(self):
        """Test conformal prediction with extreme probabilities."""
        labels = np.array([0, 0, 1, 1])
        probs = np.array(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
            ]
        )  # Extreme probabilities per class

        split_data = split_by_class(labels, probs)
        assert 0 in split_data
        assert 1 in split_data

        # Test calibration with extreme probabilities
        cal_result, pred_stats = mondrian_conformal_calibrate(split_data, alpha_target=0.1, delta=0.1)
        assert 0 in cal_result
        assert 1 in cal_result

    def test_numerical_stability(self):
        """Test numerical stability with very small/large numbers."""
        # Very small probabilities
        labels = np.array([0, 0, 1, 1])
        probs = np.array(
            [
                [1.0 - 1e-10, 1e-10],
                [1.0 - 1e-10, 1e-10],
                [1e-10, 1.0 - 1e-10],
                [1e-10, 1.0 - 1e-10],
            ]
        )

        split_data = split_by_class(labels, probs)
        cal_result, pred_stats = mondrian_conformal_calibrate(split_data, alpha_target=0.1, delta=0.1)

        # Should not crash and should produce reasonable results
        assert 0 in cal_result
        assert 1 in cal_result

    def test_memory_efficiency_large_arrays(self):
        """Test memory efficiency with large arrays."""
        # Test with moderately large arrays (not too large for CI)
        n_samples = 1000
        labels = np.random.choice([0, 1], size=n_samples)
        probs = np.random.random(n_samples)

        # Should not cause memory issues
        split_data = split_by_class(labels, probs)
        assert len(split_data[0]["labels"]) + len(split_data[1]["labels"]) == n_samples

    def test_boundary_values(self):
        """Test boundary values for various functions."""
        # Test alpha values at boundaries
        for alpha in [0.0, 0.5, 1.0]:
            if alpha == 0.0:
                # Alpha = 0 should be handled gracefully
                continue
            if alpha == 1.0:
                with pytest.raises(ValueError):
                    ssbc_correct(alpha_target=alpha, n=50, delta=0.10, mode="beta")
            else:
                result = ssbc_correct(alpha_target=alpha, n=50, delta=0.10, mode="beta")
                assert result.alpha_corrected <= result.alpha_target

    def test_invalid_inputs(self):
        """Test handling of invalid inputs."""
        # Negative alpha
        with pytest.raises(ValueError):
            ssbc_correct(alpha_target=-0.1, n=50, delta=0.10, mode="beta")

        # Alpha > 1
        with pytest.raises(ValueError):
            ssbc_correct(alpha_target=1.1, n=50, delta=0.10, mode="beta")

        # Negative n
        with pytest.raises(ValueError):
            ssbc_correct(alpha_target=0.1, n=-1, delta=0.10, mode="beta")

        # Negative delta
        with pytest.raises(ValueError):
            ssbc_correct(alpha_target=0.1, n=50, delta=-0.1, mode="beta")

        # Delta > 1
        with pytest.raises(ValueError):
            ssbc_correct(alpha_target=0.1, n=50, delta=1.1, mode="beta")

    def test_mode_validation(self):
        """Test validation of mode parameter."""
        # Invalid mode
        with pytest.raises(ValueError):
            ssbc_correct(alpha_target=0.1, n=50, delta=0.10, mode="invalid")

        # Valid modes should work
        for mode in ["beta", "beta-binomial"]:
            result = ssbc_correct(alpha_target=0.1, n=50, delta=0.10, mode=mode)
            assert result.mode == mode
