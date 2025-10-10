"""Tests for the statistics module."""

import numpy as np

from ssbc.statistics import clopper_pearson_intervals, cp_interval, ensure_ci


class TestClopperPearsonIntervals:
    """Test clopper_pearson_intervals function."""

    def test_balanced_labels(self):
        """Test with balanced binary labels."""
        labels = np.array([0, 0, 1, 1, 0, 1])
        intervals = clopper_pearson_intervals(labels, confidence=0.95)

        assert 0 in intervals
        assert 1 in intervals

        # Class 0: 3/6 = 0.5
        assert intervals[0]["count"] == 3
        assert intervals[0]["proportion"] == 0.5
        assert 0 < intervals[0]["lower"] < intervals[0]["proportion"]
        assert intervals[0]["proportion"] < intervals[0]["upper"] < 1

        # Class 1: 3/6 = 0.5
        assert intervals[1]["count"] == 3
        assert intervals[1]["proportion"] == 0.5

    def test_imbalanced_labels(self):
        """Test with imbalanced labels."""
        labels = np.array([0, 0, 0, 0, 1])
        intervals = clopper_pearson_intervals(labels)

        # Class 0: 4/5 = 0.8
        assert intervals[0]["count"] == 4
        assert intervals[0]["proportion"] == 0.8

        # Class 1: 1/5 = 0.2
        assert intervals[1]["count"] == 1
        assert intervals[1]["proportion"] == 0.2

    def test_all_zeros(self):
        """Test with all class 0 labels."""
        labels = np.zeros(10, dtype=int)
        intervals = clopper_pearson_intervals(labels)

        # Class 0: 10/10 = 1.0
        assert intervals[0]["count"] == 10
        assert intervals[0]["proportion"] == 1.0
        assert intervals[0]["upper"] == 1.0
        assert intervals[0]["lower"] < 1.0

        # Class 1: 0/10 = 0.0
        assert intervals[1]["count"] == 0
        assert intervals[1]["proportion"] == 0.0
        assert intervals[1]["lower"] == 0.0
        assert intervals[1]["upper"] > 0.0

    def test_all_ones(self):
        """Test with all class 1 labels."""
        labels = np.ones(10, dtype=int)
        intervals = clopper_pearson_intervals(labels)

        # Class 0: 0/10 = 0.0
        assert intervals[0]["count"] == 0
        assert intervals[0]["proportion"] == 0.0

        # Class 1: 10/10 = 1.0
        assert intervals[1]["count"] == 10
        assert intervals[1]["proportion"] == 1.0

    def test_different_confidence_levels(self):
        """Test with different confidence levels."""
        labels = np.array([0, 0, 1, 1, 1])

        interval_90 = clopper_pearson_intervals(labels, confidence=0.90)
        interval_95 = clopper_pearson_intervals(labels, confidence=0.95)
        interval_99 = clopper_pearson_intervals(labels, confidence=0.99)

        # Higher confidence = wider intervals
        width_90 = interval_90[0]["upper"] - interval_90[0]["lower"]
        width_95 = interval_95[0]["upper"] - interval_95[0]["lower"]
        width_99 = interval_99[0]["upper"] - interval_99[0]["lower"]

        assert width_90 < width_95 < width_99

    def test_single_sample(self):
        """Test with single sample."""
        labels = np.array([0])
        intervals = clopper_pearson_intervals(labels)

        assert intervals[0]["count"] == 1
        assert intervals[1]["count"] == 0

    def test_coverage_property(self):
        """Test that intervals have proper coverage."""
        # With many samples, interval should be narrow
        labels = np.random.choice([0, 1], size=1000, p=[0.7, 0.3])
        intervals = clopper_pearson_intervals(labels, confidence=0.95)

        # Class 0 should be around 0.7
        assert intervals[0]["proportion"] > 0.6
        assert intervals[0]["proportion"] < 0.8

        # Width should be relatively narrow with n=1000
        width = intervals[0]["upper"] - intervals[0]["lower"]
        assert width < 0.1


class TestCpInterval:
    """Test cp_interval function."""

    def test_basic_interval(self):
        """Test basic interval computation."""
        result = cp_interval(count=7, total=10, confidence=0.95)

        assert result["count"] == 7
        assert result["proportion"] == 0.7
        assert 0 < result["lower"] < 0.7
        assert 0.7 < result["upper"] < 1.0

    def test_zero_count(self):
        """Test with zero successes."""
        result = cp_interval(count=0, total=10)

        assert result["count"] == 0
        assert result["proportion"] == 0.0
        assert result["lower"] == 0.0
        assert result["upper"] > 0.0

    def test_all_successes(self):
        """Test with all successes."""
        result = cp_interval(count=10, total=10)

        assert result["count"] == 10
        assert result["proportion"] == 1.0
        assert result["lower"] < 1.0
        assert result["upper"] == 1.0

    def test_zero_total(self):
        """Test with zero total (edge case)."""
        result = cp_interval(count=0, total=0)

        assert result["count"] == 0
        assert result["proportion"] == 0.0
        assert result["lower"] == 0.0
        assert result["upper"] == 0.0

    def test_different_confidence_levels(self):
        """Test different confidence levels."""
        result_90 = cp_interval(5, 10, confidence=0.90)
        result_95 = cp_interval(5, 10, confidence=0.95)
        result_99 = cp_interval(5, 10, confidence=0.99)

        width_90 = result_90["upper"] - result_90["lower"]
        width_95 = result_95["upper"] - result_95["lower"]
        width_99 = result_99["upper"] - result_99["lower"]

        assert width_90 < width_95 < width_99

    def test_float_conversion(self):
        """Test that count and total are properly converted."""
        result = cp_interval(count=5.0, total=10.0)

        assert isinstance(result["count"], int)
        assert isinstance(result["proportion"], float)
        assert isinstance(result["lower"], float)
        assert isinstance(result["upper"], float)

    def test_consistency_with_clopper_pearson_intervals(self):
        """Test consistency with clopper_pearson_intervals."""
        labels = np.array([0, 0, 0, 1, 1])
        intervals = clopper_pearson_intervals(labels)

        # Manually compute for class 1
        manual = cp_interval(count=2, total=5)

        assert intervals[1]["proportion"] == manual["proportion"]
        assert abs(intervals[1]["lower"] - manual["lower"]) < 1e-10
        assert abs(intervals[1]["upper"] - manual["upper"]) < 1e-10


class TestEnsureCi:
    """Test ensure_ci function."""

    def test_with_existing_rate_and_ci(self):
        """Test when dict already has rate and CI."""
        d = {"rate": 0.5, "lower": 0.3, "upper": 0.7}

        rate, lower, upper = ensure_ci(d, count=5, total=10)

        assert rate == 0.5
        assert lower == 0.3
        assert upper == 0.7

    def test_with_proportion_instead_of_rate(self):
        """Test when dict has 'proportion' instead of 'rate'."""
        d = {"proportion": 0.6, "lower": 0.4, "upper": 0.8}

        rate, lower, upper = ensure_ci(d, count=6, total=10)

        assert rate == 0.6
        assert lower == 0.4
        assert upper == 0.8

    def test_with_ci_95_tuple(self):
        """Test when CI is stored as 'ci_95' tuple."""
        d = {"rate": 0.5, "ci_95": (0.3, 0.7)}

        rate, lower, upper = ensure_ci(d, count=5, total=10)

        assert rate == 0.5
        assert lower == 0.3
        assert upper == 0.7

    def test_missing_rate_computes_cp(self):
        """Test that missing rate triggers CP computation."""
        d = {}

        rate, lower, upper = ensure_ci(d, count=7, total=10)

        assert rate == 0.7
        assert 0 < lower < 0.7
        assert 0.7 < upper < 1.0

    def test_missing_ci_computes_cp(self):
        """Test that missing CI triggers CP computation."""
        d = {"rate": 0.5}

        rate, lower, upper = ensure_ci(d, count=5, total=10)

        # Should compute new CI
        assert rate > 0
        assert lower >= 0
        assert upper <= 1

    def test_with_non_dict_input(self):
        """Test with non-dict input."""
        rate, lower, upper = ensure_ci("not a dict", count=5, total=10)

        # Should compute CP interval
        assert rate == 0.5
        assert 0 < lower < 0.5
        assert 0.5 < upper < 1.0

    def test_custom_confidence(self):
        """Test with custom confidence level."""
        d = {}

        rate_95, lower_95, upper_95 = ensure_ci(d, count=5, total=10, confidence=0.95)
        rate_99, lower_99, upper_99 = ensure_ci(d, count=5, total=10, confidence=0.99)

        # Same rate, different intervals
        assert rate_95 == rate_99 == 0.5

        # 99% CI should be wider
        width_95 = upper_95 - lower_95
        width_99 = upper_99 - lower_99
        assert width_99 > width_95

    def test_zero_values_edge_case(self):
        """Test edge case with zeros."""
        d = {"rate": 0.0, "lower": 0.0, "upper": 0.0}

        # Should recompute because CI is all zeros
        rate, lower, upper = ensure_ci(d, count=5, total=10)

        assert rate > 0
        assert upper > 0

    def test_returns_floats(self):
        """Test that all return values are floats."""
        d = {"rate": 0.5, "lower": 0.3, "upper": 0.7}

        rate, lower, upper = ensure_ci(d, count=5, total=10)

        assert isinstance(rate, float)
        assert isinstance(lower, float)
        assert isinstance(upper, float)
