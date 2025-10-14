"""Tests for utils module."""

import numpy as np
import pytest

from ssbc.utils import compute_operational_rate


class TestComputeOperationalRate:
    """Test compute_operational_rate function."""

    def test_singleton_detection(self):
        """Test singleton detection."""
        pred_sets = [{0}, {0, 1}, set(), {1}]
        true_labels = np.array([0, 0, 1, 0])

        indicators = compute_operational_rate(pred_sets, true_labels, "singleton")

        # First and last are singletons
        expected = np.array([1, 0, 0, 1])
        np.testing.assert_array_equal(indicators, expected)

    def test_doublet_detection(self):
        """Test doublet detection."""
        pred_sets = [{0}, {0, 1}, set(), {1}]
        true_labels = np.array([0, 0, 1, 0])

        indicators = compute_operational_rate(pred_sets, true_labels, "doublet")

        # Only second is a doublet
        expected = np.array([0, 1, 0, 0])
        np.testing.assert_array_equal(indicators, expected)

    def test_abstention_detection(self):
        """Test abstention detection."""
        pred_sets = [{0}, {0, 1}, set(), {1}]
        true_labels = np.array([0, 0, 1, 0])

        indicators = compute_operational_rate(pred_sets, true_labels, "abstention")

        # Only third is an abstention
        expected = np.array([0, 0, 1, 0])
        np.testing.assert_array_equal(indicators, expected)

    def test_correct_in_singleton(self):
        """Test correct singleton detection."""
        pred_sets = [{0}, {0, 1}, set(), {1}, {0}]
        true_labels = np.array([0, 0, 1, 0, 1])  # Last is singleton but wrong

        indicators = compute_operational_rate(pred_sets, true_labels, "correct_in_singleton")

        # First is singleton and correct, last is singleton but incorrect
        expected = np.array([1, 0, 0, 0, 0])
        np.testing.assert_array_equal(indicators, expected)

    def test_error_in_singleton(self):
        """Test error in singleton detection."""
        pred_sets = [{0}, {0, 1}, set(), {1}, {0}]
        true_labels = np.array([0, 0, 1, 0, 1])  # Last is singleton but wrong

        indicators = compute_operational_rate(pred_sets, true_labels, "error_in_singleton")

        # Last is singleton and incorrect
        expected = np.array([0, 0, 0, 1, 1])
        np.testing.assert_array_equal(indicators, expected)

    def test_all_singletons(self):
        """Test case with only singletons."""
        pred_sets = [{0}, {1}, {0}, {1}]
        true_labels = np.array([0, 1, 0, 1])

        indicators = compute_operational_rate(pred_sets, true_labels, "singleton")

        # All are singletons
        expected = np.array([1, 1, 1, 1])
        np.testing.assert_array_equal(indicators, expected)

    def test_all_doublets(self):
        """Test case with only doublets."""
        pred_sets = [{0, 1}, {0, 1}, {0, 1}]
        true_labels = np.array([0, 1, 0])

        indicators = compute_operational_rate(pred_sets, true_labels, "doublet")

        # All are doublets
        expected = np.array([1, 1, 1])
        np.testing.assert_array_equal(indicators, expected)

    def test_all_abstentions(self):
        """Test case with only abstentions."""
        pred_sets = [set(), set(), set()]
        true_labels = np.array([0, 1, 0])

        indicators = compute_operational_rate(pred_sets, true_labels, "abstention")

        # All are abstentions
        expected = np.array([1, 1, 1])
        np.testing.assert_array_equal(indicators, expected)

    def test_empty_input(self):
        """Test with empty input."""
        pred_sets = []
        true_labels = np.array([])

        indicators = compute_operational_rate(pred_sets, true_labels, "singleton")

        # Should return empty array
        assert len(indicators) == 0

    def test_single_sample(self):
        """Test with single sample."""
        # Single singleton
        indicators = compute_operational_rate([{0}], np.array([0]), "singleton")
        assert indicators[0] == 1

        # Single doublet
        indicators = compute_operational_rate([{0, 1}], np.array([0]), "doublet")
        assert indicators[0] == 1

        # Single abstention
        indicators = compute_operational_rate([set()], np.array([0]), "abstention")
        assert indicators[0] == 1

    def test_large_array(self):
        """Test with large array."""
        n = 1000
        # Create random prediction sets
        pred_sets = []
        for _ in range(n):
            size = np.random.choice([0, 1, 2])
            if size == 0:
                pred_sets.append(set())
            elif size == 1:
                pred_sets.append({np.random.choice([0, 1])})
            else:
                pred_sets.append({0, 1})

        true_labels = np.random.choice([0, 1], size=n)

        indicators = compute_operational_rate(pred_sets, true_labels, "singleton")

        # Should be binary
        assert set(indicators) <= {0, 1}
        assert len(indicators) == n

    def test_invalid_rate_type(self):
        """Test with invalid rate type."""
        pred_sets = [{0}]
        true_labels = np.array([0])

        with pytest.raises(ValueError, match="Unknown rate_type"):
            compute_operational_rate(pred_sets, true_labels, "invalid")

    def test_return_type(self):
        """Test return type is ndarray."""
        pred_sets = [{0}, {0, 1}, {1}]
        true_labels = np.array([0, 0, 1])

        indicators = compute_operational_rate(pred_sets, true_labels, "singleton")

        assert isinstance(indicators, np.ndarray)
        assert indicators.dtype == np.int64 or indicators.dtype == np.int32

    def test_list_vs_set_input(self):
        """Test with list instead of set."""
        # Using lists instead of sets
        pred_sets = [[0], [0, 1], [], [1]]
        true_labels = np.array([0, 0, 1, 0])

        indicators = compute_operational_rate(pred_sets, true_labels, "singleton")

        expected = np.array([1, 0, 0, 1])
        np.testing.assert_array_equal(indicators, expected)

    def test_correct_vs_error_singletons(self):
        """Test that correct + error singletons = all singletons."""
        pred_sets = [{0}, {1}, {0}, {1}, {0, 1}]
        true_labels = np.array([0, 0, 1, 1, 0])  # First and third wrong

        singletons = compute_operational_rate(pred_sets, true_labels, "singleton")
        correct = compute_operational_rate(pred_sets, true_labels, "correct_in_singleton")
        errors = compute_operational_rate(pred_sets, true_labels, "error_in_singleton")

        # correct + errors should equal singletons
        np.testing.assert_array_equal(singletons, correct + errors)
