"""Tests for utils module."""

import numpy as np
import pytest

from ssbc.utils import compute_operational_rate, evaluate_test_dataset


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
            compute_operational_rate(pred_sets, true_labels, "invalid")  # type: ignore[arg-type]

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


class TestEvaluateTestDataset:
    """Test evaluate_test_dataset function."""

    def test_basic_functionality(self):
        """Test basic functionality with simple case."""
        test_labels = np.array([0, 0, 1, 1, 0])
        test_probs = np.array(
            [
                [0.8, 0.2],  # High confidence class 0
                [0.6, 0.4],  # Medium confidence class 0
                [0.3, 0.7],  # High confidence class 1
                [0.4, 0.6],  # Medium confidence class 1
                [0.5, 0.5],  # Uncertain
            ]
        )

        results = evaluate_test_dataset(test_labels, test_probs, 0.3, 0.3)

        # Check structure
        assert "marginal" in results
        assert "class_0" in results
        assert "class_1" in results
        assert "thresholds" in results
        assert "n_test" in results

        # Check marginal rates exist
        marginal = results["marginal"]
        assert "singleton_rate" in marginal
        assert "doublet_rate" in marginal
        assert "abstention_rate" in marginal
        assert "singleton_error_rate" in marginal
        assert "n_samples" in marginal

        # Check thresholds
        assert results["thresholds"]["threshold_0"] == 0.3
        assert results["thresholds"]["threshold_1"] == 0.3
        assert results["n_test"] == 5

    def test_empty_dataset(self):
        """Test with empty dataset."""
        test_labels = np.array([])
        test_probs = np.array([]).reshape(0, 2)

        with pytest.raises(ValueError, match="Test dataset cannot be empty"):
            evaluate_test_dataset(test_labels, test_probs, 0.3, 0.3)

    def test_wrong_probs_shape(self):
        """Test with wrong probability shape."""
        test_labels = np.array([0, 1])
        test_probs = np.array([[0.8, 0.2]])  # Wrong shape

        with pytest.raises(ValueError, match="test_probs must have shape"):
            evaluate_test_dataset(test_labels, test_probs, 0.3, 0.3)

    def test_all_singletons(self):
        """Test case where all predictions are singletons."""
        test_labels = np.array([0, 1, 0, 1])
        test_probs = np.array(
            [
                [0.8, 0.2],  # High confidence class 0
                [0.2, 0.8],  # High confidence class 1
                [0.7, 0.3],  # High confidence class 0
                [0.3, 0.7],  # High confidence class 1
            ]
        )

        # Use high thresholds to ensure all are singletons
        # Scores are 1 - prob, so high confidence = low scores
        results = evaluate_test_dataset(test_labels, test_probs, 0.5, 0.5)

        # All should be singletons
        assert results["marginal"]["singleton_rate"] == 1.0
        assert results["marginal"]["doublet_rate"] == 0.0
        assert results["marginal"]["abstention_rate"] == 0.0
        assert results["marginal"]["n_singletons"] == 4
        assert results["marginal"]["n_doublets"] == 0
        assert results["marginal"]["n_abstentions"] == 0

    def test_all_doublets(self):
        """Test case where all predictions are doublets."""
        test_labels = np.array([0, 1, 0, 1])
        test_probs = np.array(
            [
                [0.5, 0.5],  # Uncertain
                [0.5, 0.5],  # Uncertain
                [0.5, 0.5],  # Uncertain
                [0.5, 0.5],  # Uncertain
            ]
        )

        # Use very high thresholds to ensure all are doublets
        results = evaluate_test_dataset(test_labels, test_probs, 0.9, 0.9)

        # All should be doublets
        assert results["marginal"]["singleton_rate"] == 0.0
        assert results["marginal"]["doublet_rate"] == 1.0
        assert results["marginal"]["abstention_rate"] == 0.0
        assert results["marginal"]["n_singletons"] == 0
        assert results["marginal"]["n_doublets"] == 4
        assert results["marginal"]["n_abstentions"] == 0

    def test_all_abstentions(self):
        """Test case where all predictions are abstentions."""
        test_labels = np.array([0, 1, 0, 1])
        test_probs = np.array(
            [
                [0.5, 0.5],  # Uncertain
                [0.5, 0.5],  # Uncertain
                [0.5, 0.5],  # Uncertain
                [0.5, 0.5],  # Uncertain
            ]
        )

        # Use very low thresholds to ensure all are abstentions
        # Scores are 1 - prob = 0.5, which is > 0.1, so not included
        results = evaluate_test_dataset(test_labels, test_probs, 0.1, 0.1)

        # All should be abstentions
        assert results["marginal"]["singleton_rate"] == 0.0
        assert results["marginal"]["doublet_rate"] == 0.0
        assert results["marginal"]["abstention_rate"] == 1.0
        assert results["marginal"]["n_singletons"] == 0
        assert results["marginal"]["n_doublets"] == 0
        assert results["marginal"]["n_abstentions"] == 4

    def test_mixed_predictions(self):
        """Test case with mixed prediction types."""
        test_labels = np.array([0, 0, 1, 1, 0])
        test_probs = np.array(
            [
                [0.8, 0.2],  # High confidence class 0 -> singleton
                [0.5, 0.5],  # Uncertain -> abstention
                [0.2, 0.8],  # High confidence class 1 -> singleton
                [0.4, 0.6],  # Medium confidence -> doublet
                [0.3, 0.7],  # High confidence class 1 -> singleton
            ]
        )

        results = evaluate_test_dataset(test_labels, test_probs, 0.3, 0.3)

        # Check that rates sum to 1
        total_rate = (
            results["marginal"]["singleton_rate"]
            + results["marginal"]["doublet_rate"]
            + results["marginal"]["abstention_rate"]
        )
        assert abs(total_rate - 1.0) < 1e-10

    def test_per_class_rates(self):
        """Test per-class rate computation."""
        test_labels = np.array([0, 0, 1, 1])
        test_probs = np.array(
            [
                [0.8, 0.2],  # High confidence class 0
                [0.5, 0.5],  # Uncertain
                [0.2, 0.8],  # High confidence class 1
                [0.4, 0.6],  # Medium confidence
            ]
        )

        results = evaluate_test_dataset(test_labels, test_probs, 0.3, 0.3)

        # Check class 0 rates
        class_0 = results["class_0"]
        assert class_0["n_samples"] == 2

        # Check class 1 rates
        class_1 = results["class_1"]
        assert class_1["n_samples"] == 2

    def test_singleton_error_rate(self):
        """Test singleton error rate computation."""
        test_labels = np.array([0, 1, 0, 1])
        test_probs = np.array(
            [
                [0.8, 0.2],  # High confidence class 0 (correct)
                [0.8, 0.2],  # High confidence class 0 (wrong - true is 1)
                [0.2, 0.8],  # High confidence class 1 (correct)
                [0.2, 0.8],  # High confidence class 1 (wrong - true is 0)
            ]
        )

        results = evaluate_test_dataset(test_labels, test_probs, 0.3, 0.3)

        # All should be singletons, half should be errors
        assert results["marginal"]["singleton_rate"] == 1.0
        assert results["marginal"]["singleton_error_rate"] == 0.5

    def test_no_singletons_error_rate(self):
        """Test singleton error rate when no singletons."""
        test_labels = np.array([0, 1])
        test_probs = np.array(
            [
                [0.5, 0.5],  # Uncertain -> abstention
                [0.5, 0.5],  # Uncertain -> abstention
            ]
        )

        results = evaluate_test_dataset(test_labels, test_probs, 0.1, 0.1)

        # No singletons, so error rate should be NaN
        assert np.isnan(results["marginal"]["singleton_error_rate"])

    def test_single_sample(self):
        """Test with single sample."""
        test_labels = np.array([0])
        test_probs = np.array([[0.8, 0.2]])

        results = evaluate_test_dataset(test_labels, test_probs, 0.3, 0.3)

        assert results["n_test"] == 1
        assert results["marginal"]["n_samples"] == 1
        assert results["class_0"]["n_samples"] == 1
        assert results["class_1"]["n_samples"] == 0

    def test_threshold_effects(self):
        """Test that different thresholds produce different results."""
        test_labels = np.array([0, 1, 0, 1])
        test_probs = np.array(
            [
                [0.6, 0.4],  # Medium confidence class 0
                [0.4, 0.6],  # Medium confidence class 1
                [0.6, 0.4],  # Medium confidence class 0
                [0.4, 0.6],  # Medium confidence class 1
            ]
        )

        # Low thresholds (more restrictive)
        results_low = evaluate_test_dataset(test_labels, test_probs, 0.1, 0.1)

        # High thresholds (less restrictive)
        results_high = evaluate_test_dataset(test_labels, test_probs, 0.9, 0.9)

        # High thresholds should have more doublets, fewer abstentions
        assert results_high["marginal"]["doublet_rate"] > results_low["marginal"]["doublet_rate"]
        assert results_high["marginal"]["abstention_rate"] < results_low["marginal"]["abstention_rate"]

    def test_return_types(self):
        """Test that return values have correct types."""
        test_labels = np.array([0, 1])
        test_probs = np.array([[0.8, 0.2], [0.2, 0.8]])

        results = evaluate_test_dataset(test_labels, test_probs, 0.3, 0.3)

        # Check types
        assert isinstance(results["marginal"]["singleton_rate"], float)
        assert isinstance(results["marginal"]["n_samples"], int)
        assert isinstance(results["thresholds"]["threshold_0"], float)
        assert isinstance(results["n_test"], int)

    def test_rates_between_zero_and_one(self):
        """Test that all rates are between 0 and 1."""
        test_labels = np.array([0, 1, 0, 1, 0])
        test_probs = np.array([[0.8, 0.2], [0.2, 0.8], [0.5, 0.5], [0.3, 0.7], [0.6, 0.4]])

        results = evaluate_test_dataset(test_labels, test_probs, 0.3, 0.3)

        for scope in ["marginal", "class_0", "class_1"]:
            scope_results = results[scope]
            for rate_name in ["singleton_rate", "doublet_rate", "abstention_rate"]:
                if not np.isnan(scope_results[rate_name]):
                    assert 0.0 <= scope_results[rate_name] <= 1.0
