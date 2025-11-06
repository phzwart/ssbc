"""Tests for cross_conformal module."""

import numpy as np
import pytest

from ssbc.calibration import cross_conformal_validation, print_cross_conformal_results
from ssbc.simulation import BinaryClassifierSimulator


@pytest.fixture
def test_data():
    """Generate test data."""
    sim = BinaryClassifierSimulator(p_class1=0.3, beta_params_class0=(2, 5), beta_params_class1=(6, 2), seed=42)
    labels, probs = sim.generate(50)
    return labels, probs


class TestCrossConformalValidation:
    """Test cross_conformal_validation function."""

    def test_basic_validation(self, test_data):
        """Test basic cross-conformal validation."""
        labels, probs = test_data

        results = cross_conformal_validation(
            labels=labels,
            probs=probs,
            alpha_target=0.10,
            delta=0.10,
            n_folds=5,
            stratify=True,
            seed=42,
        )

        # Check structure
        assert "n_folds" in results
        assert "n_samples" in results
        assert "stratified" in results
        assert "fold_rates" in results
        assert "marginal" in results
        assert "class_0" in results
        assert "class_1" in results
        assert "parameters" in results

        # Check values
        assert results["n_folds"] == 5
        assert results["n_samples"] == len(labels)
        assert results["stratified"] is True

    def test_fold_rates_structure(self, test_data):
        """Test fold_rates structure."""
        labels, probs = test_data

        results = cross_conformal_validation(
            labels=labels, probs=probs, alpha_target=0.10, delta=0.10, n_folds=5, seed=42
        )

        fold_rates = results["fold_rates"]

        # Should have one entry per fold
        assert len(fold_rates) == 5

        # Each fold should have marginal and per-class rates
        for fold in fold_rates:
            assert "marginal" in fold
            assert "class_0" in fold
            assert "class_1" in fold

            # Check metrics
            for scope in ["marginal", "class_0", "class_1"]:
                assert "abstention" in fold[scope]
                assert "singleton" in fold[scope]
                assert "doublet" in fold[scope]
                assert "singleton_error" in fold[scope]

    def test_marginal_statistics(self, test_data):
        """Test marginal statistics structure."""
        labels, probs = test_data

        results = cross_conformal_validation(
            labels=labels, probs=probs, alpha_target=0.10, delta=0.10, n_folds=5, seed=42
        )

        marginal = results["marginal"]

        # Check all metrics present
        for metric in ["singleton", "doublet", "abstention", "singleton_error"]:
            assert metric in marginal
            m = marginal[metric]

            # Check structure
            assert "samples" in m
            assert "mean" in m
            assert "std" in m
            assert "quantiles" in m
            assert "ci_95" in m

            # Check types
            assert isinstance(m["samples"], np.ndarray)
            assert len(m["samples"]) == 5  # One per fold
            assert isinstance(m["mean"], float | np.floating)
            assert isinstance(m["std"], float | np.floating)

    def test_quantiles_ordering(self, test_data):
        """Test quantiles are properly ordered."""
        labels, probs = test_data

        results = cross_conformal_validation(
            labels=labels, probs=probs, alpha_target=0.10, delta=0.10, n_folds=5, seed=42
        )

        marginal = results["marginal"]

        for metric in ["singleton", "doublet", "abstention"]:
            q = marginal[metric]["quantiles"]

            # Check ordering
            valid_q = [q[k] for k in ["q05", "q25", "q50", "q75", "q95"] if not np.isnan(q[k])]
            if len(valid_q) > 1:
                assert all(valid_q[i] <= valid_q[i + 1] + 1e-10 for i in range(len(valid_q) - 1))

    def test_stratified_vs_non_stratified(self, test_data):
        """Test stratified vs non-stratified folding."""
        labels, probs = test_data

        # Stratified
        results_stratified = cross_conformal_validation(
            labels=labels,
            probs=probs,
            alpha_target=0.10,
            delta=0.10,
            n_folds=5,
            stratify=True,
            seed=42,
        )

        # Non-stratified
        results_non_stratified = cross_conformal_validation(
            labels=labels,
            probs=probs,
            alpha_target=0.10,
            delta=0.10,
            n_folds=5,
            stratify=False,
            seed=42,
        )

        # Both should have valid results
        assert results_stratified["stratified"] is True
        assert results_non_stratified["stratified"] is False

        # Results may be different
        # Just check both are valid
        assert "marginal" in results_stratified
        assert "marginal" in results_non_stratified

    def test_seed_reproducibility(self, test_data):
        """Test that seed produces reproducible results."""
        labels, probs = test_data

        results1 = cross_conformal_validation(
            labels=labels,
            probs=probs,
            alpha_target=0.10,
            delta=0.10,
            n_folds=5,
            seed=42,
        )

        results2 = cross_conformal_validation(
            labels=labels,
            probs=probs,
            alpha_target=0.10,
            delta=0.10,
            n_folds=5,
            seed=42,
        )

        # Results should be identical
        np.testing.assert_array_equal(
            results1["marginal"]["singleton"]["samples"], results2["marginal"]["singleton"]["samples"]
        )

    def test_different_n_folds(self, test_data):
        """Test with different number of folds."""
        labels, probs = test_data

        for n_folds in [3, 5, 10]:
            results = cross_conformal_validation(
                labels=labels,
                probs=probs,
                alpha_target=0.10,
                delta=0.10,
                n_folds=n_folds,
                seed=42,
            )

            assert results["n_folds"] == n_folds
            assert len(results["fold_rates"]) == n_folds

    def test_parameters_storage(self, test_data):
        """Test that parameters are correctly stored."""
        labels, probs = test_data

        results = cross_conformal_validation(
            labels=labels,
            probs=probs,
            alpha_target=0.15,
            delta=0.05,
            n_folds=7,
            stratify=False,
            seed=123,
        )

        params = results["parameters"]

        assert params["alpha_target"] == 0.15
        assert params["delta"] == 0.05
        assert params["n_folds"] == 7
        assert params["stratify"] is False


class TestPrintCrossConformalResults:
    """Test print_cross_conformal_results function."""

    def test_basic_printing(self, test_data, capsys):
        """Test basic printing."""
        labels, probs = test_data

        results = cross_conformal_validation(
            labels=labels, probs=probs, alpha_target=0.10, delta=0.10, n_folds=5, seed=42
        )

        print_cross_conformal_results(results)

        captured = capsys.readouterr()

        # Check output contains expected sections
        assert "CROSS-CONFORMAL VALIDATION RESULTS" in captured.out
        assert "MARGINAL RATES" in captured.out
        assert "CLASS 0 RATES" in captured.out
        assert "CLASS 1 RATES" in captured.out
        assert "SINGLETON" in captured.out
        assert "DOUBLET" in captured.out
        assert "ABSTENTION" in captured.out

    def test_prints_parameters(self, test_data, capsys):
        """Test that parameters are printed."""
        labels, probs = test_data

        results = cross_conformal_validation(
            labels=labels, probs=probs, alpha_target=0.10, delta=0.10, n_folds=5, seed=42
        )

        print_cross_conformal_results(results)

        captured = capsys.readouterr()

        assert "K-folds: 5" in captured.out
        assert "Samples: " in captured.out


class TestEdgeCases:
    """Test edge cases."""

    def test_small_sample_size(self):
        """Test with small sample size (ensuring each class has >= 10 samples)."""
        sim = BinaryClassifierSimulator(p_class1=0.5, beta_params_class0=(2, 5), beta_params_class1=(6, 2), seed=42)
        labels, probs = sim.generate(50)  # Enough to ensure each class has >= 10

        # Use fewer folds for small sample
        results = cross_conformal_validation(
            labels=labels,
            probs=probs,
            alpha_target=0.10,
            delta=0.10,
            n_folds=3,
            seed=42,
        )

        # Should handle gracefully
        assert results["n_samples"] == 50
        assert results["n_folds"] == 3

    def test_imbalanced_classes(self):
        """Test with imbalanced classes."""
        sim = BinaryClassifierSimulator(
            p_class1=0.1, beta_params_class0=(2, 5), beta_params_class1=(6, 2), seed=42
        )  # 10% class 1
        # Need enough samples so that each fold has at least 10 samples per class
        # With n_folds=5 and p_class1=0.1, need ~500 samples total
        labels, probs = sim.generate(500)

        results = cross_conformal_validation(
            labels=labels,
            probs=probs,
            alpha_target=0.10,
            delta=0.10,
            n_folds=5,
            stratify=True,
            seed=42,
        )

        # Should have results for both classes
        assert "class_0" in results
        assert "class_1" in results

    def test_perfect_predictions(self):
        """Test with perfect predictions."""
        # Create perfect predictions - need enough so each fold has >= 10 per class
        # With n_folds=2, need at least 40 samples total (20 per fold, 10 per class per fold)
        labels = np.array([0] * 20 + [1] * 20)
        probs = np.zeros((40, 2))
        for i in range(40):
            probs[i, labels[i]] = 1.0

        results = cross_conformal_validation(
            labels=labels,
            probs=probs,
            alpha_target=0.10,
            delta=0.10,
            n_folds=2,
            seed=42,
            stratify=True,  # Stratified to ensure balanced classes per fold
        )

        # Should handle perfect predictions
        assert "marginal" in results

    def test_nan_handling(self, test_data):
        """Test handling of NaN values."""
        labels, probs = test_data

        results = cross_conformal_validation(
            labels=labels, probs=probs, alpha_target=0.10, delta=0.10, n_folds=5, seed=42
        )

        # singleton_error may contain NaN
        # Statistics should handle this correctly
        singleton_error = results["marginal"]["singleton_error"]

        # Mean should be a valid number or NaN
        assert isinstance(singleton_error["mean"], float | np.floating)
