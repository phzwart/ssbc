"""Tests for validation module."""

import numpy as np
import pytest

from ssbc.reporting import generate_rigorous_pac_report
from ssbc.simulation import BinaryClassifierSimulator
from ssbc.validation import print_validation_results, validate_pac_bounds


@pytest.fixture
def test_report():
    """Generate a test PAC report."""
    sim = BinaryClassifierSimulator(p_class1=0.3, beta_params_class0=(2, 5), beta_params_class1=(6, 2), seed=42)
    labels, probs = sim.generate(50)

    report = generate_rigorous_pac_report(
        labels=labels, probs=probs, alpha_target=0.10, delta=0.10, test_size=100, verbose=False
    )

    return report, sim


class TestValidatePACBounds:
    """Test validate_pac_bounds function."""

    def test_basic_validation(self, test_report):
        """Test basic validation workflow."""
        report, sim = test_report

        validation = validate_pac_bounds(report=report, simulator=sim, test_size=100, n_trials=10, verbose=False)

        # Check structure
        assert "n_trials" in validation
        assert "test_size" in validation
        assert "threshold_0" in validation
        assert "threshold_1" in validation
        assert "marginal" in validation
        assert "class_0" in validation
        assert "class_1" in validation

        # Check values
        assert validation["n_trials"] == 10
        assert validation["test_size"] == 100

    def test_marginal_structure(self, test_report):
        """Test marginal validation structure."""
        report, sim = test_report

        validation = validate_pac_bounds(report=report, simulator=sim, test_size=100, n_trials=10, verbose=False)

        marginal = validation["marginal"]

        # Check all metrics present
        for metric in ["singleton", "doublet", "abstention", "singleton_error"]:
            assert metric in marginal
            m = marginal[metric]

            # Check structure
            assert "rates" in m
            assert "mean" in m
            assert "quantiles" in m
            assert "bounds" in m
            assert "expected" in m
            assert "empirical_coverage" in m

            # Check values
            assert isinstance(m["rates"], np.ndarray)
            assert len(m["rates"]) == 10
            assert isinstance(m["mean"], float | np.floating)
            assert isinstance(m["quantiles"], dict)
            assert isinstance(m["bounds"], list | tuple)  # Can be list or tuple
            assert len(m["bounds"]) == 2  # Should have lower and upper
            assert isinstance(m["expected"], float | np.floating)

    def test_per_class_structure(self, test_report):
        """Test per-class validation structure."""
        report, sim = test_report

        validation = validate_pac_bounds(report=report, simulator=sim, test_size=100, n_trials=10, verbose=False)

        for class_label in [0, 1]:
            class_val = validation[f"class_{class_label}"]

            for metric in ["singleton", "doublet", "abstention", "singleton_error"]:
                assert metric in class_val
                m = class_val[metric]

                assert "rates" in m
                assert "empirical_coverage" in m

    def test_empirical_coverage_values(self, test_report):
        """Test empirical coverage values are valid."""
        report, sim = test_report

        validation = validate_pac_bounds(report=report, simulator=sim, test_size=100, n_trials=20, verbose=False)

        # Check coverage is between 0 and 1 (or NaN)
        for scope in ["marginal", "class_0", "class_1"]:
            for metric in ["singleton", "doublet", "abstention", "singleton_error"]:
                coverage = validation[scope][metric]["empirical_coverage"]
                assert np.isnan(coverage) or (0 <= coverage <= 1)

    def test_quantiles_ordering(self, test_report):
        """Test quantiles are properly ordered."""
        report, sim = test_report

        validation = validate_pac_bounds(report=report, simulator=sim, test_size=100, n_trials=50, verbose=False)

        marginal = validation["marginal"]

        for metric in ["singleton", "doublet", "abstention"]:
            q = marginal[metric]["quantiles"]

            # Check ordering
            assert q["q05"] <= q["q25"] <= q["q50"] <= q["q75"] <= q["q95"]

    def test_rates_sum_to_one(self, test_report):
        """Test that marginal rates approximately sum to 1."""
        report, sim = test_report

        validation = validate_pac_bounds(report=report, simulator=sim, test_size=100, n_trials=20, verbose=False)

        marginal = validation["marginal"]

        # Check each trial
        for i in range(20):
            rate_sum = (
                marginal["singleton"]["rates"][i] + marginal["doublet"]["rates"][i] + marginal["abstention"]["rates"][i]
            )

            np.testing.assert_allclose(rate_sum, 1.0, rtol=1e-10)

    def test_seed_reproducibility(self, test_report):
        """Test that seed produces reproducible results."""
        report, _ = test_report

        # Create fresh simulators with same seed for reproducibility
        from ssbc.simulation import BinaryClassifierSimulator

        sim1 = BinaryClassifierSimulator(p_class1=0.3, beta_params_class0=(2, 5), beta_params_class1=(6, 2), seed=100)
        sim2 = BinaryClassifierSimulator(p_class1=0.3, beta_params_class0=(2, 5), beta_params_class1=(6, 2), seed=100)

        validation1 = validate_pac_bounds(
            report=report, simulator=sim1, test_size=100, n_trials=10, seed=42, verbose=False
        )

        validation2 = validate_pac_bounds(
            report=report, simulator=sim2, test_size=100, n_trials=10, seed=42, verbose=False
        )

        # Results should be very similar (allowing for numerical differences)
        np.testing.assert_allclose(
            validation1["marginal"]["singleton"]["mean"], validation2["marginal"]["singleton"]["mean"], rtol=0.01
        )

    def test_verbose_output(self, test_report, capsys):
        """Test verbose output."""
        report, sim = test_report

        validate_pac_bounds(report=report, simulator=sim, test_size=100, n_trials=5, verbose=True)

        captured = capsys.readouterr()
        assert "Using fixed thresholds" in captured.out
        assert "Running" in captured.out


class TestPrintValidationResults:
    """Test print_validation_results function."""

    def test_basic_printing(self, test_report, capsys):
        """Test basic printing."""
        report, sim = test_report

        validation = validate_pac_bounds(report=report, simulator=sim, test_size=100, n_trials=10, verbose=False)

        print_validation_results(validation)

        captured = capsys.readouterr()

        # Check output contains expected sections
        assert "PAC BOUNDS VALIDATION RESULTS" in captured.out
        assert "MARGINAL" in captured.out
        assert "CLASS 0" in captured.out
        assert "CLASS 1" in captured.out
        assert "SINGLETON" in captured.out
        assert "DOUBLET" in captured.out
        assert "ABSTENTION" in captured.out

    def test_prints_coverage(self, test_report, capsys):
        """Test that coverage is printed."""
        report, sim = test_report

        validation = validate_pac_bounds(report=report, simulator=sim, test_size=100, n_trials=10, verbose=False)

        print_validation_results(validation)

        captured = capsys.readouterr()

        assert "Coverage:" in captured.out


class TestEdgeCases:
    """Test edge cases."""

    def test_small_n_trials(self, test_report):
        """Test with very small number of trials."""
        report, sim = test_report

        validation = validate_pac_bounds(report=report, simulator=sim, test_size=100, n_trials=2, verbose=False)

        # Should still work
        assert validation["n_trials"] == 2
        assert len(validation["marginal"]["singleton"]["rates"]) == 2

    def test_different_test_size(self, test_report):
        """Test with different test size than report."""
        report, sim = test_report

        # Report has test_size=100, validate with 50
        validation = validate_pac_bounds(report=report, simulator=sim, test_size=50, n_trials=10, verbose=False)

        assert validation["test_size"] == 50

    def test_nan_handling(self, test_report):
        """Test handling of NaN values in singleton error rates."""
        report, sim = test_report

        validation = validate_pac_bounds(report=report, simulator=sim, test_size=100, n_trials=10, verbose=False)

        # singleton_error rates may contain NaN (when no singletons)
        # Coverage calculation should handle this
        coverage = validation["marginal"]["singleton_error"]["empirical_coverage"]

        # Should be a number or NaN, not crash
        assert isinstance(coverage, float | np.floating)
