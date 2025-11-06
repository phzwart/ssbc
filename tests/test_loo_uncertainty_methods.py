"""Tests for LOO uncertainty methods: simple, beta_binomial, hoeffding."""

import numpy as np
import pytest

from ssbc.metrics.loo_uncertainty import (
    compute_loo_corrected_prediction_bounds,
    format_prediction_bounds_report,
)


class TestComputeLOOCorrectedPredictionBounds:
    """Test compute_loo_corrected_prediction_bounds with different methods."""

    def test_simple_method_basic(self):
        """Test simple method with normal case."""
        loo_preds = np.array([1, 0, 1, 1, 0, 1, 0, 1, 0, 1])
        n_test = 100
        alpha = 0.05

        lower, upper, diagnostics = compute_loo_corrected_prediction_bounds(
            loo_preds, n_test, alpha, method="simple", verbose=False
        )

        assert 0 <= lower <= upper <= 1
        assert diagnostics["method"] == "clopper_pearson_plus_sampling"
        assert "cp_lower" in diagnostics
        assert "cp_upper" in diagnostics
        assert "inflation_factor" in diagnostics

    def test_simple_method_k_zero(self):
        """Test simple method when k_cal = 0."""
        loo_preds = np.zeros(10)
        n_test = 100
        alpha = 0.05

        lower, upper, diagnostics = compute_loo_corrected_prediction_bounds(
            loo_preds, n_test, alpha, method="simple", verbose=False
        )

        assert lower == 0.0
        assert 0 <= upper <= 1
        assert diagnostics["cp_lower"] == 0.0

    def test_simple_method_k_equals_n(self):
        """Test simple method when k_cal = n_cal."""
        loo_preds = np.ones(10)
        n_test = 100
        alpha = 0.05

        lower, upper, diagnostics = compute_loo_corrected_prediction_bounds(
            loo_preds, n_test, alpha, method="simple", verbose=False
        )

        assert 0 <= lower <= 1
        assert upper == 1.0
        assert diagnostics["cp_upper"] == 1.0

    def test_simple_method_with_inflation_factor(self):
        """Test simple method with provided inflation factor."""
        loo_preds = np.array([1, 0, 1, 1, 0, 1])
        n_test = 100
        alpha = 0.05
        inflation_factor = 2.5

        lower, upper, diagnostics = compute_loo_corrected_prediction_bounds(
            loo_preds, n_test, alpha, method="simple", inflation_factor=inflation_factor, verbose=False
        )

        assert diagnostics["inflation_factor"] == inflation_factor
        assert 0 <= lower <= upper <= 1

    def test_beta_binomial_method(self):
        """Test beta_binomial method."""
        loo_preds = np.array([1, 0, 1, 1, 0, 1, 0, 1, 0, 1])
        n_test = 100
        alpha = 0.05

        lower, upper, diagnostics = compute_loo_corrected_prediction_bounds(
            loo_preds, n_test, alpha, method="beta_binomial", verbose=False
        )

        assert 0 <= lower <= upper <= 1
        assert diagnostics["method"] == "beta_binomial_loo_corrected"
        assert "n_effective" in diagnostics
        assert "n_eff_int" in diagnostics
        assert "k_eff_int" in diagnostics

    def test_beta_binomial_method_with_inflation_factor(self):
        """Test beta_binomial method with provided inflation factor."""
        loo_preds = np.array([1, 0, 1, 1, 0, 1])
        n_test = 100
        alpha = 0.05
        inflation_factor = 2.0

        lower, upper, diagnostics = compute_loo_corrected_prediction_bounds(
            loo_preds, n_test, alpha, method="beta_binomial", inflation_factor=inflation_factor, verbose=False
        )

        assert diagnostics["inflation_factor"] == inflation_factor
        assert 0 <= lower <= upper <= 1

    def test_hoeffding_method(self):
        """Test hoeffding method."""
        loo_preds = np.array([1, 0, 1, 1, 0, 1, 0, 1, 0, 1])
        n_test = 100
        alpha = 0.05

        lower, upper, diagnostics = compute_loo_corrected_prediction_bounds(
            loo_preds, n_test, alpha, method="hoeffding", verbose=False
        )

        assert 0 <= lower <= upper <= 1
        assert diagnostics["method"] == "hoeffding_distribution_free"
        assert "t_cal" in diagnostics
        assert "t_test" in diagnostics
        assert "cal_lower" in diagnostics
        assert "cal_upper" in diagnostics

    def test_hoeffding_method_with_inflation_factor(self):
        """Test hoeffding method with provided inflation factor."""
        loo_preds = np.array([1, 0, 1, 1, 0, 1])
        n_test = 100
        alpha = 0.05
        inflation_factor = 1.5

        lower, upper, diagnostics = compute_loo_corrected_prediction_bounds(
            loo_preds, n_test, alpha, method="hoeffding", inflation_factor=inflation_factor, verbose=False
        )

        assert diagnostics["inflation_factor"] == inflation_factor
        assert 0 <= lower <= upper <= 1

    def test_invalid_method(self):
        """Test that invalid method raises error."""
        loo_preds = np.array([1, 0, 1, 1, 0, 1])
        n_test = 100
        alpha = 0.05

        with pytest.raises(ValueError, match="Unknown method"):
            compute_loo_corrected_prediction_bounds(loo_preds, n_test, alpha, method="invalid", verbose=False)

    def test_verbose_output(self, capsys):
        """Test verbose output (method may not produce output, but should not error)."""
        loo_preds = np.array([1, 0, 1, 1, 0, 1])
        n_test = 100
        alpha = 0.05

        # Should not raise error with verbose=True
        lower, upper, diagnostics = compute_loo_corrected_prediction_bounds(
            loo_preds, n_test, alpha, method="simple", verbose=True
        )

        assert 0 <= lower <= upper <= 1
        assert diagnostics["method"] == "clopper_pearson_plus_sampling"


class TestFormatPredictionBoundsReport:
    """Test format_prediction_bounds_report function."""

    def test_basic_report(self):
        """Test basic report generation."""
        loo_preds = np.array([1, 0, 1, 1, 0, 1, 0, 1, 0, 1])
        n_test = 100
        alpha = 0.05

        report = format_prediction_bounds_report("Singleton Rate", loo_preds, n_test, alpha, include_all_methods=False)

        assert isinstance(report, str)
        assert "Singleton Rate" in report
        assert "PREDICTION BOUNDS" in report
        assert "n_cal = 10" in report
        assert "n_test = 100" in report

    def test_report_with_all_methods(self):
        """Test report with all methods comparison."""
        loo_preds = np.array([1, 0, 1, 1, 0, 1, 0, 1, 0, 1])
        n_test = 100
        alpha = 0.05

        report = format_prediction_bounds_report("Singleton Rate", loo_preds, n_test, alpha, include_all_methods=True)

        assert isinstance(report, str)
        assert "METHOD COMPARISON" in report
        assert "analytical" in report.lower() or "exact" in report.lower() or "hoeffding" in report.lower()

    def test_report_structure(self):
        """Test that report has expected structure."""
        loo_preds = np.array([1, 0, 1, 1, 0, 1])
        n_test = 50
        alpha = 0.05

        report = format_prediction_bounds_report("Doublet Rate", loo_preds, n_test, alpha, include_all_methods=False)

        # Check for key sections
        assert "Calibration Data" in report
        assert "Test Data" in report
        assert "Confidence Level" in report
        assert "PREDICTION INTERVAL" in report
        assert "INTERPRETATION" in report

    def test_report_with_diagnostics(self):
        """Test report includes diagnostics when available."""
        # Use method that provides diagnostics
        loo_preds = np.array([1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1])
        n_test = 100
        alpha = 0.05

        report = format_prediction_bounds_report("Abstention Rate", loo_preds, n_test, alpha, include_all_methods=True)

        # Should include uncertainty breakdown if diagnostics available
        assert isinstance(report, str)
        assert len(report) > 0
