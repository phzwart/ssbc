"""Extended tests for validation_math module to improve coverage."""

import numpy as np
import pytest

from ssbc.reporting import generate_rigorous_pac_report
from ssbc.simulation import BinaryClassifierSimulator
from ssbc.validation_math import (
    extract_calibration_counts,
    validate_beta_binomial_predictive,
    validate_denominator_alignment,
    validate_metric_mathematical_consistency,
    validate_probability_consistency,
)


class TestValidateDenominatorAlignment:
    """Test validate_denominator_alignment function."""

    def test_valid_joint_full_sample(self):
        """Test valid joint full-sample event."""
        result = validate_denominator_alignment(
            k_cal=50, n_cal=100, n_test=200, event_type="joint_full_sample", denominator_fixed=True
        )

        assert result["valid"] is True
        assert "✅" in result["message"]
        assert len(result["issues"]) == 0

    def test_valid_conditional(self):
        """Test valid conditional event."""
        result = validate_denominator_alignment(
            k_cal=10, n_cal=50, n_test=100, event_type="conditional", denominator_fixed=True
        )

        assert result["valid"] is True

    def test_invalid_n_cal_zero(self):
        """Test invalid n_cal = 0."""
        result = validate_denominator_alignment(
            k_cal=0, n_cal=0, n_test=100, event_type="joint_full_sample", denominator_fixed=True
        )

        assert result["valid"] is False
        assert "❌" in result["message"]
        assert len(result["issues"]) > 0

    def test_invalid_k_cal_out_of_range(self):
        """Test invalid k_cal > n_cal."""
        result = validate_denominator_alignment(
            k_cal=150, n_cal=100, n_test=200, event_type="joint_full_sample", denominator_fixed=True
        )

        assert result["valid"] is False
        assert "k_cal must be in [0, n_cal]" in result["issues"][0]

    def test_invalid_n_test_zero(self):
        """Test invalid n_test = 0."""
        result = validate_denominator_alignment(
            k_cal=50, n_cal=100, n_test=0, event_type="joint_full_sample", denominator_fixed=True
        )

        assert result["valid"] is False
        assert "n_test must be > 0" in result["issues"][0]

    def test_nan_values(self):
        """Test NaN values."""
        result = validate_denominator_alignment(
            k_cal=np.nan, n_cal=100, n_test=200, event_type="joint_full_sample", denominator_fixed=True
        )

        assert result["valid"] is False
        assert "Cannot validate (NaN values)" in result["message"]

    def test_joint_full_sample_without_fixed_denominator(self):
        """Test joint_full_sample without fixed denominator (invalid)."""
        result = validate_denominator_alignment(
            k_cal=50, n_cal=100, n_test=200, event_type="joint_full_sample", denominator_fixed=False
        )

        assert result["valid"] is False
        assert "fixed denominators" in result["issues"][0]


class TestValidateProbabilityConsistency:
    """Test validate_probability_consistency function."""

    def test_valid_consistency(self):
        """Test valid probability consistency."""
        rates = {
            "singleton_rate_class0": 0.6,
            "doublet_rate_class0": 0.2,
            "abstention_rate_class0": 0.1,
            "class_rate_class0": 0.9,
        }

        result = validate_probability_consistency(rates, class_label=0, tolerance=0.01)

        # Sum is 0.9, expected is 0.9, difference is 0.0 < tolerance
        assert result["valid"] is True
        assert "✅" in result["message"]
        assert abs(result["sum"] - result["expected"]) < 0.01

    def test_invalid_consistency(self):
        """Test invalid probability consistency."""
        rates = {
            "singleton_rate_class0": 0.6,
            "doublet_rate_class0": 0.2,
            "abstention_rate_class0": 0.1,
            "class_rate_class0": 0.5,  # Sum is 0.9, but expected is 0.5
        }

        result = validate_probability_consistency(rates, class_label=0, tolerance=0.01)

        assert result["valid"] is False
        assert "❌" in result["message"]
        assert result["difference"] > 0.01

    def test_missing_class_prevalence(self):
        """Test missing class prevalence."""
        rates = {
            "singleton_rate_class0": 0.6,
            "doublet_rate_class0": 0.2,
            "abstention_rate_class0": 0.1,
            # Missing class_rate_class0
        }

        result = validate_probability_consistency(rates, class_label=0)

        assert result["valid"] is False
        assert "Cannot validate" in result["message"]
        assert np.isnan(result["expected"])

    def test_class_1(self):
        """Test with class 1."""
        rates = {
            "singleton_rate_class1": 0.4,
            "doublet_rate_class1": 0.1,
            "abstention_rate_class1": 0.0,
            "class_rate_class1": 0.5,
        }

        result = validate_probability_consistency(rates, class_label=1)

        assert result["valid"] is True


class TestValidateBetaBinomialPredictive:
    """Test validate_beta_binomial_predictive function."""

    def test_basic_validation(self):
        """Test basic validation."""
        k_cal = 50
        n_cal = 100
        n_test = 200
        test_rates = np.random.beta(51, 51, size=100)  # Approximate Beta-Binomial rates

        result = validate_beta_binomial_predictive(
            k_cal, n_cal, n_test, test_rates, confidence=0.95, n_simulations=1000
        )

        assert "valid" in result
        assert "message" in result
        assert "empirical_quantiles" in result
        assert "theoretical_quantiles" in result

    def test_invalid_calibration_counts(self):
        """Test with invalid calibration counts."""
        result = validate_beta_binomial_predictive(
            k_cal=np.nan, n_cal=100, n_test=200, test_rates=np.array([0.5, 0.6, 0.7])
        )

        assert result["valid"] is False
        assert "Cannot validate" in result["message"]

    def test_zero_n_cal(self):
        """Test with zero n_cal."""
        result = validate_beta_binomial_predictive(k_cal=0, n_cal=0, n_test=200, test_rates=np.array([0.5, 0.6, 0.7]))

        assert result["valid"] is False


class TestExtractCalibrationCounts:
    """Test extract_calibration_counts function."""

    @pytest.fixture
    def test_report(self):
        """Generate a test PAC report."""
        sim = BinaryClassifierSimulator(p_class1=0.3, beta_params_class0=(2, 5), beta_params_class1=(6, 2), seed=42)
        labels, probs = sim.generate(50)

        report = generate_rigorous_pac_report(
            labels=labels, probs=probs, alpha_target=0.10, delta=0.10, test_size=100, verbose=False
        )

        return report

    def test_extract_marginal_singleton(self, test_report):
        """Test extracting marginal singleton counts."""
        counts = extract_calibration_counts(test_report, "singleton", "marginal")

        assert "k_cal" in counts
        assert "n_cal" in counts
        assert "n_test" in counts
        assert "event_definition" in counts

    def test_extract_class_0_singleton(self, test_report):
        """Test extracting class 0 singleton counts."""
        counts = extract_calibration_counts(test_report, "singleton", "class_0")

        assert "k_cal" in counts
        assert "n_cal" in counts

    def test_extract_unknown_metric(self, test_report):
        """Test extracting unknown metric."""
        counts = extract_calibration_counts(test_report, "unknown_metric", "marginal")

        assert "event_definition" in counts
        assert "Unknown" in str(counts["event_definition"])

    def test_extract_missing_pac_bounds(self):
        """Test extracting from report without pac_bounds."""
        report = {"calibration_result": {0: {"n": 50}, 1: {"n": 50}}}

        counts = extract_calibration_counts(report, "singleton", "marginal")

        assert np.isnan(counts["k_cal"])
        assert np.isnan(counts["n_cal"])


class TestValidateMetricMathematicalConsistency:
    """Test validate_metric_mathematical_consistency function."""

    @pytest.fixture
    def test_report(self):
        """Generate a test PAC report."""
        sim = BinaryClassifierSimulator(p_class1=0.3, beta_params_class0=(2, 5), beta_params_class1=(6, 2), seed=42)
        labels, probs = sim.generate(50)

        report = generate_rigorous_pac_report(
            labels=labels, probs=probs, alpha_target=0.10, delta=0.10, test_size=100, verbose=False
        )

        return report

    def test_basic_validation(self, test_report):
        """Test basic validation."""
        validation_rates = np.array([0.7, 0.8, 0.9, 0.75, 0.85])

        result = validate_metric_mathematical_consistency(
            metric_key="singleton", scope="marginal", report=test_report, validation_rates=validation_rates
        )

        assert "event_definition" in result
        assert "denominator_alignment" in result
        assert "coverage" in result

    def test_class_0_validation(self, test_report):
        """Test class 0 validation."""
        validation_rates = np.array([0.6, 0.7, 0.8, 0.65, 0.75])

        result = validate_metric_mathematical_consistency(
            metric_key="singleton", scope="class_0", report=test_report, validation_rates=validation_rates
        )

        assert "event_definition" in result
        assert "denominator_alignment" in result

    def test_unknown_metric(self, test_report):
        """Test unknown metric."""
        validation_rates = np.array([0.5, 0.6, 0.7])

        result = validate_metric_mathematical_consistency(
            metric_key="unknown_metric", scope="marginal", report=test_report, validation_rates=validation_rates
        )

        # Should still return a result, but with warnings
        assert "event_definition" in result

    def test_validation_with_nan_rates(self, test_report):
        """Test validation with NaN rates."""
        validation_rates = np.array([0.7, np.nan, 0.9, 0.75, np.nan])

        result = validate_metric_mathematical_consistency(
            metric_key="singleton", scope="marginal", report=test_report, validation_rates=validation_rates
        )

        # Should handle NaN gracefully
        assert "event_definition" in result
