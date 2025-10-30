"""Tests for rigorous_report module."""

import numpy as np
import pytest

from ssbc.reporting import generate_rigorous_pac_report
from ssbc.simulation import BinaryClassifierSimulator


@pytest.fixture
def test_data():
    """Generate test data."""
    sim = BinaryClassifierSimulator(p_class1=0.3, beta_params_class0=(2, 5), beta_params_class1=(6, 2), seed=42)
    labels, probs = sim.generate(100)
    return labels, probs, sim


class TestGenerateRigorousPACReport:
    """Test generate_rigorous_pac_report function."""

    def test_basic_report_generation(self, test_data):
        """Test basic report generation."""
        labels, probs, _ = test_data

        report = generate_rigorous_pac_report(labels=labels, probs=probs, alpha_target=0.10, delta=0.10, verbose=False)

        # Check main keys exist
        assert "ssbc_class_0" in report
        assert "ssbc_class_1" in report
        assert "pac_bounds_marginal" in report
        assert "pac_bounds_class_0" in report
        assert "pac_bounds_class_1" in report
        assert "calibration_result" in report
        assert "prediction_stats" in report
        assert "parameters" in report

        # Check SSBC results
        assert report["ssbc_class_0"].n > 0
        assert report["ssbc_class_1"].n > 0
        assert 0 < report["ssbc_class_0"].alpha_corrected < report["ssbc_class_0"].alpha_target

        # Check parameters
        params = report["parameters"]
        assert params["alpha_target"] == {0: 0.10, 1: 0.10}
        assert params["delta"] == {0: 0.10, 1: 0.10}
        assert params["ci_level"] == 0.95
        assert params["use_union_bound"] is False  # Default is False

    def test_different_alpha_per_class(self, test_data):
        """Test with different alpha per class."""
        labels, probs, _ = test_data

        report = generate_rigorous_pac_report(
            labels=labels,
            probs=probs,
            alpha_target={0: 0.05, 1: 0.15},
            delta=0.10,
            verbose=False,
        )

        # Check alphas are different
        params = report["parameters"]
        assert params["alpha_target"][0] == 0.05
        assert params["alpha_target"][1] == 0.15

    def test_different_delta_per_class(self, test_data):
        """Test with different delta per class."""
        labels, probs, _ = test_data

        report = generate_rigorous_pac_report(
            labels=labels,
            probs=probs,
            alpha_target=0.10,
            delta={0: 0.05, 1: 0.15},
            verbose=False,
        )

        # Check deltas are different
        params = report["parameters"]
        assert params["delta"][0] == 0.05
        assert params["delta"][1] == 0.15

        # PAC levels should reflect different deltas
        assert params["pac_level_0"] == 1 - 0.05  # 95%
        assert params["pac_level_1"] == 1 - 0.15  # 85%

    def test_test_size_parameter(self, test_data):
        """Test test_size parameter."""
        labels, probs, _ = test_data

        # Without test_size (defaults to calibration size)
        report_default = generate_rigorous_pac_report(
            labels=labels, probs=probs, alpha_target=0.10, delta=0.10, verbose=False
        )

        assert report_default["parameters"]["test_size"] == len(labels)

        # With explicit test_size
        report_custom = generate_rigorous_pac_report(
            labels=labels, probs=probs, alpha_target=0.10, delta=0.10, test_size=500, verbose=False
        )

        assert report_custom["parameters"]["test_size"] == 500

    def test_union_bound_flag(self, test_data):
        """Test union bound flag."""
        labels, probs, _ = test_data

        # With union bound
        report_union = generate_rigorous_pac_report(
            labels=labels,
            probs=probs,
            alpha_target=0.10,
            delta=0.10,
            use_union_bound=True,
            verbose=False,
        )

        # Without union bound
        report_no_union = generate_rigorous_pac_report(
            labels=labels,
            probs=probs,
            alpha_target=0.10,
            delta=0.10,
            use_union_bound=False,
            verbose=False,
        )

        # Union bound should give wider (or equal) intervals
        width_union = (
            report_union["pac_bounds_marginal"]["singleton_rate_bounds"][1]
            - report_union["pac_bounds_marginal"]["singleton_rate_bounds"][0]
        )
        width_no_union = (
            report_no_union["pac_bounds_marginal"]["singleton_rate_bounds"][1]
            - report_no_union["pac_bounds_marginal"]["singleton_rate_bounds"][0]
        )

        assert width_union >= width_no_union - 1e-10

    # Note: Bootstrap and cross-conformal integration tests removed
    # These features are not currently implemented in generate_rigorous_pac_report

    def test_pac_bounds_structure(self, test_data):
        """Test PAC bounds have correct structure."""
        labels, probs, _ = test_data

        report = generate_rigorous_pac_report(labels=labels, probs=probs, alpha_target=0.10, delta=0.10, verbose=False)

        # Check marginal bounds
        marginal = report["pac_bounds_marginal"]
        required_keys = [
            "singleton_rate_bounds",
            "doublet_rate_bounds",
            "abstention_rate_bounds",
            "singleton_error_rate_bounds",
            "expected_singleton_rate",
            "expected_doublet_rate",
            "expected_abstention_rate",
            "expected_singleton_error_rate",
        ]

        for key in required_keys:
            assert key in marginal

        # Check per-class bounds
        for class_label in [0, 1]:
            class_bounds = report[f"pac_bounds_class_{class_label}"]
            for key in required_keys:
                assert key in class_bounds

    def test_calibration_result_structure(self, test_data):
        """Test calibration result structure."""
        labels, probs, _ = test_data

        report = generate_rigorous_pac_report(labels=labels, probs=probs, alpha_target=0.10, delta=0.10, verbose=False)

        cal_result = report["calibration_result"]

        # Should have results for both classes
        assert len(cal_result) == 2
        assert 0 in cal_result
        assert 1 in cal_result

        # Check structure for each class
        for class_label in [0, 1]:
            class_cal = cal_result[class_label]
            assert "threshold" in class_cal
            assert "n" in class_cal
            assert "k" in class_cal
            assert 0 <= class_cal["threshold"] <= 1

    def test_multiprocessing_parameter(self, test_data):
        """Test n_jobs parameter."""
        labels, probs, _ = test_data

        # Serial
        report_serial = generate_rigorous_pac_report(
            labels=labels,
            probs=probs,
            alpha_target=0.10,
            delta=0.10,
            n_jobs=1,
            verbose=False,
        )

        # Parallel
        report_parallel = generate_rigorous_pac_report(
            labels=labels,
            probs=probs,
            alpha_target=0.10,
            delta=0.10,
            n_jobs=2,
            verbose=False,
        )

        # Results should be identical
        np.testing.assert_allclose(
            report_serial["pac_bounds_marginal"]["singleton_rate_bounds"],
            report_parallel["pac_bounds_marginal"]["singleton_rate_bounds"],
            rtol=1e-10,
        )


class TestEdgeCases:
    """Test edge cases."""

    def test_small_sample_size(self):
        """Test with small sample size (but ensuring each class has >= 10 samples)."""
        sim = BinaryClassifierSimulator(p_class1=0.5, beta_params_class0=(2, 5), beta_params_class1=(6, 2), seed=42)
        # Generate enough samples to ensure each class has at least 10 after split
        labels, probs = sim.generate(50)

        # Should not crash
        report = generate_rigorous_pac_report(labels=labels, probs=probs, alpha_target=0.10, delta=0.10, verbose=False)

        # Should have valid structure
        assert "pac_bounds_marginal" in report
        assert report["ssbc_class_0"].n > 0

    def test_imbalanced_classes(self):
        """Test with highly imbalanced classes."""
        sim = BinaryClassifierSimulator(
            p_class1=0.05, beta_params_class0=(2, 5), beta_params_class1=(6, 2), seed=42
        )  # 5% class 1
        # Generate enough samples to ensure minority class has at least 10 samples
        # With p_class1=0.05, need ~200 samples to reliably get 10+ class 1 samples
        labels, probs = sim.generate(250)

        report = generate_rigorous_pac_report(labels=labels, probs=probs, alpha_target=0.10, delta=0.10, verbose=False)

        # Both classes should have results
        assert report["ssbc_class_0"].n > report["ssbc_class_1"].n
        assert report["ssbc_class_1"].n > 0

    # Note: Bootstrap feature not currently implemented in generate_rigorous_pac_report
