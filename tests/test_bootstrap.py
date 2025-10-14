"""Tests for bootstrap module."""

import numpy as np
import pytest

from ssbc.bootstrap import bootstrap_calibration_uncertainty
from ssbc.simulation import BinaryClassifierSimulator


@pytest.fixture
def test_data():
    """Generate test data."""
    sim = BinaryClassifierSimulator(p_class1=0.3, beta_params_class0=(2, 5), beta_params_class1=(6, 2), seed=42)
    labels, probs = sim.generate(50)
    return labels, probs, sim


class TestBootstrapCalibrationUncertainty:
    """Test bootstrap_calibration_uncertainty function."""

    def test_basic_bootstrap(self, test_data):
        """Test basic bootstrap functionality."""
        labels, probs, sim = test_data

        results = bootstrap_calibration_uncertainty(
            labels=labels,
            probs=probs,
            simulator=sim,
            alpha_target=0.10,
            delta=0.10,
            test_size=100,
            n_bootstrap=10,  # Small for speed
            n_jobs=1,
            seed=42,
        )

        # Check structure
        assert "n_bootstrap" in results
        assert "n_calibration" in results
        assert "test_size" in results
        assert "marginal" in results
        assert "class_0" in results
        assert "class_1" in results

        # Check values
        assert results["n_bootstrap"] == 10
        assert results["n_calibration"] == len(labels)
        assert results["test_size"] == 100

    def test_marginal_structure(self, test_data):
        """Test marginal results structure."""
        labels, probs, sim = test_data

        results = bootstrap_calibration_uncertainty(
            labels=labels, probs=probs, simulator=sim, test_size=100, n_bootstrap=10, n_jobs=1, seed=42
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

            # Check types
            assert isinstance(m["samples"], np.ndarray)
            assert len(m["samples"]) == 10
            assert isinstance(m["mean"], float | np.floating)
            assert isinstance(m["std"], float | np.floating)

            # Check quantiles
            q = m["quantiles"]
            assert "q05" in q
            assert "q25" in q
            assert "q50" in q
            assert "q75" in q
            assert "q95" in q

    def test_per_class_structure(self, test_data):
        """Test per-class results structure."""
        labels, probs, sim = test_data

        results = bootstrap_calibration_uncertainty(
            labels=labels, probs=probs, simulator=sim, test_size=100, n_bootstrap=10, n_jobs=1, seed=42
        )

        for class_label in [0, 1]:
            class_results = results[f"class_{class_label}"]

            for metric in ["singleton", "doublet", "abstention", "singleton_error"]:
                assert metric in class_results
                m = class_results[metric]

                assert "samples" in m
                assert "mean" in m
                assert "std" in m
                assert "quantiles" in m

    def test_quantiles_ordering(self, test_data):
        """Test quantiles are properly ordered."""
        labels, probs, sim = test_data

        results = bootstrap_calibration_uncertainty(
            labels=labels, probs=probs, simulator=sim, test_size=100, n_bootstrap=20, n_jobs=1, seed=42
        )

        marginal = results["marginal"]

        for metric in ["singleton", "doublet", "abstention"]:
            q = marginal[metric]["quantiles"]

            # Check ordering (allowing for numerical errors)
            assert q["q05"] <= q["q25"] + 1e-10
            assert q["q25"] <= q["q50"] + 1e-10
            assert q["q50"] <= q["q75"] + 1e-10
            assert q["q75"] <= q["q95"] + 1e-10

    def test_seed_reproducibility(self, test_data):
        """Test that seed produces reproducible results."""
        labels, probs, sim = test_data

        results1 = bootstrap_calibration_uncertainty(
            labels=labels, probs=probs, simulator=sim, test_size=100, n_bootstrap=10, n_jobs=1, seed=42
        )

        results2 = bootstrap_calibration_uncertainty(
            labels=labels, probs=probs, simulator=sim, test_size=100, n_bootstrap=10, n_jobs=1, seed=42
        )

        # Results should be identical
        np.testing.assert_array_equal(
            results1["marginal"]["singleton"]["samples"], results2["marginal"]["singleton"]["samples"]
        )

    def test_parallel_execution(self, test_data):
        """Test parallel vs serial execution."""
        labels, probs, sim = test_data

        # Serial
        results_serial = bootstrap_calibration_uncertainty(
            labels=labels, probs=probs, simulator=sim, test_size=100, n_bootstrap=10, n_jobs=1, seed=42
        )

        # Parallel (using 2 jobs for testing)
        results_parallel = bootstrap_calibration_uncertainty(
            labels=labels, probs=probs, simulator=sim, test_size=100, n_bootstrap=10, n_jobs=2, seed=42
        )

        # Results should be very similar (parallel may have minor numerical differences)
        np.testing.assert_allclose(
            results_serial["marginal"]["singleton"]["mean"],
            results_parallel["marginal"]["singleton"]["mean"],
            rtol=0.1,  # Allow 10% relative tolerance for parallel differences
        )

    def test_rates_sum_to_one(self, test_data):
        """Test that marginal rates approximately sum to 1."""
        labels, probs, sim = test_data

        results = bootstrap_calibration_uncertainty(
            labels=labels, probs=probs, simulator=sim, test_size=100, n_bootstrap=10, n_jobs=1, seed=42
        )

        marginal = results["marginal"]

        # Check each bootstrap sample
        for i in range(10):
            rate_sum = (
                marginal["singleton"]["samples"][i]
                + marginal["doublet"]["samples"][i]
                + marginal["abstention"]["samples"][i]
            )

            # Allow small numerical errors
            assert abs(rate_sum - 1.0) < 1e-6

    def test_different_parameters(self, test_data):
        """Test with different alpha and delta."""
        labels, probs, sim = test_data

        results = bootstrap_calibration_uncertainty(
            labels=labels,
            probs=probs,
            simulator=sim,
            alpha_target=0.05,  # Different alpha
            delta=0.05,  # Different delta
            test_size=100,
            n_bootstrap=10,
            n_jobs=1,
            seed=42,
        )

        # Should still have valid structure
        assert "marginal" in results
        assert results["marginal"]["singleton"]["mean"] >= 0


class TestPlotBootstrapDistributions:
    """Test plot_bootstrap_distributions function."""

    def test_plotting_with_matplotlib(self, test_data):
        """Test plotting when matplotlib is available."""
        labels, probs, sim = test_data

        results = bootstrap_calibration_uncertainty(
            labels=labels, probs=probs, simulator=sim, test_size=100, n_bootstrap=10, n_jobs=1, seed=42
        )

        # Import here to check if matplotlib is available
        try:
            # Should not crash (just don't show the plot)
            import matplotlib

            from ssbc.bootstrap import plot_bootstrap_distributions

            matplotlib.use("Agg")  # Non-interactive backend
            import matplotlib.pyplot as plt

            plot_bootstrap_distributions(results, save_path=None)
            plt.close("all")  # Clean up

        except ImportError:
            pytest.skip("matplotlib not available")


class TestEdgeCases:
    """Test edge cases."""

    def test_small_bootstrap_samples(self, test_data):
        """Test with very small number of bootstrap samples."""
        labels, probs, sim = test_data

        results = bootstrap_calibration_uncertainty(
            labels=labels, probs=probs, simulator=sim, test_size=100, n_bootstrap=2, n_jobs=1, seed=42
        )

        # Should still work
        assert results["n_bootstrap"] == 2
        assert len(results["marginal"]["singleton"]["samples"]) == 2

    def test_small_calibration_size(self):
        """Test with very small calibration size."""
        sim = BinaryClassifierSimulator(p_class1=0.5, seed=42)
        labels, probs = sim.generate(15)  # Very small

        results = bootstrap_calibration_uncertainty(
            labels=labels, probs=probs, simulator=sim, test_size=50, n_bootstrap=5, n_jobs=1, seed=42
        )

        # Should handle gracefully
        assert results["n_calibration"] == 15
        assert "marginal" in results

    def test_different_test_size(self, test_data):
        """Test with different test sizes."""
        labels, probs, sim = test_data

        results_small = bootstrap_calibration_uncertainty(
            labels=labels, probs=probs, simulator=sim, test_size=50, n_bootstrap=10, n_jobs=1, seed=42
        )

        results_large = bootstrap_calibration_uncertainty(
            labels=labels, probs=probs, simulator=sim, test_size=200, n_bootstrap=10, n_jobs=1, seed=42
        )

        # Both should work
        assert results_small["test_size"] == 50
        assert results_large["test_size"] == 200

    def test_nan_handling(self, test_data):
        """Test handling of NaN values."""
        labels, probs, sim = test_data

        results = bootstrap_calibration_uncertainty(
            labels=labels, probs=probs, simulator=sim, test_size=100, n_bootstrap=10, n_jobs=1, seed=42
        )

        # singleton_error may contain NaN
        # Mean and quantiles should handle this correctly
        singleton_error = results["marginal"]["singleton_error"]

        # Should be a valid number or NaN
        assert isinstance(singleton_error["mean"], float | np.floating)

        # Quantiles should all be numbers or NaN
        for q_val in singleton_error["quantiles"].values():
            assert isinstance(q_val, float | np.floating)
