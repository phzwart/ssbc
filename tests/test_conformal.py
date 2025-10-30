"""Tests for the conformal prediction module."""

import numpy as np
import pandas as pd
import pytest

from ssbc.calibration import (
    alpha_scan,
    compute_pac_operational_metrics,
    mondrian_conformal_calibrate,
    split_by_class,
)
from ssbc.simulation import BinaryClassifierSimulator


class TestSplitByClass:
    """Test split_by_class function."""

    def test_basic_split(self):
        """Test basic splitting by class."""
        labels = np.array([0, 1, 0, 1, 0])
        probs = np.array([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1], [0.2, 0.8], [0.7, 0.3]])

        class_data = split_by_class(labels, probs)

        assert 0 in class_data
        assert 1 in class_data

        # Class 0: indices 0, 2, 4
        assert class_data[0]["n"] == 3
        np.testing.assert_array_equal(class_data[0]["indices"], [0, 2, 4])
        np.testing.assert_array_equal(class_data[0]["labels"], [0, 0, 0])

        # Class 1: indices 1, 3
        assert class_data[1]["n"] == 2
        np.testing.assert_array_equal(class_data[1]["indices"], [1, 3])
        np.testing.assert_array_equal(class_data[1]["labels"], [1, 1])

    def test_split_preserves_probs(self):
        """Test that splitting preserves probabilities."""
        labels = np.array([0, 1, 0, 1])
        probs = np.array([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1], [0.2, 0.8]])

        class_data = split_by_class(labels, probs)

        # Check class 0 probs
        expected_class0_probs = probs[[0, 2]]
        np.testing.assert_array_equal(class_data[0]["probs"], expected_class0_probs)

        # Check class 1 probs
        expected_class1_probs = probs[[1, 3]]
        np.testing.assert_array_equal(class_data[1]["probs"], expected_class1_probs)

    def test_all_class_zero(self):
        """Test with all samples in class 0."""
        labels = np.zeros(5, dtype=int)
        probs = np.random.rand(5, 2)
        probs = probs / probs.sum(axis=1, keepdims=True)

        class_data = split_by_class(labels, probs)

        assert class_data[0]["n"] == 5
        assert class_data[1]["n"] == 0
        assert len(class_data[1]["indices"]) == 0

    def test_all_class_one(self):
        """Test with all samples in class 1."""
        labels = np.ones(5, dtype=int)
        probs = np.random.rand(5, 2)
        probs = probs / probs.sum(axis=1, keepdims=True)

        class_data = split_by_class(labels, probs)

        assert class_data[0]["n"] == 0
        assert class_data[1]["n"] == 5
        assert len(class_data[0]["indices"]) == 0

    def test_indices_cover_all_samples(self):
        """Test that indices cover all samples exactly once."""
        labels = np.array([0, 1, 0, 1, 0, 1])
        probs = np.random.rand(6, 2)
        probs = probs / probs.sum(axis=1, keepdims=True)

        class_data = split_by_class(labels, probs)

        all_indices = np.concatenate([class_data[0]["indices"], class_data[1]["indices"]])

        assert len(all_indices) == 6
        assert set(all_indices) == set(range(6))

    def test_single_sample_each_class(self):
        """Test with one sample per class."""
        labels = np.array([0, 1])
        probs = np.array([[0.8, 0.2], [0.3, 0.7]])

        class_data = split_by_class(labels, probs)

        assert class_data[0]["n"] == 1
        assert class_data[1]["n"] == 1


class TestMondrianConformalCalibrate:
    """Test mondrian_conformal_calibrate function."""

    @pytest.fixture
    def simple_data(self):
        """Create simple test data."""
        sim = BinaryClassifierSimulator(p_class1=0.5, beta_params_class0=(2, 8), beta_params_class1=(8, 2), seed=42)
        labels, probs = sim.generate(n_samples=100)
        class_data = split_by_class(labels, probs)
        return class_data

    def test_basic_calibration(self, simple_data):
        """Test basic Mondrian calibration."""
        cal_result, pred_stats = mondrian_conformal_calibrate(
            class_data=simple_data, alpha_target=0.10, delta=0.10, mode="beta"
        )

        # Check calibration results
        assert 0 in cal_result
        assert 1 in cal_result

        for label in [0, 1]:
            assert "alpha_corrected" in cal_result[label]
            assert "threshold" in cal_result[label]
            assert "ssbc_result" in cal_result[label]

    def test_scalar_alpha_delta(self, simple_data):
        """Test with scalar alpha and delta."""
        cal_result, pred_stats = mondrian_conformal_calibrate(
            class_data=simple_data,
            alpha_target=0.10,  # scalar
            delta=0.10,  # scalar
            mode="beta",
        )

        # Should apply same values to both classes
        assert cal_result[0]["alpha_target"] == 0.10
        assert cal_result[1]["alpha_target"] == 0.10

    def test_dict_alpha_delta(self, simple_data):
        """Test with dict alpha and delta."""
        cal_result, pred_stats = mondrian_conformal_calibrate(
            class_data=simple_data, alpha_target={0: 0.05, 1: 0.15}, delta={0: 0.05, 1: 0.15}, mode="beta"
        )

        # Should use per-class values
        assert cal_result[0]["alpha_target"] == 0.05
        assert cal_result[1]["alpha_target"] == 0.15

    def test_prediction_stats_structure(self, simple_data):
        """Test that prediction stats have expected structure."""
        _, pred_stats = mondrian_conformal_calibrate(class_data=simple_data, alpha_target=0.10, delta=0.10, mode="beta")

        # Per-class stats
        for label in [0, 1]:
            assert label in pred_stats
            stats = pred_stats[label]

            if "error" not in stats:
                assert "abstentions" in stats
                assert "singletons" in stats
                assert "doublets" in stats
                assert "pac_bounds" in stats

        # Marginal stats
        assert "marginal" in pred_stats
        marginal = pred_stats["marginal"]
        assert "coverage" in marginal
        assert "singletons" in marginal
        assert "doublets" in marginal
        assert "abstentions" in marginal

    def test_coverage_guarantee(self, simple_data):
        """Test that coverage meets target (probabilistically)."""
        _, pred_stats = mondrian_conformal_calibrate(class_data=simple_data, alpha_target=0.10, delta=0.10, mode="beta")

        marginal = pred_stats["marginal"]
        coverage_rate = marginal["coverage"]["rate"]

        # Should have coverage >= 1 - alpha_target (at least probabilistically)
        # With delta=0.10, we expect this to hold 90% of the time
        assert coverage_rate >= 0.85  # Slightly below 0.90 to account for randomness

    def test_thresholds_are_valid(self, simple_data):
        """Test that thresholds are in valid range."""
        cal_result, _ = mondrian_conformal_calibrate(class_data=simple_data, alpha_target=0.10, delta=0.10, mode="beta")

        for label in [0, 1]:
            threshold = cal_result[label]["threshold"]
            assert 0 <= threshold <= 1

    def test_prediction_set_counts(self, simple_data):
        """Test that prediction set counts sum to n_total."""
        _, pred_stats = mondrian_conformal_calibrate(class_data=simple_data, alpha_target=0.10, delta=0.10, mode="beta")

        marginal = pred_stats["marginal"]
        n_total = marginal["n_total"]

        n_abst = marginal["abstentions"]["count"]
        n_sing = marginal["singletons"]["count"]
        n_doub = marginal["doublets"]["count"]

        assert n_abst + n_sing + n_doub == n_total

    def test_singleton_breakdown(self, simple_data):
        """Test singleton breakdown by predicted class."""
        _, pred_stats = mondrian_conformal_calibrate(class_data=simple_data, alpha_target=0.10, delta=0.10, mode="beta")

        marginal = pred_stats["marginal"]
        singletons = marginal["singletons"]

        n_sing_total = singletons["count"]
        n_pred_0 = singletons["pred_0"]
        n_pred_1 = singletons["pred_1"]

        # Singleton counts should sum correctly
        assert n_pred_0 + n_pred_1 == n_sing_total

    def test_empty_class_handling(self):
        """Test handling of empty class."""
        # Create data with no class 1 samples
        labels = np.zeros(50, dtype=int)
        probs = np.random.rand(50, 2)
        probs = probs / probs.sum(axis=1, keepdims=True)

        class_data = split_by_class(labels, probs)

        cal_result, pred_stats = mondrian_conformal_calibrate(
            class_data=class_data, alpha_target=0.10, delta=0.10, mode="beta"
        )

        # Class 1 should have error message
        assert "error" in cal_result[1]
        assert "No calibration samples" in cal_result[1]["error"]

    def test_beta_binomial_mode(self, simple_data):
        """Test beta-binomial mode."""
        cal_result, pred_stats = mondrian_conformal_calibrate(
            class_data=simple_data, alpha_target=0.10, delta=0.10, mode="beta-binomial", m=200
        )

        # Check that mode is set correctly
        for label in [0, 1]:
            if "ssbc_result" in cal_result[label]:
                assert cal_result[label]["ssbc_result"].mode == "beta-binomial"
                assert cal_result[label]["ssbc_result"].details["m"] == 200

    def test_per_class_statistics(self, simple_data):
        """Test per-class statistics."""
        _, pred_stats = mondrian_conformal_calibrate(class_data=simple_data, alpha_target=0.10, delta=0.10, mode="beta")

        for label in [0, 1]:
            if "error" not in pred_stats[label]:
                stats = pred_stats[label]

                # Check that all prediction sets are accounted for
                n_class = stats["n_class"]
                n_abst = stats["abstentions"]["count"]
                n_sing = stats["singletons"]["count"]
                n_doub = stats["doublets"]["count"]

                assert n_abst + n_sing + n_doub == n_class

    def test_pac_bounds_computation(self, simple_data):
        """Test PAC bounds computation."""
        _, pred_stats = mondrian_conformal_calibrate(class_data=simple_data, alpha_target=0.10, delta=0.10, mode="beta")

        for label in [0, 1]:
            if "error" not in pred_stats[label]:
                pac = pred_stats[label]["pac_bounds"]

                # Check that PAC metrics are computed when applicable
                if pac["rho"] is not None:
                    assert pac["rho"] > 0
                    assert 0 <= pac["kappa"] <= 1

    def test_different_alphas_per_class(self, simple_data):
        """Test with different alpha values per class."""
        cal_result, _ = mondrian_conformal_calibrate(
            class_data=simple_data, alpha_target={0: 0.05, 1: 0.20}, delta=0.10, mode="beta"
        )

        # Class 0 should be more conservative (lower alpha)
        # This typically means higher threshold for nonconformity scores
        alpha_0 = cal_result[0]["alpha_corrected"]
        alpha_1 = cal_result[1]["alpha_corrected"]

        assert alpha_0 < alpha_1

    def test_marginal_coverage_dict(self, simple_data):
        """Test marginal coverage dictionary structure."""
        _, pred_stats = mondrian_conformal_calibrate(class_data=simple_data, alpha_target=0.10, delta=0.10, mode="beta")

        coverage = pred_stats["marginal"]["coverage"]

        assert "count" in coverage
        assert "rate" in coverage
        assert isinstance(coverage["ci_95"], dict)

    def test_prediction_sets_are_lists(self, simple_data):
        """Test that prediction sets are stored correctly."""
        _, pred_stats = mondrian_conformal_calibrate(class_data=simple_data, alpha_target=0.10, delta=0.10, mode="beta")

        # Check per-class prediction sets
        for label in [0, 1]:
            if "prediction_sets" in pred_stats[label]:
                pred_sets = pred_stats[label]["prediction_sets"]
                assert isinstance(pred_sets, list)

                # Each prediction set should be a list
                for ps in pred_sets:
                    assert isinstance(ps, list)
                    # Each set should contain only 0, 1, or both
                    assert all(val in [0, 1] for val in ps)

        # Check marginal prediction sets
        if "prediction_sets" in pred_stats["marginal"]:
            marginal_sets = pred_stats["marginal"]["prediction_sets"]
            assert isinstance(marginal_sets, list)

    def test_reproducibility(self, simple_data):
        """Test that results are reproducible."""
        cal1, pred1 = mondrian_conformal_calibrate(class_data=simple_data, alpha_target=0.10, delta=0.10, mode="beta")

        cal2, pred2 = mondrian_conformal_calibrate(class_data=simple_data, alpha_target=0.10, delta=0.10, mode="beta")

        # Results should be identical
        for label in [0, 1]:
            if "alpha_corrected" in cal1[label]:
                assert cal1[label]["alpha_corrected"] == cal2[label]["alpha_corrected"]
                assert cal1[label]["threshold"] == cal2[label]["threshold"]


class TestAlphaScan:
    """Test alpha_scan function."""

    @pytest.fixture
    def simple_calibration_data(self):
        """Create simple calibration data for testing."""
        sim = BinaryClassifierSimulator(p_class1=0.5, beta_params_class0=(2, 8), beta_params_class1=(8, 2), seed=42)
        labels, probs = sim.generate(n_samples=50)
        return labels, probs

    def test_returns_dataframe(self, simple_calibration_data):
        """Test that alpha_scan returns a DataFrame."""
        labels, probs = simple_calibration_data
        result = alpha_scan(labels, probs)
        # When no fixed_threshold is provided, result should be a DataFrame
        assert isinstance(result, pd.DataFrame)

    def test_dataframe_columns(self, simple_calibration_data):
        """Test that DataFrame has expected columns."""
        labels, probs = simple_calibration_data
        result = alpha_scan(labels, probs)
        # When no fixed_threshold is provided, result should be a DataFrame
        assert isinstance(result, pd.DataFrame)
        df = result

        expected_columns = [
            "alpha",
            "qhat_0",
            "qhat_1",
            "n_abstentions",
            "n_singletons",
            "n_doublets",
            "n_singletons_correct",
            "singleton_coverage",
            "n_singletons_0",
            "n_singletons_correct_0",
            "singleton_coverage_0",
            "n_singletons_1",
            "n_singletons_correct_1",
            "singleton_coverage_1",
        ]
        assert list(df.columns) == expected_columns

    def test_fixed_threshold_included(self, simple_calibration_data):
        """Test that fixed threshold is returned as dict."""
        labels, probs = simple_calibration_data
        result = alpha_scan(labels, probs, fixed_threshold=0.5)

        # Should return a tuple
        assert isinstance(result, tuple)
        assert len(result) == 2

        df, fixed = result
        assert isinstance(df, pd.DataFrame)
        assert isinstance(fixed, dict)

        # Check that the fixed threshold is 0.5 for both classes
        assert fixed["qhat_0"] == 0.5
        assert fixed["qhat_1"] == 0.5

        # Check that fixed dict has all expected keys
        expected_keys = [
            "alpha",
            "qhat_0",
            "qhat_1",
            "n_abstentions",
            "n_singletons",
            "n_doublets",
            "n_singletons_correct",
            "singleton_coverage",
            "n_singletons_0",
            "n_singletons_correct_0",
            "singleton_coverage_0",
            "n_singletons_1",
            "n_singletons_correct_1",
            "singleton_coverage_1",
        ]
        assert set(fixed.keys()) == set(expected_keys)

    def test_counts_sum_to_total(self, simple_calibration_data):
        """Test that prediction set counts sum to total number of samples."""
        labels, probs = simple_calibration_data
        result = alpha_scan(labels, probs)
        # When no fixed_threshold is provided, result should be a DataFrame
        assert isinstance(result, pd.DataFrame)
        df = result

        n_total = len(labels)

        for _, row in df.iterrows():
            total_count = row["n_abstentions"] + row["n_singletons"] + row["n_doublets"]
            assert total_count == n_total

    def test_alpha_values_in_valid_range(self, simple_calibration_data):
        """Test that alpha values are in [0, 1]."""
        labels, probs = simple_calibration_data
        result = alpha_scan(labels, probs)
        # When no fixed_threshold is provided, result should be a DataFrame
        assert isinstance(result, pd.DataFrame)
        df = result

        assert (df["alpha"] >= 0).all()
        assert (df["alpha"] <= 1).all()

    def test_threshold_values_in_valid_range(self, simple_calibration_data):
        """Test that threshold values are in [0, 1]."""
        labels, probs = simple_calibration_data
        result = alpha_scan(labels, probs)
        # When no fixed_threshold is provided, result should be a DataFrame
        assert isinstance(result, pd.DataFrame)
        df = result

        assert (df["qhat_0"] >= 0).all()
        assert (df["qhat_0"] <= 1).all()
        assert (df["qhat_1"] >= 0).all()
        assert (df["qhat_1"] <= 1).all()

    def test_counts_are_non_negative(self, simple_calibration_data):
        """Test that all counts are non-negative."""
        labels, probs = simple_calibration_data
        result = alpha_scan(labels, probs)
        # When no fixed_threshold is provided, result should be a DataFrame
        assert isinstance(result, pd.DataFrame)
        df = result

        assert (df["n_abstentions"] >= 0).all()
        assert (df["n_singletons"] >= 0).all()
        assert (df["n_doublets"] >= 0).all()

    def test_multiple_alpha_values(self, simple_calibration_data):
        """Test that multiple alpha values are scanned."""
        labels, probs = simple_calibration_data
        result = alpha_scan(labels, probs)
        # When no fixed_threshold is provided, result should be a DataFrame
        assert isinstance(result, pd.DataFrame)
        df = result

        # Should have multiple rows (one for each possible alpha + fixed)
        assert len(df) > 2

        # Should have multiple unique alpha values
        unique_alphas = df["alpha"].nunique()
        assert unique_alphas > 2

    def test_custom_fixed_threshold(self, simple_calibration_data):
        """Test with custom fixed threshold."""
        labels, probs = simple_calibration_data
        fixed_threshold = 0.3
        df, fixed = alpha_scan(labels, probs, fixed_threshold=fixed_threshold)

        assert fixed["qhat_0"] == fixed_threshold
        assert fixed["qhat_1"] == fixed_threshold

    def test_no_fixed_threshold(self, simple_calibration_data):
        """Test that no fixed threshold returns just DataFrame."""
        labels, probs = simple_calibration_data
        result = alpha_scan(labels, probs)

        # Should return just a DataFrame, not a tuple
        assert isinstance(result, pd.DataFrame)
        assert not isinstance(result, tuple)

    def test_small_dataset(self):
        """Test with a small dataset."""
        labels = np.array([0, 1, 0, 1])
        probs = np.array([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1], [0.2, 0.8]])

        result = alpha_scan(labels, probs)
        # When no fixed_threshold is provided, result should be a DataFrame
        assert isinstance(result, pd.DataFrame)
        df = result

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

        # All counts should sum to 4
        for _, row in df.iterrows():
            total = row["n_abstentions"] + row["n_singletons"] + row["n_doublets"]
            assert total == 4

    def test_all_same_class(self):
        """Test with all samples from the same class."""
        labels = np.zeros(10, dtype=int)
        probs = np.random.rand(10, 2)
        probs = probs / probs.sum(axis=1, keepdims=True)

        result = alpha_scan(labels, probs)
        # When no fixed_threshold is provided, result should be a DataFrame
        assert isinstance(result, pd.DataFrame)
        df = result

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_extreme_probabilities(self):
        """Test with extreme probability values."""
        labels = np.array([0, 1, 0, 1])
        # Very confident predictions
        probs = np.array([[0.99, 0.01], [0.01, 0.99], [0.95, 0.05], [0.05, 0.95]])

        result = alpha_scan(labels, probs)
        # When no fixed_threshold is provided, result should be a DataFrame
        assert isinstance(result, pd.DataFrame)
        df = result

        assert isinstance(df, pd.DataFrame)
        # With confident predictions, we should see more singletons at some alpha values
        max_singletons = df["n_singletons"].max()
        assert max_singletons > 0

    def test_reproducibility(self, simple_calibration_data):
        """Test that results are reproducible."""
        labels, probs = simple_calibration_data
        df1 = alpha_scan(labels, probs)
        df2 = alpha_scan(labels, probs)

        pd.testing.assert_frame_equal(df1, df2)

    def test_alpha_ordering(self, simple_calibration_data):
        """Test that DataFrame can be sorted by alpha."""
        labels, probs = simple_calibration_data
        result = alpha_scan(labels, probs)
        # When no fixed_threshold is provided, result should be a DataFrame
        assert isinstance(result, pd.DataFrame)
        df = result

        # Sort by alpha
        df_sorted = df.sort_values("alpha")

        # Should not raise any errors
        assert len(df_sorted) == len(df)

    def test_fixed_counts_sum_to_total(self, simple_calibration_data):
        """Test that fixed threshold counts sum to total."""
        labels, probs = simple_calibration_data
        df, fixed = alpha_scan(labels, probs, fixed_threshold=0.5)

        n_total = len(labels)
        total_count = fixed["n_abstentions"] + fixed["n_singletons"] + fixed["n_doublets"]
        assert total_count == n_total

    def test_singleton_coverage_values(self, simple_calibration_data):
        """Test that singleton coverage values are in valid range."""
        labels, probs = simple_calibration_data
        result = alpha_scan(labels, probs)
        # When no fixed_threshold is provided, result should be a DataFrame
        assert isinstance(result, pd.DataFrame)
        df = result

        # All coverage values should be between 0 and 1
        assert (df["singleton_coverage"] >= 0).all()
        assert (df["singleton_coverage"] <= 1).all()
        assert (df["singleton_coverage_0"] >= 0).all()
        assert (df["singleton_coverage_0"] <= 1).all()
        assert (df["singleton_coverage_1"] >= 0).all()
        assert (df["singleton_coverage_1"] <= 1).all()

    def test_singleton_coverage_consistency(self, simple_calibration_data):
        """Test that singleton coverage is consistent with counts."""
        labels, probs = simple_calibration_data
        result = alpha_scan(labels, probs)
        # When no fixed_threshold is provided, result should be a DataFrame
        assert isinstance(result, pd.DataFrame)
        df = result

        for _, row in df.iterrows():
            # Check marginal coverage calculation
            if row["n_singletons"] > 0:
                expected_coverage = row["n_singletons_correct"] / row["n_singletons"]
                assert abs(row["singleton_coverage"] - expected_coverage) < 1e-10
            else:
                assert row["singleton_coverage"] == 0.0

            # Check class 0 coverage
            if row["n_singletons_0"] > 0:
                expected_coverage_0 = row["n_singletons_correct_0"] / row["n_singletons_0"]
                assert abs(row["singleton_coverage_0"] - expected_coverage_0) < 1e-10
            else:
                assert row["singleton_coverage_0"] == 0.0

            # Check class 1 coverage
            if row["n_singletons_1"] > 0:
                expected_coverage_1 = row["n_singletons_correct_1"] / row["n_singletons_1"]
                assert abs(row["singleton_coverage_1"] - expected_coverage_1) < 1e-10
            else:
                assert row["singleton_coverage_1"] == 0.0

    def test_singleton_counts_consistency(self, simple_calibration_data):
        """Test that per-class singleton counts sum to total."""
        labels, probs = simple_calibration_data
        result = alpha_scan(labels, probs)
        # When no fixed_threshold is provided, result should be a DataFrame
        assert isinstance(result, pd.DataFrame)
        df = result

        for _, row in df.iterrows():
            # Per-class singletons should sum to total singletons
            assert row["n_singletons_0"] + row["n_singletons_1"] == row["n_singletons"]
            # Per-class correct singletons should sum to total correct singletons
            assert row["n_singletons_correct_0"] + row["n_singletons_correct_1"] == row["n_singletons_correct"]

    def test_correct_singletons_bounded(self, simple_calibration_data):
        """Test that correct singletons don't exceed total singletons."""
        labels, probs = simple_calibration_data
        result = alpha_scan(labels, probs)
        # When no fixed_threshold is provided, result should be a DataFrame
        assert isinstance(result, pd.DataFrame)
        df = result

        assert (df["n_singletons_correct"] <= df["n_singletons"]).all()
        assert (df["n_singletons_correct_0"] <= df["n_singletons_0"]).all()
        assert (df["n_singletons_correct_1"] <= df["n_singletons_1"]).all()

    def test_high_coverage_with_confident_predictions(self):
        """Test that confident predictions lead to high singleton coverage."""
        labels = np.array([0, 1, 0, 1, 0, 1] * 10)
        # Very confident and correct predictions
        probs = np.array([[0.95, 0.05], [0.05, 0.95], [0.98, 0.02], [0.02, 0.98], [0.97, 0.03], [0.03, 0.97]] * 10)

        result = alpha_scan(labels, probs)
        # When no fixed_threshold is provided, result should be a DataFrame
        assert isinstance(result, pd.DataFrame)
        df = result

        # With very confident correct predictions, we should see high coverage
        # at some alpha values
        max_coverage = df["singleton_coverage"].max()
        assert max_coverage > 0.9  # Should have very high coverage somewhere


class TestComputePACOperationalMetrics:
    """Test compute_pac_operational_metrics function."""

    @pytest.fixture
    def calibration_data(self):
        """Generate calibration data for testing."""
        sim = BinaryClassifierSimulator(p_class1=0.5, beta_params_class0=(2, 7), beta_params_class1=(7, 2), seed=42)
        labels, probs = sim.generate(n_samples=100)
        return labels, probs

    def test_basic_execution(self, calibration_data):
        """Test that function executes without error."""
        labels, probs = calibration_data
        result = compute_pac_operational_metrics(
            y_cal=labels, probs_cal=probs, alpha=0.1, delta=0.1, ci_level=0.95, class_label=1
        )

        # Check that result is a dict with expected keys
        assert isinstance(result, dict)
        expected_keys = [
            "alpha_adj",
            "singleton_rate_ci",
            "doublet_rate_ci",
            "abstention_rate_ci",
            "expected_singleton_rate",
            "expected_doublet_rate",
            "expected_abstention_rate",
            "alpha_grid",
            "singleton_fractions",
            "doublet_fractions",
            "abstention_fractions",
            "beta_weights",
            "n_calibration",
        ]
        for key in expected_keys:
            assert key in result

    def test_output_types(self, calibration_data):
        """Test that output has correct types."""
        labels, probs = calibration_data
        result = compute_pac_operational_metrics(
            y_cal=labels, probs_cal=probs, alpha=0.1, delta=0.1, ci_level=0.95, class_label=1
        )

        assert isinstance(result["alpha_adj"], float)
        assert isinstance(result["singleton_rate_ci"], list)
        assert len(result["singleton_rate_ci"]) == 2
        assert isinstance(result["expected_singleton_rate"], float)
        assert isinstance(result["alpha_grid"], list)
        assert isinstance(result["beta_weights"], list)
        assert isinstance(result["n_calibration"], int | np.integer)

    def test_bounds_validity(self, calibration_data):
        """Test that bounds are valid (lower <= upper)."""
        labels, probs = calibration_data
        result = compute_pac_operational_metrics(
            y_cal=labels, probs_cal=probs, alpha=0.1, delta=0.1, ci_level=0.95, class_label=1
        )

        # Check singleton bounds
        assert result["singleton_rate_ci"][0] <= result["singleton_rate_ci"][1]
        # Check abstention bounds
        assert result["abstention_rate_ci"][0] <= result["abstention_rate_ci"][1]

    def test_rates_in_valid_range(self, calibration_data):
        """Test that all rates are in [0, 1]."""
        labels, probs = calibration_data
        result = compute_pac_operational_metrics(
            y_cal=labels, probs_cal=probs, alpha=0.1, delta=0.1, ci_level=0.95, class_label=1
        )

        # Check expected rates
        assert 0 <= result["expected_singleton_rate"] <= 1
        assert 0 <= result["expected_abstention_rate"] <= 1
        assert 0 <= result["expected_doublet_rate"] <= 1

        # Check bounds
        assert 0 <= result["singleton_rate_ci"][0] <= 1
        assert 0 <= result["singleton_rate_ci"][1] <= 1
        assert 0 <= result["abstention_rate_ci"][0] <= 1
        assert 0 <= result["abstention_rate_ci"][1] <= 1

    def test_rates_sum_to_one(self, calibration_data):
        """Test that singleton + doublet + abstention ≈ 1."""
        labels, probs = calibration_data
        result = compute_pac_operational_metrics(
            y_cal=labels, probs_cal=probs, alpha=0.1, delta=0.1, ci_level=0.95, class_label=1
        )

        rate_sum = (
            result["expected_singleton_rate"] + result["expected_doublet_rate"] + result["expected_abstention_rate"]
        )
        assert abs(rate_sum - 1.0) < 1e-10

    def test_beta_weights_sum_to_one(self, calibration_data):
        """Test that Beta weights sum to 1."""
        labels, probs = calibration_data
        result = compute_pac_operational_metrics(
            y_cal=labels, probs_cal=probs, alpha=0.1, delta=0.1, ci_level=0.95, class_label=1
        )

        weight_sum = sum(result["beta_weights"])
        assert abs(weight_sum - 1.0) < 1e-10

    def test_alpha_grid_size(self, calibration_data):
        """Test that alpha grid has correct size."""
        labels, probs = calibration_data
        result = compute_pac_operational_metrics(
            y_cal=labels, probs_cal=probs, alpha=0.1, delta=0.1, ci_level=0.95, class_label=1
        )

        # Grid size should equal number of calibration points for class_label
        n_class1 = np.sum(labels == 1)
        assert len(result["alpha_grid"]) == n_class1
        assert len(result["singleton_fractions"]) == n_class1
        assert len(result["beta_weights"]) == n_class1

    def test_different_class_labels(self, calibration_data):
        """Test with different class labels."""
        labels, probs = calibration_data

        result_0 = compute_pac_operational_metrics(
            y_cal=labels, probs_cal=probs, alpha=0.1, delta=0.1, ci_level=0.95, class_label=0
        )

        result_1 = compute_pac_operational_metrics(
            y_cal=labels, probs_cal=probs, alpha=0.1, delta=0.1, ci_level=0.95, class_label=1
        )

        # Results should be different for different classes
        assert result_0["n_calibration"] != result_1["n_calibration"]

    def test_1d_probabilities(self, calibration_data):
        """Test with 1D probability array."""
        labels, probs = calibration_data

        # Use 1D array (P(class=1))
        probs_1d = probs[:, 1]

        result = compute_pac_operational_metrics(
            y_cal=labels, probs_cal=probs_1d, alpha=0.1, delta=0.1, ci_level=0.95, class_label=1
        )

        assert isinstance(result, dict)
        assert result["n_calibration"] > 0

    def test_invalid_inputs(self, calibration_data):
        """Test that invalid inputs raise errors."""
        labels, probs = calibration_data

        # Invalid alpha
        with pytest.raises(ValueError):
            compute_pac_operational_metrics(y_cal=labels, probs_cal=probs, alpha=1.5, delta=0.1, class_label=1)

        # Invalid delta
        with pytest.raises(ValueError):
            compute_pac_operational_metrics(y_cal=labels, probs_cal=probs, alpha=0.1, delta=-0.1, class_label=1)

        # Invalid class_label
        with pytest.raises(ValueError):
            compute_pac_operational_metrics(y_cal=labels, probs_cal=probs, alpha=0.1, delta=0.1, class_label=2)

    def test_small_sample_size(self):
        """Test with minimum required calibration set (n >= 10)."""
        labels = np.array([1] * 10)
        probs = np.array([[0.2, 0.8]] * 10)

        result = compute_pac_operational_metrics(
            y_cal=labels, probs_cal=probs, alpha=0.2, delta=0.2, ci_level=0.90, class_label=1
        )

        assert result["n_calibration"] == 10
        # alpha_grid length depends on n_calibration
        assert len(result["alpha_grid"]) == 10
