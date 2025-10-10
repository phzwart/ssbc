"""Tests for the conformal prediction module."""

import pytest
import numpy as np
from ssbc.conformal import split_by_class, mondrian_conformal_calibrate
from ssbc.simulation import BinaryClassifierSimulator


class TestSplitByClass:
    """Test split_by_class function."""
    
    def test_basic_split(self):
        """Test basic splitting by class."""
        labels = np.array([0, 1, 0, 1, 0])
        probs = np.array([
            [0.8, 0.2],
            [0.3, 0.7],
            [0.9, 0.1],
            [0.2, 0.8],
            [0.7, 0.3]
        ])
        
        class_data = split_by_class(labels, probs)
        
        assert 0 in class_data
        assert 1 in class_data
        
        # Class 0: indices 0, 2, 4
        assert class_data[0]['n'] == 3
        np.testing.assert_array_equal(class_data[0]['indices'], [0, 2, 4])
        np.testing.assert_array_equal(class_data[0]['labels'], [0, 0, 0])
        
        # Class 1: indices 1, 3
        assert class_data[1]['n'] == 2
        np.testing.assert_array_equal(class_data[1]['indices'], [1, 3])
        np.testing.assert_array_equal(class_data[1]['labels'], [1, 1])
    
    def test_split_preserves_probs(self):
        """Test that splitting preserves probabilities."""
        labels = np.array([0, 1, 0, 1])
        probs = np.array([
            [0.8, 0.2],
            [0.3, 0.7],
            [0.9, 0.1],
            [0.2, 0.8]
        ])
        
        class_data = split_by_class(labels, probs)
        
        # Check class 0 probs
        expected_class0_probs = probs[[0, 2]]
        np.testing.assert_array_equal(class_data[0]['probs'], expected_class0_probs)
        
        # Check class 1 probs
        expected_class1_probs = probs[[1, 3]]
        np.testing.assert_array_equal(class_data[1]['probs'], expected_class1_probs)
    
    def test_all_class_zero(self):
        """Test with all samples in class 0."""
        labels = np.zeros(5, dtype=int)
        probs = np.random.rand(5, 2)
        probs = probs / probs.sum(axis=1, keepdims=True)
        
        class_data = split_by_class(labels, probs)
        
        assert class_data[0]['n'] == 5
        assert class_data[1]['n'] == 0
        assert len(class_data[1]['indices']) == 0
    
    def test_all_class_one(self):
        """Test with all samples in class 1."""
        labels = np.ones(5, dtype=int)
        probs = np.random.rand(5, 2)
        probs = probs / probs.sum(axis=1, keepdims=True)
        
        class_data = split_by_class(labels, probs)
        
        assert class_data[0]['n'] == 0
        assert class_data[1]['n'] == 5
        assert len(class_data[0]['indices']) == 0
    
    def test_indices_cover_all_samples(self):
        """Test that indices cover all samples exactly once."""
        labels = np.array([0, 1, 0, 1, 0, 1])
        probs = np.random.rand(6, 2)
        probs = probs / probs.sum(axis=1, keepdims=True)
        
        class_data = split_by_class(labels, probs)
        
        all_indices = np.concatenate([
            class_data[0]['indices'],
            class_data[1]['indices']
        ])
        
        assert len(all_indices) == 6
        assert set(all_indices) == set(range(6))
    
    def test_single_sample_each_class(self):
        """Test with one sample per class."""
        labels = np.array([0, 1])
        probs = np.array([[0.8, 0.2], [0.3, 0.7]])
        
        class_data = split_by_class(labels, probs)
        
        assert class_data[0]['n'] == 1
        assert class_data[1]['n'] == 1


class TestMondrianConformalCalibrate:
    """Test mondrian_conformal_calibrate function."""
    
    @pytest.fixture
    def simple_data(self):
        """Create simple test data."""
        sim = BinaryClassifierSimulator(
            p_class1=0.5,
            beta_params_class0=(2, 8),
            beta_params_class1=(8, 2),
            seed=42
        )
        labels, probs = sim.generate(n_samples=100)
        class_data = split_by_class(labels, probs)
        return class_data
    
    def test_basic_calibration(self, simple_data):
        """Test basic Mondrian calibration."""
        cal_result, pred_stats = mondrian_conformal_calibrate(
            class_data=simple_data,
            alpha_target=0.10,
            delta=0.10,
            mode="beta"
        )
        
        # Check calibration results
        assert 0 in cal_result
        assert 1 in cal_result
        
        for label in [0, 1]:
            assert 'alpha_corrected' in cal_result[label]
            assert 'threshold' in cal_result[label]
            assert 'ssbc_result' in cal_result[label]
    
    def test_scalar_alpha_delta(self, simple_data):
        """Test with scalar alpha and delta."""
        cal_result, pred_stats = mondrian_conformal_calibrate(
            class_data=simple_data,
            alpha_target=0.10,  # scalar
            delta=0.10,         # scalar
            mode="beta"
        )
        
        # Should apply same values to both classes
        assert cal_result[0]['alpha_target'] == 0.10
        assert cal_result[1]['alpha_target'] == 0.10
    
    def test_dict_alpha_delta(self, simple_data):
        """Test with dict alpha and delta."""
        cal_result, pred_stats = mondrian_conformal_calibrate(
            class_data=simple_data,
            alpha_target={0: 0.05, 1: 0.15},
            delta={0: 0.05, 1: 0.15},
            mode="beta"
        )
        
        # Should use per-class values
        assert cal_result[0]['alpha_target'] == 0.05
        assert cal_result[1]['alpha_target'] == 0.15
    
    def test_prediction_stats_structure(self, simple_data):
        """Test that prediction stats have expected structure."""
        _, pred_stats = mondrian_conformal_calibrate(
            class_data=simple_data,
            alpha_target=0.10,
            delta=0.10,
            mode="beta"
        )
        
        # Per-class stats
        for label in [0, 1]:
            assert label in pred_stats
            stats = pred_stats[label]
            
            if 'error' not in stats:
                assert 'abstentions' in stats
                assert 'singletons' in stats
                assert 'doublets' in stats
                assert 'pac_bounds' in stats
        
        # Marginal stats
        assert 'marginal' in pred_stats
        marginal = pred_stats['marginal']
        assert 'coverage' in marginal
        assert 'singletons' in marginal
        assert 'doublets' in marginal
        assert 'abstentions' in marginal
    
    def test_coverage_guarantee(self, simple_data):
        """Test that coverage meets target (probabilistically)."""
        _, pred_stats = mondrian_conformal_calibrate(
            class_data=simple_data,
            alpha_target=0.10,
            delta=0.10,
            mode="beta"
        )
        
        marginal = pred_stats['marginal']
        coverage_rate = marginal['coverage']['rate']
        
        # Should have coverage >= 1 - alpha_target (at least probabilistically)
        # With delta=0.10, we expect this to hold 90% of the time
        assert coverage_rate >= 0.85  # Slightly below 0.90 to account for randomness
    
    def test_thresholds_are_valid(self, simple_data):
        """Test that thresholds are in valid range."""
        cal_result, _ = mondrian_conformal_calibrate(
            class_data=simple_data,
            alpha_target=0.10,
            delta=0.10,
            mode="beta"
        )
        
        for label in [0, 1]:
            threshold = cal_result[label]['threshold']
            assert 0 <= threshold <= 1
    
    def test_prediction_set_counts(self, simple_data):
        """Test that prediction set counts sum to n_total."""
        _, pred_stats = mondrian_conformal_calibrate(
            class_data=simple_data,
            alpha_target=0.10,
            delta=0.10,
            mode="beta"
        )
        
        marginal = pred_stats['marginal']
        n_total = marginal['n_total']
        
        n_abst = marginal['abstentions']['count']
        n_sing = marginal['singletons']['count']
        n_doub = marginal['doublets']['count']
        
        assert n_abst + n_sing + n_doub == n_total
    
    def test_singleton_breakdown(self, simple_data):
        """Test singleton breakdown by predicted class."""
        _, pred_stats = mondrian_conformal_calibrate(
            class_data=simple_data,
            alpha_target=0.10,
            delta=0.10,
            mode="beta"
        )
        
        marginal = pred_stats['marginal']
        singletons = marginal['singletons']
        
        n_sing_total = singletons['count']
        n_pred_0 = singletons['pred_0']
        n_pred_1 = singletons['pred_1']
        
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
            class_data=class_data,
            alpha_target=0.10,
            delta=0.10,
            mode="beta"
        )
        
        # Class 1 should have error message
        assert 'error' in cal_result[1]
        assert 'No calibration samples' in cal_result[1]['error']
    
    def test_beta_binomial_mode(self, simple_data):
        """Test beta-binomial mode."""
        cal_result, pred_stats = mondrian_conformal_calibrate(
            class_data=simple_data,
            alpha_target=0.10,
            delta=0.10,
            mode="beta-binomial",
            m=200
        )
        
        # Check that mode is set correctly
        for label in [0, 1]:
            if 'ssbc_result' in cal_result[label]:
                assert cal_result[label]['ssbc_result'].mode == "beta-binomial"
                assert cal_result[label]['ssbc_result'].details['m'] == 200
    
    def test_per_class_statistics(self, simple_data):
        """Test per-class statistics."""
        _, pred_stats = mondrian_conformal_calibrate(
            class_data=simple_data,
            alpha_target=0.10,
            delta=0.10,
            mode="beta"
        )
        
        for label in [0, 1]:
            if 'error' not in pred_stats[label]:
                stats = pred_stats[label]
                
                # Check that all prediction sets are accounted for
                n_class = stats['n_class']
                n_abst = stats['abstentions']['count']
                n_sing = stats['singletons']['count']
                n_doub = stats['doublets']['count']
                
                assert n_abst + n_sing + n_doub == n_class
    
    def test_pac_bounds_computation(self, simple_data):
        """Test PAC bounds computation."""
        _, pred_stats = mondrian_conformal_calibrate(
            class_data=simple_data,
            alpha_target=0.10,
            delta=0.10,
            mode="beta"
        )
        
        for label in [0, 1]:
            if 'error' not in pred_stats[label]:
                pac = pred_stats[label]['pac_bounds']
                
                # Check that PAC metrics are computed when applicable
                if pac['rho'] is not None:
                    assert pac['rho'] > 0
                    assert 0 <= pac['kappa'] <= 1
    
    def test_different_alphas_per_class(self, simple_data):
        """Test with different alpha values per class."""
        cal_result, _ = mondrian_conformal_calibrate(
            class_data=simple_data,
            alpha_target={0: 0.05, 1: 0.20},
            delta=0.10,
            mode="beta"
        )
        
        # Class 0 should be more conservative (lower alpha)
        # This typically means higher threshold for nonconformity scores
        alpha_0 = cal_result[0]['alpha_corrected']
        alpha_1 = cal_result[1]['alpha_corrected']
        
        assert alpha_0 < alpha_1
    
    def test_marginal_coverage_dict(self, simple_data):
        """Test marginal coverage dictionary structure."""
        _, pred_stats = mondrian_conformal_calibrate(
            class_data=simple_data,
            alpha_target=0.10,
            delta=0.10,
            mode="beta"
        )
        
        coverage = pred_stats['marginal']['coverage']
        
        assert 'count' in coverage
        assert 'rate' in coverage
        assert isinstance(coverage['ci_95'], dict)
    
    def test_prediction_sets_are_lists(self, simple_data):
        """Test that prediction sets are stored correctly."""
        _, pred_stats = mondrian_conformal_calibrate(
            class_data=simple_data,
            alpha_target=0.10,
            delta=0.10,
            mode="beta"
        )
        
        # Check per-class prediction sets
        for label in [0, 1]:
            if 'prediction_sets' in pred_stats[label]:
                pred_sets = pred_stats[label]['prediction_sets']
                assert isinstance(pred_sets, list)
                
                # Each prediction set should be a list
                for ps in pred_sets:
                    assert isinstance(ps, list)
                    # Each set should contain only 0, 1, or both
                    assert all(val in [0, 1] for val in ps)
        
        # Check marginal prediction sets
        if 'prediction_sets' in pred_stats['marginal']:
            marginal_sets = pred_stats['marginal']['prediction_sets']
            assert isinstance(marginal_sets, list)
    
    def test_reproducibility(self, simple_data):
        """Test that results are reproducible."""
        cal1, pred1 = mondrian_conformal_calibrate(
            class_data=simple_data,
            alpha_target=0.10,
            delta=0.10,
            mode="beta"
        )
        
        cal2, pred2 = mondrian_conformal_calibrate(
            class_data=simple_data,
            alpha_target=0.10,
            delta=0.10,
            mode="beta"
        )
        
        # Results should be identical
        for label in [0, 1]:
            if 'alpha_corrected' in cal1[label]:
                assert cal1[label]['alpha_corrected'] == cal2[label]['alpha_corrected']
                assert cal1[label]['threshold'] == cal2[label]['threshold']

