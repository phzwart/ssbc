"""Tests for the simulation module."""

import numpy as np
import pytest

from ssbc.simulation import BinaryClassifierSimulator


class TestBinaryClassifierSimulator:
    """Test BinaryClassifierSimulator class."""

    def test_initialization(self):
        """Test simulator initialization."""
        sim = BinaryClassifierSimulator(p_class1=0.3, beta_params_class0=(2, 8), beta_params_class1=(8, 2), seed=42)

        assert sim.p_class1 == 0.3
        assert sim.p_class0 == 0.7
        assert sim.a0 == 2
        assert sim.b0 == 8
        assert sim.a1 == 8
        assert sim.b1 == 2

    def test_generate_shape(self):
        """Test that generate returns correct shapes."""
        sim = BinaryClassifierSimulator(p_class1=0.5, beta_params_class0=(2, 8), beta_params_class1=(8, 2), seed=42)

        labels, probs = sim.generate(n_samples=100)

        assert labels.shape == (100,)
        assert probs.shape == (100, 2)

    def test_generate_labels_binary(self):
        """Test that labels are binary (0 or 1)."""
        sim = BinaryClassifierSimulator(p_class1=0.5, beta_params_class0=(2, 8), beta_params_class1=(8, 2), seed=42)

        labels, _ = sim.generate(n_samples=100)

        assert np.all((labels == 0) | (labels == 1))

    def test_probs_sum_to_one(self):
        """Test that probabilities sum to 1 for each sample."""
        sim = BinaryClassifierSimulator(p_class1=0.5, beta_params_class0=(2, 8), beta_params_class1=(8, 2), seed=42)

        _, probs = sim.generate(n_samples=100)

        row_sums = probs.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, rtol=1e-10)

    def test_probs_in_valid_range(self):
        """Test that probabilities are in [0, 1]."""
        sim = BinaryClassifierSimulator(p_class1=0.5, beta_params_class0=(2, 8), beta_params_class1=(8, 2), seed=42)

        _, probs = sim.generate(n_samples=100)

        assert np.all(probs >= 0)
        assert np.all(probs <= 1)

    def test_class_distribution(self):
        """Test that class distribution matches p_class1."""
        sim = BinaryClassifierSimulator(p_class1=0.3, beta_params_class0=(2, 8), beta_params_class1=(8, 2), seed=42)

        labels, _ = sim.generate(n_samples=10000)

        observed_p_class1 = np.mean(labels == 1)

        # With large sample, should be close to 0.3
        assert abs(observed_p_class1 - 0.3) < 0.02

    def test_beta_distribution_class0(self):
        """Test that class 0 samples have low P(class=1)."""
        # Class 0 with Beta(2, 8) should have low P(class=1)
        sim = BinaryClassifierSimulator(
            p_class1=0.5,
            beta_params_class0=(2, 8),  # Mean = 2/10 = 0.2
            beta_params_class1=(8, 2),
            seed=42,
        )

        labels, probs = sim.generate(n_samples=1000)

        # For class 0 samples, P(class=1) should be low on average
        class0_probs = probs[labels == 0, 1]
        mean_prob = np.mean(class0_probs)

        # Should be around 0.2 (Beta(2,8) mean)
        assert 0.15 < mean_prob < 0.25

    def test_beta_distribution_class1(self):
        """Test that class 1 samples have high P(class=1)."""
        # Class 1 with Beta(8, 2) should have high P(class=1)
        sim = BinaryClassifierSimulator(
            p_class1=0.5,
            beta_params_class0=(2, 8),
            beta_params_class1=(8, 2),  # Mean = 8/10 = 0.8
            seed=42,
        )

        labels, probs = sim.generate(n_samples=1000)

        # For class 1 samples, P(class=1) should be high on average
        class1_probs = probs[labels == 1, 1]
        mean_prob = np.mean(class1_probs)

        # Should be around 0.8 (Beta(8,2) mean)
        assert 0.75 < mean_prob < 0.85

    def test_reproducibility_with_seed(self):
        """Test that results are reproducible with same seed."""
        sim1 = BinaryClassifierSimulator(p_class1=0.3, beta_params_class0=(2, 8), beta_params_class1=(8, 2), seed=42)

        sim2 = BinaryClassifierSimulator(p_class1=0.3, beta_params_class0=(2, 8), beta_params_class1=(8, 2), seed=42)

        labels1, probs1 = sim1.generate(n_samples=100)
        labels2, probs2 = sim2.generate(n_samples=100)

        np.testing.assert_array_equal(labels1, labels2)
        np.testing.assert_allclose(probs1, probs2)

    def test_different_seeds_produce_different_results(self):
        """Test that different seeds produce different results."""
        sim1 = BinaryClassifierSimulator(p_class1=0.3, beta_params_class0=(2, 8), beta_params_class1=(8, 2), seed=42)

        sim2 = BinaryClassifierSimulator(p_class1=0.3, beta_params_class0=(2, 8), beta_params_class1=(8, 2), seed=123)

        labels1, probs1 = sim1.generate(n_samples=100)
        labels2, probs2 = sim2.generate(n_samples=100)

        # Should be different
        assert not np.array_equal(labels1, labels2)

    def test_edge_case_p_class1_zero(self):
        """Test with p_class1=0 (all class 0)."""
        sim = BinaryClassifierSimulator(p_class1=0.0, beta_params_class0=(2, 8), beta_params_class1=(8, 2), seed=42)

        labels, _ = sim.generate(n_samples=100)

        assert np.all(labels == 0)

    def test_edge_case_p_class1_one(self):
        """Test with p_class1=1 (all class 1)."""
        sim = BinaryClassifierSimulator(p_class1=1.0, beta_params_class0=(2, 8), beta_params_class1=(8, 2), seed=42)

        labels, _ = sim.generate(n_samples=100)

        assert np.all(labels == 1)

    def test_invalid_p_class1_negative(self):
        """Test that negative p_class1 raises ValueError."""
        with pytest.raises(ValueError, match="p_class1 must be in"):
            BinaryClassifierSimulator(p_class1=-0.1, beta_params_class0=(2, 8), beta_params_class1=(8, 2))

    def test_invalid_p_class1_greater_than_one(self):
        """Test that p_class1 > 1 raises ValueError."""
        with pytest.raises(ValueError, match="p_class1 must be in"):
            BinaryClassifierSimulator(p_class1=1.5, beta_params_class0=(2, 8), beta_params_class1=(8, 2))

    def test_invalid_beta_params_class0_negative(self):
        """Test that negative beta parameters raise ValueError."""
        with pytest.raises(ValueError, match="Beta parameters for class 0"):
            BinaryClassifierSimulator(p_class1=0.5, beta_params_class0=(-1, 8), beta_params_class1=(8, 2))

    def test_invalid_beta_params_class1_negative(self):
        """Test that negative beta parameters raise ValueError."""
        with pytest.raises(ValueError, match="Beta parameters for class 1"):
            BinaryClassifierSimulator(p_class1=0.5, beta_params_class0=(2, 8), beta_params_class1=(8, -2))

    def test_invalid_n_samples_zero(self):
        """Test that n_samples=0 raises ValueError."""
        sim = BinaryClassifierSimulator(p_class1=0.5, beta_params_class0=(2, 8), beta_params_class1=(8, 2))

        with pytest.raises(ValueError, match="n_samples must be positive"):
            sim.generate(n_samples=0)

    def test_invalid_n_samples_negative(self):
        """Test that negative n_samples raises ValueError."""
        sim = BinaryClassifierSimulator(p_class1=0.5, beta_params_class0=(2, 8), beta_params_class1=(8, 2))

        with pytest.raises(ValueError, match="n_samples must be positive"):
            sim.generate(n_samples=-10)

    def test_repr(self):
        """Test string representation."""
        sim = BinaryClassifierSimulator(p_class1=0.3, beta_params_class0=(2, 8), beta_params_class1=(8, 2))

        repr_str = repr(sim)

        assert "BinaryClassifierSimulator" in repr_str
        assert "0.300" in repr_str
        assert "(2" in repr_str
        assert "8)" in repr_str

    def test_small_sample(self):
        """Test with very small sample."""
        sim = BinaryClassifierSimulator(p_class1=0.5, beta_params_class0=(2, 8), beta_params_class1=(8, 2), seed=42)

        labels, probs = sim.generate(n_samples=1)

        assert labels.shape == (1,)
        assert probs.shape == (1, 2)
        assert probs[0, 0] + probs[0, 1] == pytest.approx(1.0)

    def test_large_sample(self):
        """Test with large sample."""
        sim = BinaryClassifierSimulator(p_class1=0.5, beta_params_class0=(2, 8), beta_params_class1=(8, 2), seed=42)

        labels, probs = sim.generate(n_samples=10000)

        assert labels.shape == (10000,)
        assert probs.shape == (10000, 2)
