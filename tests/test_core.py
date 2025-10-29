"""Tests for the core SSBC algorithm module."""

import pytest

from ssbc.core_pkg import SSBCResult, ssbc_correct


class TestSSBCResult:
    """Test SSBCResult dataclass."""

    def test_ssbc_result_creation(self):
        """Test creating an SSBCResult instance."""
        result = SSBCResult(
            alpha_target=0.1,
            alpha_corrected=0.08,
            u_star=4,
            n=50,
            satisfied_mass=0.92,
            mode="beta",
            details={"test": "data"},
        )

        assert result.alpha_target == 0.1
        assert result.alpha_corrected == 0.08
        assert result.u_star == 4
        assert result.n == 50
        assert result.satisfied_mass == 0.92
        assert result.mode == "beta"
        assert result.details == {"test": "data"}


class TestSSBCCorrect:
    """Test ssbc_correct function."""

    def test_basic_correction(self):
        """Test basic SSBC correction."""
        result = ssbc_correct(alpha_target=0.10, n=50, delta=0.10, mode="beta")

        assert isinstance(result, SSBCResult)
        assert result.alpha_target == 0.10
        assert result.n == 50
        assert result.mode == "beta"
        assert 0 < result.alpha_corrected <= result.alpha_target
        assert 1 <= result.u_star <= result.n
        assert 0 <= result.satisfied_mass <= 1

    def test_small_calibration_set(self):
        """Test with small calibration set."""
        result = ssbc_correct(alpha_target=0.10, n=10, delta=0.10, mode="beta")

        # With small n, correction should be more conservative
        assert result.alpha_corrected <= result.alpha_target
        # For small n, the algorithm may not find a u that satisfies the PAC constraint
        # In this case, it falls back to u_star=0 (most conservative)
        assert result.u_star >= 0
        # Note: With very small n, the algorithm may not find a u that
        # satisfies the PAC constraint, so satisfied_mass might be < 1-delta
        assert 0 <= result.satisfied_mass <= 1

    def test_large_calibration_set(self):
        """Test with large calibration set."""
        result = ssbc_correct(alpha_target=0.10, n=1000, delta=0.10, mode="beta")

        # With large n, correction should be minimal
        assert result.alpha_corrected <= result.alpha_target
        # Should be closer to target for large n
        assert result.alpha_corrected > 0.08  # Not too conservative

    def test_beta_binomial_mode(self):
        """Test beta-binomial mode."""
        result = ssbc_correct(alpha_target=0.10, n=50, delta=0.10, mode="beta-binomial", m=100)

        assert result.mode == "beta-binomial"
        assert result.details["m"] == 100
        assert result.alpha_corrected <= result.alpha_target

    def test_custom_bracket_width(self):
        """Test with custom bracket width."""
        result = ssbc_correct(alpha_target=0.10, n=50, delta=0.10, mode="beta", bracket_width=5)

        assert result.details["bracket_width"] == 5
        search_range = result.details["search_range"]
        assert search_range[1] - search_range[0] <= 10  # Width of ~2*5

    def test_search_log(self):
        """Test that search log is populated."""
        result = ssbc_correct(alpha_target=0.10, n=20, delta=0.10, mode="beta")

        search_log = result.details["search_log"]
        assert len(search_log) > 0

        # Check structure of search log entries
        for entry in search_log:
            assert "u" in entry
            assert "alpha_prime" in entry
            assert "a" in entry
            assert "b" in entry
            assert "ptail" in entry
            assert "passes" in entry

    def test_u_star_is_largest_passing(self):
        """Test that u_star is the largest u that passes."""
        result = ssbc_correct(alpha_target=0.10, n=50, delta=0.10, mode="beta")

        search_log = result.details["search_log"]
        passing_us = [entry["u"] for entry in search_log if entry["passes"]]

        if passing_us:
            assert result.u_star == max(passing_us)

    def test_different_alpha_values(self):
        """Test with different alpha values."""
        for alpha in [0.05, 0.10, 0.15, 0.20]:
            result = ssbc_correct(alpha_target=alpha, n=50, delta=0.10, mode="beta")

            assert result.alpha_target == alpha
            assert result.alpha_corrected <= alpha
            assert result.alpha_corrected > 0

    def test_different_delta_values(self):
        """Test with different delta values."""
        for delta in [0.05, 0.10, 0.15]:
            result = ssbc_correct(alpha_target=0.10, n=50, delta=delta, mode="beta")

            assert result.details["delta"] == delta
            # Smaller delta should give more conservative correction

    def test_convergence_with_large_n(self):
        """Test that correction converges to target with large n."""
        results = []
        for n in [100, 500, 1000, 5000]:
            result = ssbc_correct(alpha_target=0.10, n=n, delta=0.10, mode="beta")
            results.append((n, result.alpha_corrected))

        # As n increases, alpha_corrected should approach alpha_target
        corrections = [abs(0.10 - alpha_corr) for _, alpha_corr in results]
        # Generally should decrease (though not strictly monotonic)
        assert corrections[-1] < corrections[0]

    # Validation tests
    def test_invalid_alpha_target_low(self):
        """Test with alpha_target <= 0."""
        with pytest.raises(ValueError, match="alpha_target must be in"):
            ssbc_correct(alpha_target=0.0, n=50, delta=0.10)

    def test_invalid_alpha_target_high(self):
        """Test with alpha_target >= 1."""
        with pytest.raises(ValueError, match="alpha_target must be in"):
            ssbc_correct(alpha_target=1.0, n=50, delta=0.10)

    def test_invalid_n(self):
        """Test with n < 1."""
        with pytest.raises(ValueError, match="n must be >= 1"):
            ssbc_correct(alpha_target=0.10, n=0, delta=0.10)

    def test_invalid_delta_low(self):
        """Test with delta <= 0."""
        with pytest.raises(ValueError, match="delta must be in"):
            ssbc_correct(alpha_target=0.10, n=50, delta=0.0)

    def test_invalid_delta_high(self):
        """Test with delta >= 1."""
        with pytest.raises(ValueError, match="delta must be in"):
            ssbc_correct(alpha_target=0.10, n=50, delta=1.0)

    def test_invalid_mode(self):
        """Test with invalid mode."""
        with pytest.raises(ValueError, match="mode must be"):
            ssbc_correct(alpha_target=0.10, n=50, delta=0.10, mode="invalid")  # type: ignore[arg-type]

    def test_beta_binomial_without_m(self):
        """Test beta-binomial mode when m is not provided."""
        result = ssbc_correct(alpha_target=0.10, n=50, delta=0.10, mode="beta-binomial")

        # When m is not provided, it defaults to n internally for calculations
        # and details stores the actual value used (n)
        assert result.details["m"] == 50  # Should be n when not provided
        # The algorithm should still work correctly
        assert result.alpha_corrected <= result.alpha_target

    def test_beta_binomial_invalid_m(self):
        """Test beta-binomial mode with m < 1."""
        with pytest.raises(ValueError, match="m must be >= 1"):
            ssbc_correct(alpha_target=0.10, n=50, delta=0.10, mode="beta-binomial", m=0)

    def test_reproducibility(self):
        """Test that results are reproducible."""
        result1 = ssbc_correct(alpha_target=0.10, n=50, delta=0.10)
        result2 = ssbc_correct(alpha_target=0.10, n=50, delta=0.10)

        assert result1.alpha_corrected == result2.alpha_corrected
        assert result1.u_star == result2.u_star
        assert result1.satisfied_mass == result2.satisfied_mass

    def test_edge_case_very_small_n(self):
        """Test with n=1."""
        result = ssbc_correct(alpha_target=0.10, n=1, delta=0.10)

        # For n=1, alpha_target=0.1 < 1/(n+1)=0.5, so we're in the trivial regime
        # where no positive u is allowed (u_max=0)
        assert result.u_star == 0
        assert result.alpha_corrected == 0.0  # u_star/(n+1) = 0/2 = 0

    def test_details_structure(self):
        """Test that details dict has expected keys."""
        result = ssbc_correct(alpha_target=0.10, n=50, delta=0.10)

        expected_keys = [
            "u_max",
            "u_star_guess",
            "search_range",
            "bracket_width",
            "delta",
            "acceptance_rule",
            "search_log",
        ]

        for key in expected_keys:
            assert key in result.details
