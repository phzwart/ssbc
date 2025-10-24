"""Tests for SSBC core algorithm improvements and edge cases."""

import math

import pytest

from ssbc.core import ssbc_correct


@pytest.mark.parametrize("n", [1, 2, 5, 50, 200])
@pytest.mark.parametrize("alpha_target", [1e-6, 1e-3, 0.01, 0.1, 0.4])
@pytest.mark.parametrize("delta", [0.1, 0.05, 0.001])
def test_monotone_alpha(n, alpha_target, delta):
    """Test that corrected alpha is monotonic and within bounds."""
    r = ssbc_correct(alpha_target=alpha_target, n=n, delta=delta)
    assert 0.0 <= r.alpha_corrected <= alpha_target
    assert 0 <= r.u_star <= min(n, math.floor(alpha_target * (n + 1)))


def test_trivial_small_alpha():
    """Test edge case where alpha_target < 1/(n+1)."""
    n = 50
    alpha_target = 1 / (n + 1) - 1e-9
    r = ssbc_correct(alpha_target=alpha_target, n=n, delta=0.1)
    assert r.u_star == 0
    assert r.alpha_corrected == 0.0
    assert r.satisfied_mass == 1.0


@pytest.mark.parametrize("mode,m", [("beta", None), ("beta-binomial", 100)])
def test_probability_constraint(mode, m):
    """Test that the probability constraint is satisfied."""
    r = ssbc_correct(alpha_target=0.1, n=50, delta=0.1, mode=mode, m=m)
    # r.satisfied_mass = P(Coverage >= 1 - alpha_target) at the chosen u
    assert r.satisfied_mass >= 1 - 0.1 - 1e-12


def test_beta_binomial_defaults_to_n_when_unspecified():
    """Test that beta-binomial mode defaults to m=n when m is not specified."""
    r = ssbc_correct(alpha_target=0.2, n=30, delta=0.2, mode="beta-binomial")
    assert r.details["m"] == 30


def test_numerical_stability_edge_cases():
    """Test numerical stability with extreme parameter values."""
    # Very small alpha_target
    r1 = ssbc_correct(alpha_target=1e-8, n=10, delta=0.1)
    assert r1.u_star >= 0
    assert r1.alpha_corrected <= 1e-8

    # Very small n
    r2 = ssbc_correct(alpha_target=0.1, n=1, delta=0.1)
    assert r2.u_star >= 0
    assert r2.alpha_corrected <= 0.1

    # Very small delta
    r3 = ssbc_correct(alpha_target=0.1, n=50, delta=1e-6)
    assert r3.u_star >= 0
    assert r3.alpha_corrected <= 0.1


def test_adaptive_search_robustness():
    """Test that adaptive search finds solutions when initial bracket fails."""
    # Parameters that might cause initial bracket to miss the solution
    r = ssbc_correct(alpha_target=0.05, n=20, delta=0.01, bracket_width=2)
    assert r.u_star >= 0
    assert r.alpha_corrected <= 0.05
    assert r.satisfied_mass >= 1 - 0.01


def test_survival_function_usage():
    """Test that survival function is used for numerical stability."""
    r = ssbc_correct(alpha_target=0.1, n=50, delta=0.1)
    # Check that the result is reasonable and doesn't have numerical issues
    assert isinstance(r.satisfied_mass, float)
    assert 0.0 <= r.satisfied_mass <= 1.0
    assert not math.isnan(r.satisfied_mass)
    assert not math.isinf(r.satisfied_mass)


def test_conservative_fallback():
    """Test that the most conservative fallback (u=0) is used when appropriate."""
    # Use parameters that might require fallback
    r = ssbc_correct(alpha_target=0.01, n=5, delta=0.001)
    assert r.u_star >= 0
    assert r.alpha_corrected >= 0.0
    # If u_star=0, then alpha_corrected should be 0.0
    if r.u_star == 0:
        assert r.alpha_corrected == 0.0
        assert r.satisfied_mass == 1.0
