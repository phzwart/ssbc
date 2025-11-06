import numpy as np

from ssbc.core_pkg import SSBCResult
from ssbc.metrics.operational_bounds_simple import compute_pac_operational_bounds_marginal_loo_corrected
from ssbc.simulation import BinaryClassifierSimulator


def test_operational_bounds_loo_diagnostics_present() -> None:
    sim = BinaryClassifierSimulator(p_class1=0.3, beta_params_class0=(2, 5), beta_params_class1=(6, 2), seed=5)
    labels, probs = sim.generate(60)

    # Create minimal SSBCResult-like objects with required fields
    n0 = int(np.sum(labels == 0))
    n1 = int(np.sum(labels == 1))
    # Minimal valid SSBCResult instances
    u0 = max(1, int((n0 + 1) * 0.1))
    u1 = max(1, int((n1 + 1) * 0.1))
    ssbc0 = SSBCResult(
        alpha_target=0.1,
        alpha_corrected=0.1,
        u_star=u0,
        n=n0,
        satisfied_mass=1.0,
        mode="beta",
        details={},
    )
    ssbc1 = SSBCResult(
        alpha_target=0.1,
        alpha_corrected=0.1,
        u_star=u1,
        n=n1,
        satisfied_mass=1.0,
        mode="beta",
        details={},
    )

    pac = compute_pac_operational_bounds_marginal_loo_corrected(
        ssbc_result_0=ssbc0,
        ssbc_result_1=ssbc1,
        labels=labels,
        probs=probs,
        test_size=30,
        ci_level=0.95,
        pac_level=0.9,
        prediction_method="all",
        loo_inflation_factor=2.0,
        verbose=False,
    )

    assert "loo_diagnostics" in pac
    # Marginal diagnostics should NOT include conditional rates (those are in per-class scopes)
    # It should include joint rates (normalized by total) and rates conditioned on predicted class
    expected_keys = {
        "singleton",
        "doublet",
        "abstention",
        "singleton_error_class0",
        "singleton_error_class1",
        "singleton_correct_class0",
        "singleton_correct_class1",
        "singleton_error_pred_class0",
        "singleton_error_pred_class1",
        "singleton_correct_pred_class0",
        "singleton_correct_pred_class1",
    }
    assert expected_keys.issubset(pac["loo_diagnostics"].keys())
