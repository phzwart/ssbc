import io
import sys

from ssbc.reporting import generate_rigorous_pac_report
from ssbc.simulation import BinaryClassifierSimulator


def test_reporting_prints_method_comparison() -> None:
    sim = BinaryClassifierSimulator(p_class1=0.3, beta_params_class0=(2, 5), beta_params_class1=(6, 2), seed=7)
    labels, probs = sim.generate(60)

    generate_rigorous_pac_report(
        labels=labels,
        probs=probs,
        alpha_target=0.1,
        delta=0.1,
        test_size=30,
        ci_level=0.95,
        prediction_method="all",
        use_loo_correction=True,
        verbose=True,
    )

    # Capture print
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        # Re-print by calling the internal printer indirectly: regenerate with verbose
        generate_rigorous_pac_report(
            labels=labels,
            probs=probs,
            alpha_target=0.1,
            delta=0.1,
            test_size=30,
            ci_level=0.95,
            prediction_method="all",
            use_loo_correction=True,
            verbose=True,
        )
    finally:
        sys.stdout = old
    out = buf.getvalue()

    assert "Method comparison:" in out
    assert "Selected bounds:" in out
