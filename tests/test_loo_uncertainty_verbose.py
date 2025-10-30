import io
import sys

import numpy as np

from ssbc.metrics import compute_robust_prediction_bounds


def test_loo_uncertainty_verbose_auto_and_provided() -> None:
    # Construct simple LOO predictions array
    loo_preds = np.array([1, 0, 1, 1, 0, 1, 0, 1])

    # Capture output for method="all" with estimated factor
    buf1 = io.StringIO()
    old = sys.stdout
    sys.stdout = buf1
    try:
        compute_robust_prediction_bounds(loo_preds, n_test=20, alpha=0.05, method="all", inflation_factor=None, verbose=True)
    finally:
        sys.stdout = old
    out1 = buf1.getvalue()
    assert "Using estimated LOO inflation factor" in out1

    # Capture output with provided factor
    buf2 = io.StringIO()
    old = sys.stdout
    sys.stdout = buf2
    try:
        compute_robust_prediction_bounds(loo_preds, n_test=20, alpha=0.05, method="all", inflation_factor=2.0, verbose=True)
    finally:
        sys.stdout = old
    out2 = buf2.getvalue()
    assert "Using provided LOO inflation factor" in out2

