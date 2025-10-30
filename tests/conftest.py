"""Pytest configuration for SSBC tests.

Sets a non-interactive matplotlib backend and disables plt.show() calls
to avoid warnings and GUI requirements in CI environments.
"""

import warnings

import matplotlib


def pytest_configure() -> None:
    # Use non-interactive backend
    matplotlib.use("Agg", force=True)
    # Silence expected small-sample warnings from LOO paths in tests
    warnings.filterwarnings(
        "ignore",
        message=r"^n_cal=\d+ is very small\. Consider using Method 2 \(exact binomial\) or Method 3 \(Hoeffding\)",
        category=UserWarning,
    )
    try:
        import matplotlib.pyplot as plt

        # Disable plt.show() in tests
        plt.show = lambda *args, **kwargs: None  # type: ignore[assignment]
    except Exception:  # pragma: no cover - best effort only
        pass
