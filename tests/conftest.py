"""Pytest configuration for SSBC tests.

Sets a non-interactive matplotlib backend and disables plt.show() calls
to avoid warnings and GUI requirements in CI environments.
"""

import matplotlib


def pytest_configure() -> None:
    # Use non-interactive backend
    matplotlib.use("Agg", force=True)
    try:
        import matplotlib.pyplot as plt  # noqa: WPS433

        # Disable plt.show() in tests
        plt.show = lambda *args, **kwargs: None  # type: ignore[assignment]
    except Exception:  # pragma: no cover - best effort only
        pass
