"""Tests for the hyperparameter module."""

import numpy as np
import pandas as pd
import pytest

from ssbc.conformal import split_by_class
from ssbc.hyperparameter import sweep_and_plot_parallel_plotly, sweep_hyperparams_and_collect
from ssbc.simulation import BinaryClassifierSimulator


class TestSweepHyperparamsAndCollect:
    """Test sweep_hyperparams_and_collect function."""

    @pytest.fixture
    def sample_class_data(self):
        """Create sample class data for testing."""
        sim = BinaryClassifierSimulator(p_class1=0.5, beta_params_class0=(2, 8), beta_params_class1=(8, 2), seed=42)
        labels, probs = sim.generate(n_samples=100)
        return split_by_class(labels, probs)

    def test_basic_sweep(self, sample_class_data):
        """Test basic hyperparameter sweep."""
        alpha_grid = np.array([0.05, 0.10])
        delta_grid = np.array([0.05, 0.10])

        df = sweep_hyperparams_and_collect(
            class_data=sample_class_data,
            alpha_0=alpha_grid,
            delta_0=delta_grid,
            alpha_1=alpha_grid,
            delta_1=delta_grid,
            mode="beta",
            quiet=True,
        )

        # Should have 2*2*2*2 = 16 rows
        assert len(df) == 16
        assert isinstance(df, pd.DataFrame)

    def test_dataframe_columns(self, sample_class_data):
        """Test that dataframe has expected columns."""
        alpha_grid = np.array([0.10])
        delta_grid = np.array([0.10])

        df = sweep_hyperparams_and_collect(
            class_data=sample_class_data,
            alpha_0=alpha_grid,
            delta_0=delta_grid,
            alpha_1=alpha_grid,
            delta_1=delta_grid,
            quiet=True,
        )

        expected_cols = [
            "a0",
            "d0",
            "a1",
            "d1",
            "cov",
            "sing_rate",
            "err_all",
            "err_pred0",
            "err_pred1",
            "err_y0",
            "err_y1",
            "esc_rate",
            "n_total",
            "sing_count",
            "m_abst",
            "m_doublets",
        ]

        for col in expected_cols:
            assert col in df.columns

    def test_values_in_valid_range(self, sample_class_data):
        """Test that all values are in valid ranges."""
        alpha_grid = np.array([0.10, 0.15])
        delta_grid = np.array([0.10, 0.15])

        df = sweep_hyperparams_and_collect(
            class_data=sample_class_data,
            alpha_0=alpha_grid,
            delta_0=delta_grid,
            alpha_1=alpha_grid,
            delta_1=delta_grid,
            quiet=True,
        )

        # Check hyperparameters
        assert df["a0"].min() >= 0
        assert df["a0"].max() <= 1
        assert df["d0"].min() >= 0
        assert df["d0"].max() <= 1

        # Check metrics (rates should be in [0, 1])
        assert df["cov"].min() >= 0
        assert df["cov"].max() <= 1
        assert df["sing_rate"].min() >= 0
        assert df["sing_rate"].max() <= 1
        assert df["err_all"].min() >= 0
        assert df["err_all"].max() <= 1
        assert df["esc_rate"].min() >= 0
        assert df["esc_rate"].max() <= 1

    def test_sorted_output(self, sample_class_data):
        """Test that output is sorted by hyperparameters."""
        alpha_grid = np.array([0.15, 0.05, 0.10])  # Unsorted
        delta_grid = np.array([0.10, 0.05])

        df = sweep_hyperparams_and_collect(
            class_data=sample_class_data,
            alpha_0=alpha_grid,
            delta_0=delta_grid,
            alpha_1=alpha_grid,
            delta_1=delta_grid,
            quiet=True,
        )

        # Should be sorted by a0, d0, a1, d1
        assert df["a0"].is_monotonic_increasing or df["a0"].nunique() == 1

    def test_quiet_mode(self, sample_class_data, capsys):
        """Test quiet mode suppresses output."""
        alpha_grid = np.array([0.10])
        delta_grid = np.array([0.10])

        sweep_hyperparams_and_collect(
            class_data=sample_class_data,
            alpha_0=alpha_grid,
            delta_0=delta_grid,
            alpha_1=alpha_grid,
            delta_1=delta_grid,
            quiet=True,
        )

        captured = capsys.readouterr()
        assert captured.out == ""

    def test_verbose_mode(self, sample_class_data, capsys):
        """Test verbose mode prints progress."""
        alpha_grid = np.array([0.10])
        delta_grid = np.array([0.10])

        sweep_hyperparams_and_collect(
            class_data=sample_class_data,
            alpha_0=alpha_grid,
            delta_0=delta_grid,
            alpha_1=alpha_grid,
            delta_1=delta_grid,
            quiet=False,
        )

        captured = capsys.readouterr()
        assert "a0=" in captured.out
        assert "d0=" in captured.out

    def test_beta_binomial_mode(self, sample_class_data):
        """Test with beta-binomial mode."""
        alpha_grid = np.array([0.10])
        delta_grid = np.array([0.10])

        df = sweep_hyperparams_and_collect(
            class_data=sample_class_data,
            alpha_0=alpha_grid,
            delta_0=delta_grid,
            alpha_1=alpha_grid,
            delta_1=delta_grid,
            mode="beta-binomial",
            quiet=True,
        )

        assert len(df) == 1
        # Results should still be valid
        assert 0 <= df.iloc[0]["cov"] <= 1

    def test_extra_metrics(self, sample_class_data):
        """Test with extra metrics function."""
        alpha_grid = np.array([0.10])
        delta_grid = np.array([0.10])

        def custom_metric(summary):
            """Custom metric: sum of error rates."""
            return summary.get("marginal", {}).get("singletons", {}).get("errors", {}).get("rate", 0.0)

        extra_metrics = {"custom": custom_metric}

        df = sweep_hyperparams_and_collect(
            class_data=sample_class_data,
            alpha_0=alpha_grid,
            delta_0=delta_grid,
            alpha_1=alpha_grid,
            delta_1=delta_grid,
            extra_metrics=extra_metrics,
            quiet=True,
        )

        assert "custom" in df.columns

    def test_extra_metrics_with_exception(self, sample_class_data):
        """Test that exceptions in extra metrics are handled."""
        alpha_grid = np.array([0.10])
        delta_grid = np.array([0.10])

        def bad_metric(summary):
            """Metric that raises an exception."""
            raise ValueError("Test error")

        extra_metrics = {"bad": bad_metric}

        df = sweep_hyperparams_and_collect(
            class_data=sample_class_data,
            alpha_0=alpha_grid,
            delta_0=delta_grid,
            alpha_1=alpha_grid,
            delta_1=delta_grid,
            extra_metrics=extra_metrics,
            quiet=True,
        )

        # Should have NaN for the bad metric
        assert "bad" in df.columns
        assert pd.isna(df.iloc[0]["bad"])

    def test_grid_size(self, sample_class_data):
        """Test that grid size is computed correctly."""
        alpha_grid = np.array([0.05, 0.10, 0.15])
        delta_grid = np.array([0.10, 0.15])

        df = sweep_hyperparams_and_collect(
            class_data=sample_class_data,
            alpha_0=alpha_grid,
            delta_0=delta_grid,
            alpha_1=alpha_grid,
            delta_1=delta_grid,
            quiet=True,
        )

        # 3 * 2 * 3 * 2 = 36 combinations
        assert len(df) == 36

    def test_single_configuration(self, sample_class_data):
        """Test with single configuration (no sweep)."""
        df = sweep_hyperparams_and_collect(
            class_data=sample_class_data,
            alpha_0=np.array([0.10]),
            delta_0=np.array([0.10]),
            alpha_1=np.array([0.10]),
            delta_1=np.array([0.10]),
            quiet=True,
        )

        assert len(df) == 1
        assert df.iloc[0]["a0"] == 0.10
        assert df.iloc[0]["d0"] == 0.10


class TestSweepAndPlotParallelPlotly:
    """Test sweep_and_plot_parallel_plotly function."""

    @pytest.fixture
    def sample_class_data(self):
        """Create sample class data for testing."""
        sim = BinaryClassifierSimulator(p_class1=0.5, beta_params_class0=(2, 8), beta_params_class1=(8, 2), seed=42)
        labels, probs = sim.generate(n_samples=50)
        return split_by_class(labels, probs)

    def test_basic_sweep_and_plot(self, sample_class_data):
        """Test basic sweep and plot."""
        alpha_grid = np.array([0.10, 0.15])
        delta_grid = np.array([0.10, 0.15])

        df, fig = sweep_and_plot_parallel_plotly(
            class_data=sample_class_data,
            delta_0=delta_grid,
            delta_1=delta_grid,
            alpha_0=alpha_grid,
            alpha_1=alpha_grid,
            mode="beta",
        )

        # Check dataframe
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 16  # 2*2*2*2

        # Check figure
        assert fig is not None
        assert hasattr(fig, "data")
        assert hasattr(fig, "layout")

    def test_returns_both_df_and_fig(self, sample_class_data):
        """Test that both dataframe and figure are returned."""
        alpha_grid = np.array([0.10])
        delta_grid = np.array([0.10])

        result = sweep_and_plot_parallel_plotly(
            class_data=sample_class_data, delta_0=delta_grid, delta_1=delta_grid, alpha_0=alpha_grid, alpha_1=alpha_grid
        )

        assert isinstance(result, tuple)
        assert len(result) == 2

        df, fig = result
        assert isinstance(df, pd.DataFrame)
        assert fig is not None

    def test_custom_color(self, sample_class_data):
        """Test with custom color parameter."""
        alpha_grid = np.array([0.10])
        delta_grid = np.array([0.10])

        df, fig = sweep_and_plot_parallel_plotly(
            class_data=sample_class_data,
            delta_0=delta_grid,
            delta_1=delta_grid,
            alpha_0=alpha_grid,
            alpha_1=alpha_grid,
            color="cov",
        )

        assert fig is not None

    def test_custom_title(self, sample_class_data):
        """Test with custom title."""
        alpha_grid = np.array([0.10])
        delta_grid = np.array([0.10])

        custom_title = "My Custom Title"
        df, fig = sweep_and_plot_parallel_plotly(
            class_data=sample_class_data,
            delta_0=delta_grid,
            delta_1=delta_grid,
            alpha_0=alpha_grid,
            alpha_1=alpha_grid,
            title=custom_title,
        )

        assert fig.layout.title.text == custom_title

    def test_default_title(self, sample_class_data):
        """Test that default title includes config count."""
        alpha_grid = np.array([0.10, 0.15])
        delta_grid = np.array([0.10])

        df, fig = sweep_and_plot_parallel_plotly(
            class_data=sample_class_data, delta_0=delta_grid, delta_1=delta_grid, alpha_0=alpha_grid, alpha_1=alpha_grid
        )

        # Default title should mention number of configs
        assert "4 configs" in fig.layout.title.text or "n=4" in fig.layout.title.text

    def test_custom_height(self, sample_class_data):
        """Test with custom height."""
        alpha_grid = np.array([0.10])
        delta_grid = np.array([0.10])

        custom_height = 800
        df, fig = sweep_and_plot_parallel_plotly(
            class_data=sample_class_data,
            delta_0=delta_grid,
            delta_1=delta_grid,
            alpha_0=alpha_grid,
            alpha_1=alpha_grid,
            height=custom_height,
        )

        assert fig.layout.height == custom_height

    def test_with_extra_metrics(self, sample_class_data):
        """Test with extra metrics."""
        alpha_grid = np.array([0.10])
        delta_grid = np.array([0.10])

        def custom_metric(summary):
            return 0.5

        extra_metrics = {"custom": custom_metric}

        df, fig = sweep_and_plot_parallel_plotly(
            class_data=sample_class_data,
            delta_0=delta_grid,
            delta_1=delta_grid,
            alpha_0=alpha_grid,
            alpha_1=alpha_grid,
            extra_metrics=extra_metrics,
        )

        assert "custom" in df.columns

    def test_colorscale(self, sample_class_data):
        """Test with custom colorscale."""
        import plotly.express as px

        alpha_grid = np.array([0.10])
        delta_grid = np.array([0.10])

        df, fig = sweep_and_plot_parallel_plotly(
            class_data=sample_class_data,
            delta_0=delta_grid,
            delta_1=delta_grid,
            alpha_0=alpha_grid,
            alpha_1=alpha_grid,
            color_continuous_scale=px.colors.sequential.Reds,
        )

        assert fig is not None

    def test_reproducibility(self, sample_class_data):
        """Test that results are reproducible."""
        alpha_grid = np.array([0.10, 0.15])
        delta_grid = np.array([0.10])

        df1, _ = sweep_and_plot_parallel_plotly(
            class_data=sample_class_data, delta_0=delta_grid, delta_1=delta_grid, alpha_0=alpha_grid, alpha_1=alpha_grid
        )

        df2, _ = sweep_and_plot_parallel_plotly(
            class_data=sample_class_data, delta_0=delta_grid, delta_1=delta_grid, alpha_0=alpha_grid, alpha_1=alpha_grid
        )

        # DataFrames should be identical
        pd.testing.assert_frame_equal(df1, df2)
