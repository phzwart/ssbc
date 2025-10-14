"""Tests for the visualization module."""

import pandas as pd
import pytest

from ssbc.conformal import mondrian_conformal_calibrate, split_by_class
from ssbc.simulation import BinaryClassifierSimulator
from ssbc.visualization import plot_parallel_coordinates_plotly, report_prediction_stats


class TestReportPredictionStats:
    """Test report_prediction_stats function."""

    @pytest.fixture
    def sample_data(self):
        """Create sample calibration and prediction stats."""
        sim = BinaryClassifierSimulator(p_class1=0.5, beta_params_class0=(2, 8), beta_params_class1=(8, 2), seed=42)
        labels, probs = sim.generate(n_samples=100)
        class_data = split_by_class(labels, probs)

        cal_result, pred_stats = mondrian_conformal_calibrate(
            class_data=class_data, alpha_target=0.10, delta=0.10, mode="beta"
        )

        return cal_result, pred_stats, labels, probs

    def test_basic_report_verbose(self, sample_data, capsys):
        """Test basic report with verbose=True."""
        cal_result, pred_stats, _, _ = sample_data

        report_prediction_stats(pred_stats, cal_result, verbose=True)

        # Check that output was printed
        captured = capsys.readouterr()
        assert "MONDRIAN CONFORMAL PREDICTION REPORT" in captured.out
        assert "CLASS 0" in captured.out
        assert "CLASS 1" in captured.out

    def test_basic_report_quiet(self, sample_data, capsys):
        """Test basic report with verbose=False."""
        cal_result, pred_stats, _, _ = sample_data

        summary = report_prediction_stats(pred_stats, cal_result, verbose=False)

        # Check that nothing was printed
        captured = capsys.readouterr()
        assert captured.out == ""

        # But summary should still be returned
        assert isinstance(summary, dict)

    def test_summary_structure(self, sample_data):
        """Test that summary has expected structure."""
        cal_result, pred_stats, _, _ = sample_data

        summary = report_prediction_stats(pred_stats, cal_result, verbose=False)

        # Should have per-class sections
        assert 0 in summary
        assert 1 in summary

    # Note: Tests for operational bounds moved to test_rigorous_report.py
    # The old compute_mondrian_operational_bounds and compute_marginal_operational_bounds
    # have been removed in favor of the unified generate_rigorous_pac_report() workflow

    def test_handles_missing_data_gracefully(self):
        """Test that function handles missing/incomplete data."""
        # Create minimal prediction stats
        pred_stats = {
            0: {"n_class": 0, "error": "No samples"},
            1: {"n_class": 0, "error": "No samples"},
        }

        cal_result = {0: {"alpha_target": 0.1, "delta": 0.1}, 1: {"alpha_target": 0.1, "delta": 0.1}}

        # Should not raise an error
        summary = report_prediction_stats(pred_stats, cal_result, verbose=False)

        assert isinstance(summary, dict)


class TestPlotParallelCoordinatesPlotly:
    """Test plot_parallel_coordinates_plotly function."""

    @pytest.fixture
    def sample_dataframe(self):
        """Create sample dataframe for plotting."""
        return pd.DataFrame(
            {
                "a0": [0.05, 0.05, 0.10, 0.10],
                "d0": [0.05, 0.10, 0.05, 0.10],
                "a1": [0.05, 0.10, 0.05, 0.10],
                "d1": [0.05, 0.05, 0.10, 0.10],
                "cov": [0.95, 0.94, 0.93, 0.92],
                "sing_rate": [0.80, 0.75, 0.70, 0.65],
                "err_all": [0.05, 0.06, 0.07, 0.08],
                "esc_rate": [0.15, 0.19, 0.23, 0.27],
            }
        )

    def test_creates_figure(self, sample_dataframe):
        """Test that function creates a plotly figure."""
        fig = plot_parallel_coordinates_plotly(sample_dataframe)

        assert fig is not None
        assert hasattr(fig, "data")
        assert hasattr(fig, "layout")

    def test_default_columns(self, sample_dataframe):
        """Test with default column selection."""
        fig = plot_parallel_coordinates_plotly(sample_dataframe)

        # Should have created a figure
        assert len(fig.data) > 0

    def test_custom_columns(self, sample_dataframe):
        """Test with custom column selection."""
        columns = ["a0", "a1", "cov", "err_all"]
        fig = plot_parallel_coordinates_plotly(sample_dataframe, columns=columns)

        assert fig is not None

    def test_custom_color(self, sample_dataframe):
        """Test with custom color column."""
        fig = plot_parallel_coordinates_plotly(sample_dataframe, color="cov")

        assert fig is not None

    def test_custom_title(self, sample_dataframe):
        """Test with custom title."""
        custom_title = "Test Plot Title"
        fig = plot_parallel_coordinates_plotly(sample_dataframe, title=custom_title)

        assert fig.layout.title.text == custom_title

    def test_custom_height(self, sample_dataframe):
        """Test with custom height."""
        custom_height = 800
        fig = plot_parallel_coordinates_plotly(sample_dataframe, height=custom_height)

        assert fig.layout.height == custom_height

    def test_handles_missing_columns(self, sample_dataframe):
        """Test that plotly raises error for missing columns."""
        columns = ["a0", "nonexistent_column", "cov"]

        # Plotly will raise ValueError for nonexistent columns
        with pytest.raises(ValueError, match="not the name of a column"):
            plot_parallel_coordinates_plotly(sample_dataframe, columns=columns)

    def test_empty_dataframe(self):
        """Test with empty dataframe."""
        df = pd.DataFrame()

        # Should still create a figure (may be empty)
        fig = plot_parallel_coordinates_plotly(df)

        assert fig is not None

    def test_single_row_dataframe(self):
        """Test with single row dataframe."""
        df = pd.DataFrame({"a0": [0.05], "a1": [0.05], "cov": [0.95], "err_all": [0.05]})

        fig = plot_parallel_coordinates_plotly(df)

        assert fig is not None

    def test_color_column_not_in_df(self, sample_dataframe):
        """Test when color column doesn't exist."""
        # Should handle gracefully
        fig = plot_parallel_coordinates_plotly(sample_dataframe, color="nonexistent")

        assert fig is not None

    def test_custom_colorscale(self, sample_dataframe):
        """Test with custom color scale."""
        import plotly.express as px

        fig = plot_parallel_coordinates_plotly(
            sample_dataframe, color="err_all", color_continuous_scale=px.colors.sequential.Reds
        )

        assert fig is not None

    def test_opacity_parameters(self, sample_dataframe):
        """Test opacity parameters."""
        fig = plot_parallel_coordinates_plotly(sample_dataframe, base_opacity=0.8, unselected_opacity=0.1)

        assert fig is not None

        # Check that unselected lines have low opacity
        if fig.data:
            unselected_line = fig.data[0].unselected.line
            assert "color" in unselected_line
