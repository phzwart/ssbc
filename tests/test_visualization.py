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

        return cal_result, pred_stats

    def test_basic_report_verbose(self, sample_data, capsys):
        """Test basic report with verbose=True."""
        cal_result, pred_stats = sample_data

        report_prediction_stats(pred_stats, cal_result, verbose=True)

        # Check that output was printed
        captured = capsys.readouterr()
        assert "PREDICTION SET STATISTICS" in captured.out
        assert "CLASS 0" in captured.out
        assert "CLASS 1" in captured.out
        assert "MARGINAL ANALYSIS" in captured.out

    def test_basic_report_quiet(self, sample_data, capsys):
        """Test basic report with verbose=False."""
        cal_result, pred_stats = sample_data

        summary = report_prediction_stats(pred_stats, cal_result, verbose=False)

        # Check that nothing was printed
        captured = capsys.readouterr()
        assert captured.out == ""

        # But summary should still be returned
        assert isinstance(summary, dict)

    def test_summary_structure(self, sample_data):
        """Test that summary has expected structure."""
        cal_result, pred_stats = sample_data

        summary = report_prediction_stats(pred_stats, cal_result, verbose=False)

        # Should have per-class and marginal sections
        assert 0 in summary
        assert 1 in summary
        assert "marginal" in summary

    def test_per_class_summary(self, sample_data):
        """Test per-class summary structure."""
        cal_result, pred_stats = sample_data

        summary = report_prediction_stats(pred_stats, cal_result, verbose=False)

        for label in [0, 1]:
            if "error" not in summary[label]:
                class_summary = summary[label]

                assert "n" in class_summary
                assert "abstentions" in class_summary
                assert "singletons" in class_summary
                assert "doublets" in class_summary
                assert "pac_bounds" in class_summary

                # Check singletons breakdown
                singletons = class_summary["singletons"]
                assert "correct" in singletons
                assert "incorrect" in singletons
                assert "error_given_singleton" in singletons

    def test_marginal_summary(self, sample_data):
        """Test marginal summary structure."""
        cal_result, pred_stats = sample_data

        summary = report_prediction_stats(pred_stats, cal_result, verbose=False)

        marginal = summary["marginal"]

        assert "n_total" in marginal
        assert "coverage" in marginal
        assert "abstentions" in marginal
        assert "singletons" in marginal
        assert "doublets" in marginal
        assert "pac_bounds" in marginal

    def test_confidence_intervals_present(self, sample_data):
        """Test that all CIs are present."""
        cal_result, pred_stats = sample_data

        summary = report_prediction_stats(pred_stats, cal_result, verbose=False)

        # Check per-class CIs
        for label in [0, 1]:
            if "error" not in summary[label]:
                class_summary = summary[label]

                for key in ["abstentions", "singletons", "doublets"]:
                    assert "ci_95" in class_summary[key]
                    ci = class_summary[key]["ci_95"]
                    assert isinstance(ci, tuple)
                    assert len(ci) == 2

        # Check marginal CIs
        marginal = summary["marginal"]
        assert "ci_95" in marginal["coverage"]
        assert "ci_95" in marginal["abstentions"]
        assert "ci_95" in marginal["singletons"]
        assert "ci_95" in marginal["doublets"]

    def test_singleton_errors_by_predicted_class(self, sample_data):
        """Test singleton errors breakdown by predicted class."""
        cal_result, pred_stats = sample_data

        summary = report_prediction_stats(pred_stats, cal_result, verbose=False)

        marginal = summary["marginal"]
        errors_by_pred = marginal["singletons"]["errors_by_pred"]

        assert "pred_0" in errors_by_pred
        assert "pred_1" in errors_by_pred

        for pred_class in ["pred_0", "pred_1"]:
            err_info = errors_by_pred[pred_class]
            assert "count" in err_info
            assert "denom" in err_info
            assert "rate" in err_info
            assert "ci_95" in err_info

    def test_handles_missing_data_gracefully(self):
        """Test that function handles missing/incomplete data."""
        # Create minimal prediction stats
        pred_stats = {
            0: {"n_class": 0, "error": "No samples"},
            1: {"n_class": 0, "error": "No samples"},
            "marginal": {"n_total": 0},
        }

        cal_result = {0: {"alpha_target": 0.1, "delta": 0.1}, 1: {"alpha_target": 0.1, "delta": 0.1}}

        # Should not raise an error
        summary = report_prediction_stats(pred_stats, cal_result, verbose=False)

        assert isinstance(summary, dict)

    def test_pac_bounds_formatted(self, sample_data, capsys):
        """Test that PAC bounds are formatted in output."""
        cal_result, pred_stats = sample_data

        report_prediction_stats(pred_stats, cal_result, verbose=True)

        captured = capsys.readouterr()

        # Should include PAC-related output
        if "PAC" in captured.out:
            assert "ρ" in captured.out or "rho" in captured.out
            assert "κ" in captured.out or "kappa" in captured.out

    def test_handles_dict_variations(self):
        """Test handling of different dict schema variations."""
        # Test with 'proportion' instead of 'rate'
        pred_stats = {
            0: {
                "n_class": 10,
                "abstentions": {"count": 1, "proportion": 0.1, "lower": 0.0, "upper": 0.3},
                "singletons": {"count": 8, "proportion": 0.8},
                "doublets": {"count": 1, "proportion": 0.1},
                "pac_bounds": {},
            },
            1: {
                "n_class": 10,
                "abstentions": {"count": 1},
                "singletons": {"count": 8},
                "doublets": {"count": 1},
                "pac_bounds": {},
            },
            "marginal": {
                "n_total": 20,
                "coverage": {"count": 18, "rate": 0.9},
                "abstentions": {"count": 2},
                "singletons": {"count": 16, "pred_0": 8, "pred_1": 8, "errors": 2},
                "doublets": {"count": 2},
                "pac_bounds": {},
            },
        }

        cal_result = {0: {"alpha_target": 0.1, "delta": 0.1}, 1: {"alpha_target": 0.1, "delta": 0.1}}

        summary = report_prediction_stats(pred_stats, cal_result, verbose=False)

        assert isinstance(summary, dict)
        assert 0 in summary
        assert 1 in summary
        assert "marginal" in summary


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
