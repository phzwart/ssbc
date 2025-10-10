"""Visualization and reporting utilities for conformal prediction results."""

from typing import Any

from .statistics import cp_interval


def report_prediction_stats(
    prediction_stats: dict[Any, Any], calibration_result: dict[Any, Any], verbose: bool = True
) -> dict[str | int, Any]:
    """Pretty/robust summary for Mondrian conformal prediction stats.

    Tolerates multiple schema shapes:
      - dicts with 'rate'/'ci_95' or 'proportion'/'lower'/'upper'
      - raw ints for counts (e.g., marginal['singletons']['pred_0'] = 339)
      - per-class singleton correct/incorrect either nested under 'singletons'
        OR as top-level aliases 'singletons_correct' / 'singletons_incorrect'.

    Also computes Clopper-Pearson CIs when missing, and splits marginal
    singleton errors by predicted class.

    Parameters
    ----------
    prediction_stats : dict
        Output from mondrian_conformal_calibrate (second return value)
    calibration_result : dict
        Output from mondrian_conformal_calibrate (first return value)
    verbose : bool, default=True
        If True, print detailed statistics to stdout

    Returns
    -------
    dict
        Structured summary with CIs for all metrics, containing:
        - Keys 0, 1 for per-class statistics
        - Key 'marginal' for overall deployment statistics

    Examples
    --------
    >>> # After calibration
    >>> cal_result, pred_stats = mondrian_conformal_calibrate(...)
    >>> summary = report_prediction_stats(pred_stats, cal_result, verbose=True)
    >>> print(summary['marginal']['coverage']['rate'])
    """

    # Helper functions
    def as_dict(x: Any) -> dict[str, Any]:
        """Ensure x is a dict."""
        return x if isinstance(x, dict) else {}

    def get_count(x: Any, default: int = 0) -> int:
        """Extract count from dict or int."""
        if isinstance(x, dict):
            return int(x.get("count", default))
        if isinstance(x, int):
            return int(x)
        return default

    def get_rate(x: Any, default: float | None = 0.0) -> float | None:
        """Extract rate from dict or float."""
        if isinstance(x, dict):
            if "rate" in x:
                return float(x["rate"])
            if "proportion" in x:
                return float(x["proportion"])
            return default
        if isinstance(x, float):
            return float(x)
        return default

    def get_ci_tuple(x: Any) -> tuple[float, float]:
        """Extract CI bounds from dict."""
        if not isinstance(x, dict):
            return (0.0, 0.0)
        if "ci_95" in x and isinstance(x["ci_95"], tuple | list) and len(x["ci_95"]) == 2:
            return float(x["ci_95"][0]), float(x["ci_95"][1])
        lo = x.get("lower", 0.0)
        hi = x.get("upper", 0.0)
        return float(lo), float(hi)

    def ensure_ci(d: dict[str, Any], count: int, total: int) -> tuple[float, float, float]:
        """Return (rate, lo, hi). If d already has rate/CI, use them; else compute CP from count/total."""
        r = get_rate(d, default=None)
        lo, hi = get_ci_tuple(d)
        if r is None or (lo == 0.0 and hi == 0.0 and (count > 0 or total > 0)):
            ci = cp_interval(count, total)
            return ci["proportion"], ci["lower"], ci["upper"]
        return float(r), float(lo), float(hi)

    def pct(x: float) -> str:
        """Format percentage."""
        return f"{x:6.2%}"

    summary: dict[str | int, Any] = {}

    if verbose:
        print("=" * 80)
        print("PREDICTION SET STATISTICS (All rates with 95% Clopper-Pearson CIs)")
        print("=" * 80)

    # ----------------- per-class (conditioned on Y) -----------------
    for class_label in [0, 1]:
        if class_label not in prediction_stats:
            continue
        cls = prediction_stats[class_label]

        if isinstance(cls, dict) and "error" in cls:
            if verbose:
                print(f"\nClass {class_label}: {cls['error']}")
            summary[class_label] = {"error": cls["error"]}
            continue

        n = int(cls.get("n", cls.get("n_class", 0)))
        alpha_target = cls.get("alpha_target", calibration_result.get(class_label, {}).get("alpha_target", None))
        delta = cls.get("delta", calibration_result.get(class_label, {}).get("delta", None))

        abst = as_dict(cls.get("abstentions", {}))
        sing = as_dict(cls.get("singletons", {}))
        # Accept both nested and flat aliases
        sing_corr = as_dict(sing.get("correct", cls.get("singletons_correct", {})))
        sing_inc = as_dict(sing.get("incorrect", cls.get("singletons_incorrect", {})))
        doub = as_dict(cls.get("doublets", {}))
        pac = as_dict(cls.get("pac_bounds", {}))

        # Counts
        abst_count = get_count(abst)
        sing_count = get_count(sing)
        sing_corr_count = get_count(sing_corr)
        sing_inc_count = get_count(sing_inc)
        doub_count = get_count(doub)

        # Ensure rates/CIs (fallback to CP if missing)
        abst_rate, abst_lo, abst_hi = ensure_ci(abst, abst_count, n)
        sing_rate, sing_lo, sing_hi = ensure_ci(sing, sing_count, n)
        sing_corr_rate, sing_corr_lo, sing_corr_hi = ensure_ci(sing_corr, sing_corr_count, n)
        sing_inc_rate, sing_inc_lo, sing_inc_hi = ensure_ci(sing_inc, sing_inc_count, n)
        doub_rate, doub_lo, doub_hi = ensure_ci(doub, doub_count, n)

        # P(error | singleton, Y=class)
        err_given_single_ci = cp_interval(sing_inc_count, sing_count if sing_count > 0 else 1)

        class_summary = {
            "n": n,
            "alpha_target": alpha_target,
            "delta": delta,
            "abstentions": {"count": abst_count, "rate": abst_rate, "ci_95": (abst_lo, abst_hi)},
            "singletons": {
                "count": sing_count,
                "rate": sing_rate,
                "ci_95": (sing_lo, sing_hi),
                "correct": {"count": sing_corr_count, "rate": sing_corr_rate, "ci_95": (sing_corr_lo, sing_corr_hi)},
                "incorrect": {"count": sing_inc_count, "rate": sing_inc_rate, "ci_95": (sing_inc_lo, sing_inc_hi)},
                "error_given_singleton": {
                    "count": sing_inc_count,
                    "denom": sing_count,
                    "rate": err_given_single_ci["proportion"],
                    "ci_95": (err_given_single_ci["lower"], err_given_single_ci["upper"]),
                },
            },
            "doublets": {"count": doub_count, "rate": doub_rate, "ci_95": (doub_lo, doub_hi)},
            "pac_bounds": pac,
        }
        summary[class_label] = class_summary

        if verbose:
            print(f"\n{'=' * 80}")
            print(f"CLASS {class_label} (Conditioned on True Label = {class_label})")
            print(f"{'=' * 80}")
            alpha_str = f"{alpha_target:.3f}" if alpha_target is not None else "n/a"
            delta_str = f"{delta:.3f}" if delta is not None else "n/a"
            print(f"  n={n}, α_target={alpha_str}, δ={delta_str}")

            print("\nPrediction Set Breakdown:")
            print(
                f"  Abstentions:  {abst_count:4d} / {n:4d} = {pct(abst_rate)}  95% CI: [{abst_lo:.4f}, {abst_hi:.4f}]"
            )
            print(
                f"  Singletons:   {sing_count:4d} / {n:4d} = {pct(sing_rate)}  95% CI: [{sing_lo:.4f}, {sing_hi:.4f}]"
            )
            print(
                f"    ├─ Correct:   {sing_corr_count:4d} / {n:4d} = {pct(sing_corr_rate)}  "
                f"95% CI: [{sing_corr_lo:.4f}, {sing_corr_hi:.4f}]"
            )
            print(
                f"    └─ Incorrect: {sing_inc_count:4d} / {n:4d} = {pct(sing_inc_rate)}  "
                f"95% CI: [{sing_inc_lo:.4f}, {sing_inc_hi:.4f}]"
            )

            print(
                f"  Singleton error | Y={class_label}: "
                f"{sing_inc_count:4d} / {sing_count:4d} = {pct(err_given_single_ci['proportion'])}  "
                f"95% CI: [{err_given_single_ci['lower']:.4f}, {err_given_single_ci['upper']:.4f}]"
            )

            print(
                f"\n  Doublets:     {doub_count:4d} / {n:4d} = {pct(doub_rate)}  95% CI: [{doub_lo:.4f}, {doub_hi:.4f}]"
            )

            if pac and pac.get("rho", None) is not None:
                print(f"\n  PAC Singleton Error Rate (δ={delta_str}):")
                print(f"    ρ = {pac.get('rho', 0):.3f}, κ = {pac.get('kappa', 0):.3f}")
                if "alpha_singlet_bound" in pac and "alpha_singlet_observed" in pac:
                    bound = float(pac["alpha_singlet_bound"])
                    observed = float(pac["alpha_singlet_observed"])
                    ok = "✓" if observed <= bound else "✗"
                    print(f"    α'_bound:    {bound:.4f}")
                    print(f"    α'_observed: {observed:.4f} {ok}")

    # ----------------- marginal / deployment view -----------------
    if "marginal" in prediction_stats:
        marg = prediction_stats["marginal"]
        n_total = int(marg["n_total"])

        cov = as_dict(marg.get("coverage", {}))
        abst_m = as_dict(marg.get("abstentions", {}))
        sing_m = as_dict(marg.get("singletons", {}))
        doub_m = as_dict(marg.get("doublets", {}))
        pac_m = as_dict(marg.get("pac_bounds", {}))

        cov_count = get_count(cov)
        abst_m_count = get_count(abst_m)
        sing_total = get_count(sing_m)
        doub_m_count = get_count(doub_m)

        cov_rate, cov_lo, cov_hi = ensure_ci(cov, cov_count, n_total)
        abst_m_rate, abst_m_lo, abst_m_hi = ensure_ci(abst_m, abst_m_count, n_total)
        sing_m_rate, sing_m_lo, sing_m_hi = ensure_ci(sing_m, sing_total, n_total)
        doub_m_rate, doub_m_lo, doub_m_hi = ensure_ci(doub_m, doub_m_count, n_total)

        # pred_0 / pred_1 may be dicts or ints (counts)
        raw_s0 = sing_m.get("pred_0", 0)
        raw_s1 = sing_m.get("pred_1", 0)
        s0_count = get_count(raw_s0)
        s1_count = get_count(raw_s1)

        # Prefer provided rate/CI, else compute off n_total
        if isinstance(raw_s0, dict):
            s0_rate, s0_lo, s0_hi = ensure_ci(raw_s0, s0_count, n_total)
        else:
            s0_ci = cp_interval(s0_count, n_total)
            s0_rate, s0_lo, s0_hi = s0_ci["proportion"], s0_ci["lower"], s0_ci["upper"]

        if isinstance(raw_s1, dict):
            s1_rate, s1_lo, s1_hi = ensure_ci(raw_s1, s1_count, n_total)
        else:
            s1_ci = cp_interval(s1_count, n_total)
            s1_rate, s1_lo, s1_hi = s1_ci["proportion"], s1_ci["lower"], s1_ci["upper"]

        # Overall singleton errors (dict or int). Denominator should be sing_total.
        raw_s_err = sing_m.get("errors", 0)
        s_err_count = get_count(raw_s_err)
        if isinstance(raw_s_err, dict):
            s_err_rate, s_err_lo, s_err_hi = ensure_ci(raw_s_err, s_err_count, sing_total if sing_total > 0 else 1)
        else:
            se_ci = cp_interval(s_err_count, sing_total if sing_total > 0 else 1)
            s_err_rate, s_err_lo, s_err_hi = se_ci["proportion"], se_ci["lower"], se_ci["upper"]

        # Errors by predicted class via per-class incorrect singletons
        # (pred 0 errors happen when Y=1 singleton is wrong; pred 1 errors when Y=0 singleton is wrong)
        err_pred0_count = int(
            prediction_stats.get(1, {})
            .get("singletons", {})
            .get("incorrect", prediction_stats.get(1, {}).get("singletons_incorrect", {}))
            .get("count", 0)
        )
        err_pred1_count = int(
            prediction_stats.get(0, {})
            .get("singletons", {})
            .get("incorrect", prediction_stats.get(0, {}).get("singletons_incorrect", {}))
            .get("count", 0)
        )
        pred0_err_ci = cp_interval(err_pred0_count, s0_count if s0_count > 0 else 1)
        pred1_err_ci = cp_interval(err_pred1_count, s1_count if s1_count > 0 else 1)

        marginal_summary = {
            "n_total": n_total,
            "coverage": {"count": cov_count, "rate": cov_rate, "ci_95": (cov_lo, cov_hi)},
            "abstentions": {"count": abst_m_count, "rate": abst_m_rate, "ci_95": (abst_m_lo, abst_m_hi)},
            "singletons": {
                "count": sing_total,
                "rate": sing_m_rate,
                "ci_95": (sing_m_lo, sing_m_hi),
                "pred_0": {"count": s0_count, "rate": s0_rate, "ci_95": (s0_lo, s0_hi)},
                "pred_1": {"count": s1_count, "rate": s1_rate, "ci_95": (s1_lo, s1_hi)},
                "errors": {"count": s_err_count, "rate": s_err_rate, "ci_95": (s_err_lo, s_err_hi)},
                "errors_by_pred": {
                    "pred_0": {
                        "count": err_pred0_count,
                        "denom": s0_count,
                        "rate": pred0_err_ci["proportion"],
                        "ci_95": (pred0_err_ci["lower"], pred0_err_ci["upper"]),
                    },
                    "pred_1": {
                        "count": err_pred1_count,
                        "denom": s1_count,
                        "rate": pred1_err_ci["proportion"],
                        "ci_95": (pred1_err_ci["lower"], pred1_err_ci["upper"]),
                    },
                },
            },
            "doublets": {"count": doub_m_count, "rate": doub_m_rate, "ci_95": (doub_m_lo, doub_m_hi)},
            "pac_bounds": pac_m,
        }
        summary["marginal"] = marginal_summary

        if verbose:
            print(f"\n{'=' * 80}")
            print("MARGINAL ANALYSIS (Deployment View - Ignores True Labels)")
            print(f"{'=' * 80}")
            print(f"  Total samples: {n_total}")

            print("\nOverall Coverage:")
            print(f"  Covered: {cov_count:4d} / {n_total:4d} = {pct(cov_rate)}  95% CI: [{cov_lo:.4f}, {cov_hi:.4f}]")

            print("\nPrediction Set Distribution:")
            print(
                f"  Abstentions: {abst_m_count:4d} / {n_total:4d} = {pct(abst_m_rate)}  "
                f"95% CI: [{abst_m_lo:.4f}, {abst_m_hi:.4f}]"
            )
            print(
                f"  Singletons:  {sing_total:4d} / {n_total:4d} = {pct(sing_m_rate)}  "
                f"95% CI: [{sing_m_lo:.4f}, {sing_m_hi:.4f}]"
            )
            print(f"    ├─ Pred 0: {s0_count:4d} / {n_total:4d} = {pct(s0_rate)}  95% CI: [{s0_lo:.4f}, {s0_hi:.4f}]")
            print(f"    ├─ Pred 1: {s1_count:4d} / {n_total:4d} = {pct(s1_rate)}  95% CI: [{s1_lo:.4f}, {s1_hi:.4f}]")
            print(
                f"    ├─ Errors (overall): {s_err_count:4d} / {sing_total:4d} = {pct(s_err_rate)}  "
                f"95% CI: [{s_err_lo:.4f}, {s_err_hi:.4f}]"
            )
            print(
                f"    ├─ Pred 0 errors:    {err_pred0_count:4d} / {s0_count:4d} = {pct(pred0_err_ci['proportion'])}  "
                f"95% CI: [{pred0_err_ci['lower']:.4f}, {pred0_err_ci['upper']:.4f}]"
            )
            print(
                f"    └─ Pred 1 errors:    {err_pred1_count:4d} / {s1_count:4d} = {pct(pred1_err_ci['proportion'])}  "
                f"95% CI: [{pred1_err_ci['lower']:.4f}, {pred1_err_ci['upper']:.4f}]"
            )

            print(
                f"  Doublets:    {doub_m_count:4d} / {n_total:4d} = {pct(doub_m_rate)}  "
                f"95% CI: [{doub_m_lo:.4f}, {doub_m_hi:.4f}]"
            )

            if pac_m and pac_m.get("rho", None) is not None:
                aw = pac_m.get("alpha_weighted", None)
                aw_str = f"{float(aw):.3f}" if aw is not None else "n/a"
                print(f"\n  Overall PAC Bounds (weighted α={aw_str}):")
                print(f"    ρ = {pac_m.get('rho', 0):.3f}, κ = {pac_m.get('kappa', 0):.3f}")
                if "alpha_singlet_bound" in pac_m and "alpha_singlet_observed" in pac_m:
                    bound = float(pac_m["alpha_singlet_bound"])
                    observed = float(pac_m["alpha_singlet_observed"])
                    ok = "✓" if observed <= bound else "✗"
                    print(f"    α'_bound:    {bound:.4f}")
                    print(f"    α'_observed: {observed:.4f} {ok}")

                n_escalations = int(pac_m.get("n_escalations", doub_m_count + abst_m_count))
                print("\n  Deployment Decision Mix:")
                print(f"    Automate: {sing_total} singletons ({sing_m_rate:.1%})")
                print(f"    Escalate: {n_escalations} doublets+abstentions ({n_escalations / n_total:.1%})")

    return summary


def plot_parallel_coordinates_plotly(
    df,
    columns: list[str] | None = None,
    color: str = "err_all",
    color_continuous_scale=None,
    title: str = "Mondrian sweep – interactive parallel coordinates",
    height: int = 600,
    base_opacity: float = 0.9,
    unselected_opacity: float = 0.06,
):
    """Create interactive parallel coordinates plot for hyperparameter sweep results.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with hyperparameter sweep results
    columns : list of str, optional
        Columns to display in parallel coordinates
        Default: ['a0','d0','a1','d1','cov','sing_rate','err_all','err_pred0','err_pred1','err_y0','err_y1','esc_rate']
    color : str, default='err_all'
        Column to use for coloring lines
    color_continuous_scale : plotly colorscale, optional
        Color scale for the lines
    title : str, default="Mondrian sweep – interactive parallel coordinates"
        Plot title
    height : int, default=600
        Plot height in pixels
    base_opacity : float, default=0.9
        Opacity of selected lines
    unselected_opacity : float, default=0.06
        Opacity of unselected lines (creates contrast)

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive plotly figure

    Examples
    --------
    >>> import pandas as pd
    >>> df = sweep_hyperparams_and_collect(...)
    >>> fig = plot_parallel_coordinates_plotly(df, color='err_all')
    >>> fig.show()  # In notebook
    >>> # Or save: fig.write_html("sweep_results.html")
    """
    import plotly.express as px

    if columns is None:
        default_cols = [
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
        ]
        columns = [c for c in default_cols if c in df.columns]

    fig = px.parallel_coordinates(
        df,
        dimensions=columns,
        color=color if color in df.columns else None,
        color_continuous_scale=color_continuous_scale or px.colors.sequential.Blugrn,
        labels={c: c for c in columns},
    )

    # Maximize contrast between selected and unselected lines
    if fig.data:
        # Fade unselected lines
        fig.data[0].unselected.update(line=dict(color=f"rgba(1,1,1,{float(unselected_opacity)})"))

    fig.update_layout(
        title=title,
        height=height,
        margin=dict(l=40, r=40, t=60, b=40),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=14),
        uirevision=True,  # keep user brushing across updates
    )

    # Make axis labels and ranges more readable
    fig.update_traces(labelfont=dict(size=14), rangefont=dict(size=12), tickfont=dict(size=12))

    # Optional: title for colorbar if we're coloring by a column
    if color in df.columns and fig.data and getattr(fig.data[0], "line", None):
        if getattr(fig.data[0].line, "colorbar", None) is not None:
            fig.data[0].line.colorbar.title = color

    return fig
