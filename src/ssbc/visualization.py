"""Visualization and reporting utilities for conformal prediction results."""

from typing import Any

from .statistics import cp_interval


def compute_conditional_rate_bounds(
    numerator_fold_results: list[dict],
    denominator_fold_results: list[dict],
    weights: list[float],
) -> tuple[float, float, float]:
    """Compute bounds on conditional rate from fold-level counts.

    For conditional rate P(A | B), computes cross-validated bounds by:
    1. Computing A_count / B_count in each fold
    2. Using Clopper-Pearson on aggregated counts

    Parameters
    ----------
    numerator_fold_results : list[dict]
        Fold results for numerator event (e.g., correct_in_singleton)
    denominator_fold_results : list[dict]
        Fold results for denominator event (e.g., singleton)
    weights : list[float]
        Fold weights

    Returns
    -------
    rate : float
        Estimated conditional rate
    lower : float
        Lower CI bound
    upper : float
        Upper CI bound
    """
    # Aggregate counts across folds
    total_numerator = sum(fold["K_fr"] for fold in numerator_fold_results)
    total_denominator = sum(fold["K_fr"] for fold in denominator_fold_results)

    if total_denominator == 0:
        return 0.0, 0.0, 1.0

    # Compute CP interval on aggregated counts
    ci = cp_interval(total_numerator, total_denominator)
    return ci["proportion"], ci["lower"], ci["upper"]


def report_prediction_stats(
    prediction_stats: dict[Any, Any],
    calibration_result: dict[Any, Any],
    operational_bounds_per_class: dict[int, Any] | None = None,
    marginal_operational_bounds: Any | None = None,
    verbose: bool = True,
) -> dict[str | int, Any]:
    """Report rigorous statistics for Mondrian conformal prediction with valid CIs.

    Only displays statistics with valid confidence intervals:
    - Per-class statistics from calibration data (valid within class)
    - Per-class operational bounds from cross-validation (rigorous PAC bounds)
    - Marginal operational bounds from cross-validated Mondrian (rigorous PAC bounds)

    Does NOT display marginal statistics from calibration data (invalid CIs for Mondrian).

    Parameters
    ----------
    prediction_stats : dict
        Output from mondrian_conformal_calibrate (second return value)
    calibration_result : dict
        Output from mondrian_conformal_calibrate (first return value)
    operational_bounds_per_class : dict[int, OperationalRateBoundsResult], optional
        Per-class operational bounds from compute_mondrian_operational_bounds
    marginal_operational_bounds : OperationalRateBoundsResult, optional
        Marginal operational bounds from compute_marginal_operational_bounds
    verbose : bool, default=True
        If True, print detailed statistics to stdout

    Returns
    -------
    dict
        Structured summary with valid CIs:
        - Keys 0, 1 for per-class statistics
        - Key 'marginal_bounds' if marginal_operational_bounds provided

    Examples
    --------
    >>> # Basic: Only calibration data (limited info)
    >>> cal_result, pred_stats = mondrian_conformal_calibrate(...)
    >>> summary = report_prediction_stats(pred_stats, cal_result)
    >>>
    >>> # With per-class operational bounds (rigorous)
    >>> op_bounds = compute_mondrian_operational_bounds(cal_result, class_data, delta=0.05)
    >>> summary = report_prediction_stats(pred_stats, cal_result, op_bounds)
    >>>
    >>> # With marginal bounds too (complete SLA)
    >>> marginal = compute_marginal_operational_bounds(labels, probs, 0.1, 0.05, 0.05)
    >>> summary = report_prediction_stats(pred_stats, cal_result, op_bounds, marginal)
    """
    from .statistics import cp_interval

    summary: dict[str | int, Any] = {}

    if verbose:
        print("=" * 80)
        print("MONDRIAN CONFORMAL PREDICTION REPORT")
        print("=" * 80)

    # ==================== PER-CLASS STATISTICS ====================
    for class_label in sorted([k for k in prediction_stats.keys() if isinstance(k, int)]):
        cls = prediction_stats[class_label]

        if isinstance(cls, dict) and "error" in cls:
            if verbose:
                print(f"\nCLASS {class_label}: {cls['error']}")
            summary[class_label] = {"error": cls["error"]}
            continue

        n = int(cls.get("n", cls.get("n_class", 0)))
        if n == 0:
            continue

        # Get calibration info
        cal = calibration_result.get(class_label, {})
        alpha_target = cal.get("alpha_target")
        alpha_corrected = cal.get("alpha_corrected")
        delta = cal.get("delta")
        threshold = cal.get("threshold")

        if verbose:
            print(f"\n{'=' * 80}")
            print(f"CLASS {class_label} (Conditioned on True Label = {class_label})")
            print(f"{'=' * 80}")
            print(f"  Calibration size: n = {n}")
            if alpha_target is not None:
                print(f"  Target miscoverage: α = {alpha_target:.3f}")
            if alpha_corrected is not None:
                print(f"  SSBC-corrected α:   α' = {alpha_corrected:.4f}")
            if delta is not None:
                print(f"  PAC risk:           δ = {delta:.3f}")
            if threshold is not None:
                print(f"  Conformal threshold: {threshold:.4f}")

        # Per-class stats from calibration data (VALID - exchangeable within class)
        if verbose:
            print(f"\n  📊 Statistics from Calibration Data (n={n}):")
            print("     [Basic CP CIs without PAC guarantee - evaluated on calibration data]")

        # Abstentions
        abstentions = cls.get("abstentions", {})
        if isinstance(abstentions, dict):
            abst_count = abstentions.get("count", 0)
            abst_ci = cp_interval(abst_count, n)
            if verbose:
                print(
                    f"    Abstentions:  {abst_count:4d} / {n:4d} = {abst_ci['proportion']:6.2%}  "
                    f"95% CI: [{abst_ci['lower']:.3f}, {abst_ci['upper']:.3f}]"
                )

        # Singletons (note: singletons_correct/incorrect are at top level, not nested)
        singletons = cls.get("singletons", {})
        singletons_correct = cls.get("singletons_correct", {})
        singletons_incorrect = cls.get("singletons_incorrect", {})

        if isinstance(singletons, dict):
            sing_count = singletons.get("count", 0)
            sing_correct = singletons_correct.get("count", 0) if isinstance(singletons_correct, dict) else 0
            sing_incorrect = singletons_incorrect.get("count", 0) if isinstance(singletons_incorrect, dict) else 0

            # Compute valid CIs (exchangeable within class)
            sing_ci = cp_interval(sing_count, n)
            sing_corr_ci = cp_interval(sing_correct, n)
            sing_inc_ci = cp_interval(sing_incorrect, n)

            if verbose:
                print(
                    f"    Singletons:   {sing_count:4d} / {n:4d} = {sing_ci['proportion']:6.2%}  "
                    f"95% CI: [{sing_ci['lower']:.3f}, {sing_ci['upper']:.3f}]"
                )
                print(
                    f"      Correct:    {sing_correct:4d} / {n:4d} = {sing_corr_ci['proportion']:6.2%}  "
                    f"95% CI: [{sing_corr_ci['lower']:.3f}, {sing_corr_ci['upper']:.3f}]"
                )
                print(
                    f"      Incorrect:  {sing_incorrect:4d} / {n:4d} = {sing_inc_ci['proportion']:6.2%}  "
                    f"95% CI: [{sing_inc_ci['lower']:.3f}, {sing_inc_ci['upper']:.3f}]"
                )

                # Error rate given singleton
                if sing_count > 0:
                    err_given_sing = cp_interval(sing_incorrect, sing_count)
                    print(
                        f"    Error | singleton: {sing_incorrect:4d} / {sing_count:4d} = "
                        f"{err_given_sing['proportion']:6.2%}  "
                        f"95% CI: [{err_given_sing['lower']:.3f}, {err_given_sing['upper']:.3f}]"
                    )

        # Doublets
        doublets = cls.get("doublets", {})
        if isinstance(doublets, dict):
            doub_count = doublets.get("count", 0)
            doub_ci = cp_interval(doub_count, n)
            if verbose:
                print(
                    f"    Doublets:     {doub_count:4d} / {n:4d} = {doub_ci['proportion']:6.2%}  "
                    f"95% CI: [{doub_ci['lower']:.3f}, {doub_ci['upper']:.3f}]"
                )

        # PAC bounds (ρ, κ, α'_bound) - important theoretical guarantees
        pac_bounds = cls.get("pac_bounds", {})
        if isinstance(pac_bounds, dict) and pac_bounds.get("rho") is not None:
            if verbose:
                print(f"\n  📐 PAC Singleton Error Bound (δ={delta:.3f}):")
                print(f"     ρ = {pac_bounds.get('rho', 0):.3f}, κ = {pac_bounds.get('kappa', 0):.3f}")
                if "alpha_singlet_bound" in pac_bounds and "alpha_singlet_observed" in pac_bounds:
                    bound = float(pac_bounds["alpha_singlet_bound"])
                    observed = float(pac_bounds["alpha_singlet_observed"])
                    ok = "✓" if observed <= bound else "✗"
                    print(f"     α'_bound:    {bound:.4f}")
                    print(f"     α'_observed: {observed:.4f} {ok}")

        # Operational bounds (RIGOROUS - cross-validated with PAC guarantees)
        if operational_bounds_per_class and class_label in operational_bounds_per_class:
            op_bounds = operational_bounds_per_class[class_label]

            if verbose:
                print("\n  ✅ RIGOROUS Operational Bounds (LOO-CV)")
                print(f"     PAC confidence: {op_bounds.rate_confidence:.1%}")
                print(f"     Calibration size: n = {op_bounds.n_calibration}")

            # Show main rates (singleton, doublet, abstention)
            for rate_name in ["abstention", "singleton", "doublet"]:
                if rate_name in op_bounds.rate_bounds:
                    bounds = op_bounds.rate_bounds[rate_name]
                    if verbose:
                        print(f"\n     {rate_name.upper()}:")
                        print(f"       Bounds: [{bounds.lower_bound:.3f}, {bounds.upper_bound:.3f}]")
                        print(f"       Count: {bounds.n_successes}/{bounds.n_evaluations}")

            # Show conditional singleton rates (conditional on having a singleton)
            has_correct = "correct_in_singleton" in op_bounds.rate_bounds
            has_error = "error_in_singleton" in op_bounds.rate_bounds
            has_singleton = "singleton" in op_bounds.rate_bounds

            if verbose and (has_correct or has_error) and has_singleton:
                print("\n     CONDITIONAL RATES (conditioned on singleton, with CP+PAC bounds):")

                singleton_bounds = op_bounds.rate_bounds["singleton"]
                n_singletons = singleton_bounds.n_successes

                # P(correct | singleton) with rigorous CP bounds
                if has_correct and n_singletons > 0:
                    correct_bounds = op_bounds.rate_bounds["correct_in_singleton"]
                    n_correct = correct_bounds.n_successes

                    # Conditional rate and CP interval
                    rate = n_correct / n_singletons if n_singletons > 0 else 0.0
                    ci = cp_interval(n_correct, n_singletons)

                    print(f"       P(correct | singleton) = {rate:.3f}  95% CI: [{ci['lower']:.3f}, {ci['upper']:.3f}]")

                # P(error | singleton) with rigorous CP bounds
                if has_error and n_singletons > 0:
                    error_bounds = op_bounds.rate_bounds["error_in_singleton"]
                    n_error = error_bounds.n_successes

                    # Conditional rate and CP interval
                    rate = n_error / n_singletons if n_singletons > 0 else 0.0
                    ci = cp_interval(n_error, n_singletons)

                    print(f"       P(error | singleton)   = {rate:.3f}  95% CI: [{ci['lower']:.3f}, {ci['upper']:.3f}]")

        # Store in summary
        summary[class_label] = {
            "n": n,
            "alpha_target": alpha_target,
            "alpha_corrected": alpha_corrected,
            "threshold": threshold,
            "calibration_stats": {
                "abstentions": abstentions,
                "singletons": singletons,
                "doublets": doublets,
            },
            "pac_bounds": pac_bounds,
        }
        if operational_bounds_per_class and class_label in operational_bounds_per_class:
            summary[class_label]["operational_bounds"] = operational_bounds_per_class[class_label]

    # ==================== MARGINAL STATISTICS ====================
    if marginal_operational_bounds is not None:
        if verbose:
            print(f"\n{'=' * 80}")
            print("MARGINAL STATISTICS (Deployment View - Ignores True Labels)")
            print(f"{'=' * 80}")
            print(f"  Total samples: n = {marginal_operational_bounds.n_calibration}")

            print("\n  ✅ RIGOROUS Marginal Bounds (LOO-CV)")
            print(f"     PAC confidence: {marginal_operational_bounds.rate_confidence:.1%}")
            print(f"     Total evaluations: n = {marginal_operational_bounds.n_calibration}")

        # Show main rates
        for rate_name in ["abstention", "singleton", "doublet"]:
            if rate_name in marginal_operational_bounds.rate_bounds:
                bounds = marginal_operational_bounds.rate_bounds[rate_name]
                if verbose:
                    print(f"\n     {rate_name.upper()}:")
                    print(f"       Bounds: [{bounds.lower_bound:.3f}, {bounds.upper_bound:.3f}]")
                    print(f"       Count: {bounds.n_successes}/{bounds.n_evaluations}")

        # Show conditional singleton rates (marginal)
        has_correct = "correct_in_singleton" in marginal_operational_bounds.rate_bounds
        has_error = "error_in_singleton" in marginal_operational_bounds.rate_bounds
        has_singleton = "singleton" in marginal_operational_bounds.rate_bounds

        if verbose and (has_correct or has_error) and has_singleton:
            print("\n     CONDITIONAL RATES (conditioned on singleton, with CP+PAC bounds):")

            singleton_bounds = marginal_operational_bounds.rate_bounds["singleton"]
            n_singletons = singleton_bounds.n_successes

            if has_correct and n_singletons > 0:
                correct_bounds = marginal_operational_bounds.rate_bounds["correct_in_singleton"]
                n_correct = correct_bounds.n_successes

                # Conditional rate and CP interval
                rate = n_correct / n_singletons if n_singletons > 0 else 0.0
                ci = cp_interval(n_correct, n_singletons)

                print(f"       P(correct | singleton) = {rate:.3f}  95% CI: [{ci['lower']:.3f}, {ci['upper']:.3f}]")

            if has_error and n_singletons > 0:
                error_bounds = marginal_operational_bounds.rate_bounds["error_in_singleton"]
                n_error = error_bounds.n_successes

                # Conditional rate and CP interval
                rate = n_error / n_singletons if n_singletons > 0 else 0.0
                ci = cp_interval(n_error, n_singletons)

                print(f"       P(error | singleton)   = {rate:.3f}  95% CI: [{ci['lower']:.3f}, {ci['upper']:.3f}]")

        summary["marginal_bounds"] = marginal_operational_bounds

        if verbose:
            # Deployment interpretation
            sing_bounds = marginal_operational_bounds.rate_bounds.get("singleton")
            doub_bounds = marginal_operational_bounds.rate_bounds.get("doublet")
            abst_bounds = marginal_operational_bounds.rate_bounds.get("abstention")

            if sing_bounds:
                print("\n  📈 Deployment Expectations:")
                print(
                    f"     Automation (singletons): "
                    f"{sing_bounds.lower_bound:.1%} - {sing_bounds.upper_bound:.1%} of cases"
                )

                # Escalation = doublets + abstentions
                if doub_bounds and abst_bounds:
                    esc_lower = doub_bounds.lower_bound + abst_bounds.lower_bound
                    esc_upper = doub_bounds.upper_bound + abst_bounds.upper_bound
                    print(f"     Escalation (doublets+abstentions): {esc_lower:.1%} - {esc_upper:.1%} of cases")
                elif doub_bounds:
                    print(
                        f"     Escalation (doublets): "
                        f"{doub_bounds.lower_bound:.1%} - {doub_bounds.upper_bound:.1%} of cases"
                    )

    # ==================== WARNINGS ====================
    if verbose:
        print(f"\n{'=' * 80}")
        print("NOTES")
        print(f"{'=' * 80}")
        print("\n✓ Per-class CIs are valid (Clopper-Pearson, exchangeable within class)")

        if operational_bounds_per_class or marginal_operational_bounds:
            print("✓ Operational bounds have PAC guarantees via cross-validation")
        else:
            print("\n⚠️  For rigorous deployment bounds, run:")
            print("   - compute_mondrian_operational_bounds() for per-class bounds")
            print("   - compute_marginal_operational_bounds() for marginal bounds")

        if prediction_stats.get("marginal") and marginal_operational_bounds is None:
            print("\n⚠️  Marginal stats from calibration data NOT shown (invalid CIs for Mondrian)")
            print("   Use compute_marginal_operational_bounds() for valid marginal bounds")

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
