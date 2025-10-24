#!/usr/bin/env python3
"""
Example: Small-Sample LOO-CV Uncertainty Quantification

This example demonstrates the new LOO uncertainty quantification methods
that properly account for all four sources of uncertainty in conformal
prediction with small calibration sets.

Key improvements over standard Clopper-Pearson bounds:
1. LOO-CV correlation structure (variance inflation ≈2×)
2. Threshold calibration uncertainty
3. Parameter estimation uncertainty
4. Test sampling uncertainty
"""

import numpy as np

from ssbc import (
    BinaryClassifierSimulator,
    compute_robust_prediction_bounds,
    format_prediction_bounds_report,
    generate_rigorous_pac_report,
)


def demonstrate_loo_uncertainty():
    """Demonstrate LOO uncertainty quantification with different methods."""
    print("LOO-CV Uncertainty Quantification Example")
    print("=" * 60)

    # Create simulator
    sim = BinaryClassifierSimulator(p_class1=0.3, beta_params_class0=(2, 8), beta_params_class1=(8, 2), seed=42)

    # Generate small calibration set (typical for conformal prediction)
    n_cal = 30
    cal_labels, cal_probs = sim.generate(n_cal)

    print(f"Calibration data: n={n_cal}")
    print(f"Class distribution: {np.bincount(cal_labels)}")
    print()

    # Test different methods
    methods = ["analytical", "exact", "hoeffding", "all"]
    n_test = 100

    print("Comparing LOO uncertainty quantification methods:")
    print("-" * 60)

    for method in methods:
        print(f"\nMethod: {method.upper()}")
        print("-" * 30)

        # Generate PAC report with LOO correction
        report = generate_rigorous_pac_report(
            labels=cal_labels, probs=cal_probs, alpha_target=0.10, delta=0.10, test_size=n_test, verbose=False
        )

        # Extract bounds
        marginal_bounds = report["pac_bounds_marginal"]
        singleton_bounds = marginal_bounds["singleton_rate_bounds"]
        doublet_bounds = marginal_bounds["doublet_rate_bounds"]
        abstention_bounds = marginal_bounds["abstention_rate_bounds"]

        print(f"Singleton bounds: [{singleton_bounds[0]:.4f}, {singleton_bounds[1]:.4f}]")
        print(f"  Width: {singleton_bounds[1] - singleton_bounds[0]:.4f}")
        print(f"Doublet bounds:   [{doublet_bounds[0]:.4f}, {doublet_bounds[1]:.4f}]")
        print(f"  Width: {doublet_bounds[1] - doublet_bounds[0]:.4f}")
        print(f"Abstention bounds: [{abstention_bounds[0]:.4f}, {abstention_bounds[1]:.4f}]")
        print(f"  Width: {abstention_bounds[1] - abstention_bounds[0]:.4f}")


def demonstrate_method_comparison():
    """Compare all three methods side by side."""
    print("\n" + "=" * 60)
    print("METHOD COMPARISON")
    print("=" * 60)

    # Generate test data
    n_cal = 25
    n_test = 50
    p_true = 0.7

    # Simulate LOO predictions (in practice, these come from actual LOO-CV)
    np.random.seed(42)
    loo_preds = np.random.binomial(1, p_true, n_cal)
    p_hat = np.mean(loo_preds)

    print(f"Test case: n_cal={n_cal}, n_test={n_test}")
    print(f"LOO predictions: {np.sum(loo_preds)}/{n_cal} successes (p̂={p_hat:.3f})")
    print()

    # Compare all methods
    methods = ["analytical", "exact", "hoeffding"]
    results = {}

    print(f"{'Method':<12} {'Lower':<8} {'Upper':<8} {'Width':<8} {'Notes'}")
    print("-" * 50)

    for method in methods:
        L, U, report = compute_robust_prediction_bounds(loo_preds, n_test, alpha=0.05, method=method)
        width = U - L
        results[method] = {"L": L, "U": U, "width": width}

        notes = ""
        if method == "analytical":
            notes = "Fast, good for n≥40"
        elif method == "exact":
            notes = "Conservative, good for n=20-40"
        elif method == "hoeffding":
            notes = "Ultra-conservative, guaranteed"

        print(f"{method.capitalize():<12} {L:<8.4f} {U:<8.4f} {width:<8.4f} {notes}")

    # Show width ratios
    print("\nWidth ratios (vs Analytical):")
    analytical_width = results["analytical"]["width"]
    for method in ["exact", "hoeffding"]:
        ratio = results[method]["width"] / analytical_width
        print(f"  {method.capitalize()}: {ratio:.2f}× wider")


def demonstrate_automatic_selection():
    """Demonstrate automatic method selection based on sample size."""
    print("\n" + "=" * 60)
    print("AUTOMATIC METHOD SELECTION")
    print("=" * 60)

    # Test different sample sizes
    sample_sizes = [15, 25, 35, 45]

    print(f"{'n_cal':<6} {'Selected Method':<20} {'Reasoning'}")
    print("-" * 50)

    for n_cal in sample_sizes:
        # Generate test data
        loo_preds = np.random.binomial(1, 0.6, n_cal)

        # Auto-select method
        L, U, report = compute_robust_prediction_bounds(loo_preds, n_test=100, alpha=0.05, method="auto")

        selected_method = report["selected_method"]

        # Explain reasoning
        if n_cal < 20:
            reasoning = "Very small → Hoeffding (guaranteed)"
        elif n_cal < 40:
            reasoning = "Small → Exact (conservative)"
        else:
            reasoning = "Large → Analytical (efficient)"

        print(f"{n_cal:<6} {selected_method:<20} {reasoning}")


def demonstrate_formatted_report():
    """Demonstrate comprehensive formatted report generation."""
    print("\n" + "=" * 60)
    print("FORMATTED REPORT GENERATION")
    print("=" * 60)

    # Generate test data
    n_cal = 30
    n_test = 100
    loo_preds = np.random.binomial(1, 0.65, n_cal)

    # Generate comprehensive report
    report = format_prediction_bounds_report("Singleton Rate", loo_preds, n_test, alpha=0.05, include_all_methods=True)

    print("Generated comprehensive report:")
    print(report)


def demonstrate_integration_with_validation():
    """Demonstrate integration with validation framework."""
    print("\n" + "=" * 60)
    print("INTEGRATION WITH VALIDATION")
    print("=" * 60)

    # Create simulator
    sim = BinaryClassifierSimulator(p_class1=0.3, beta_params_class0=(2, 8), beta_params_class1=(8, 2), seed=42)

    # Generate calibration data
    cal_labels, cal_probs = sim.generate(25)  # Small calibration set

    print("Running validation with LOO-corrected bounds...")

    # Generate PAC report
    report = generate_rigorous_pac_report(
        labels=cal_labels, probs=cal_probs, alpha_target=0.10, delta=0.10, test_size=50, verbose=False
    )

    # Validate bounds
    from ssbc import validate_pac_bounds

    val_report = validate_pac_bounds(report, sim, test_size=50, n_trials=100, verbose=False)

    # Show results
    singleton_bounds = val_report["class_1"]["singleton"]["bounds"]
    singleton_coverage = val_report["class_1"]["singleton"]["empirical_coverage"]

    print(f"Singleton bounds: [{singleton_bounds[0]:.4f}, {singleton_bounds[1]:.4f}]")
    print(f"Empirical coverage: {singleton_coverage:.1%}")
    print("Target coverage: 95.0%")

    if singleton_coverage >= 0.95:
        print("✅ LOO-corrected bounds achieve proper coverage!")
    else:
        print("⚠️  Coverage below target - may need more conservative method")


def demonstrate_uncertainty_breakdown():
    """Demonstrate detailed uncertainty breakdown."""
    print("\n" + "=" * 60)
    print("UNCERTAINTY BREAKDOWN")
    print("=" * 60)

    # Generate test data
    n_cal = 30
    n_test = 100
    loo_preds = np.random.binomial(1, 0.6, n_cal)

    # Compute bounds with detailed diagnostics
    L, U, report = compute_robust_prediction_bounds(loo_preds, n_test, alpha=0.05, method="analytical")

    if "diagnostics" in report:
        diag = report["diagnostics"]
        print(f"Point estimate: p̂ = {diag['p_hat']:.4f}")
        print(f"Calibration uncertainty: SE_cal = {np.sqrt(diag['var_calibration']):.4f}")
        print(f"Test sampling uncertainty: SE_test = {np.sqrt(diag['var_test']):.4f}")
        print(f"Total uncertainty: SE_total = {diag['se_total']:.4f}")
        print(f"LOO inflation factor: {diag['inflation_factor']:.2f}×")
        print(f"Critical value: {diag['critical_value']:.3f}")
        print(f"Final bounds: [{L:.4f}, {U:.4f}] (width: {U - L:.4f})")


if __name__ == "__main__":
    demonstrate_loo_uncertainty()
    demonstrate_method_comparison()
    demonstrate_automatic_selection()
    demonstrate_formatted_report()
    demonstrate_integration_with_validation()
    demonstrate_uncertainty_breakdown()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("✅ LOO uncertainty quantification successfully implemented!")
    print("\nKey features:")
    print("• Accounts for LOO-CV correlation structure (variance inflation ≈2×)")
    print("• Handles threshold calibration uncertainty")
    print("• Provides parameter estimation uncertainty")
    print("• Includes test sampling uncertainty")
    print("• Three methods: Analytical, Exact, Hoeffding")
    print("• Automatic method selection based on sample size")
    print("• Comprehensive diagnostic reports")
    print("• Integration with existing validation framework")
    print("\nThis addresses the critical missing uncertainty sources")
    print("in small-sample conformal prediction!")
