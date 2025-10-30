"""Complete rigorous PAC report workflow.

Demonstrates the comprehensive rigorous_pac_report with:
1. PAC Bounds (LOO-CV + Prediction Bounds) - Always included
2. Method comparison (analytical, exact, Hoeffding) - Optional
3. Class-conditional error metrics - Always included for marginal scope

All from a single function call!
"""

import numpy as np

from ssbc import BinaryClassifierSimulator, generate_rigorous_pac_report

# Create simulator
sim = BinaryClassifierSimulator(p_class1=0.20, beta_params_class0=(1, 7), beta_params_class1=(5, 2), seed=42)

# Generate calibration data
labels, probs = sim.generate(n_samples=100)

print("=" * 80)
print("RIGOROUS PAC REPORT - COMPLETE WORKFLOW")
print("=" * 80)
print(f"\nCalibration: {len(labels)} samples ({np.sum(labels == 0)} class 0, {np.sum(labels == 1)} class 1)")
print("\n✓ Generating comprehensive report with LOO-corrected bounds and method comparison...")

# Generate complete report with method comparison
report = generate_rigorous_pac_report(
    labels=labels,
    probs=probs,
    alpha_target=0.10,
    delta=0.10,
    test_size=1000,
    ci_level=0.95,
    use_union_bound=True,
    n_jobs=-1,
    verbose=True,
    prediction_method="all",  # Compare analytical, exact, and Hoeffding methods
    use_loo_correction=True,
)

print("\n" + "=" * 80)
print("COMPARATIVE ANALYSIS")
print("=" * 80)

# Extract results for comparison
pac_class_0 = report["pac_bounds_class_0"]
pac_marginal = report["pac_bounds_marginal"]

print("\nClass 0 Singleton Rate Summary:")
print("-" * 80)

pac_bounds = pac_class_0["singleton_rate_bounds"]
print(f"\n1. PAC Bounds (LOO-CV + CP): [{pac_bounds[0]:.3f}, {pac_bounds[1]:.3f}]")
print(f"   Expected: {pac_class_0['expected_singleton_rate']:.3f}")
print("   → Rigorous deployment guarantees")

# Show method comparison if available
if "loo_diagnostics" in pac_class_0:
    singleton_diag = pac_class_0["loo_diagnostics"].get("singleton", {})
    if "comparison" in singleton_diag:
        print("\n2. Method Comparison:")
        comp = singleton_diag["comparison"]
        selected = singleton_diag.get("selected_method", "unknown")
        for method, lower, upper, width in zip(
            comp["method"], comp["lower"], comp["upper"], comp["width"], strict=False
        ):
            marker = " ← Selected" if method.lower() in selected.lower() else ""
            print(f"   {method:15s}: [{lower:.3f}, {upper:.3f}] (width: {width:.3f}){marker}")

# Show class-conditional error metrics (marginal scope)
print("\n" + "=" * 80)
print("CLASS-CONDITIONAL ERROR METRICS (Marginal Scope)")
print("=" * 80)

if "singleton_error_rate_cond_class0_bounds" in pac_marginal:
    cond_c0 = pac_marginal["singleton_error_rate_cond_class0_bounds"]
    cond_c1 = pac_marginal["singleton_error_rate_cond_class1_bounds"]
    print(f"\nP(error | singleton & class=0): [{cond_c0[0]:.3f}, {cond_c0[1]:.3f}]")
    print(f"P(error | singleton & class=1): [{cond_c1[0]:.3f}, {cond_c1[1]:.3f}]")

print("\n" + "=" * 80)
print("DEPLOYMENT RECOMMENDATION")
print("=" * 80)

pac_lower, pac_upper = pac_class_0["singleton_rate_bounds"]

print(f"\n✅ Use PAC bounds for SLA: [{pac_lower:.3f}, {pac_upper:.3f}] (90% guarantee)")
print(f"   → Monitor: Alert if observed rate < {pac_lower:.3f}")

print("\n" + "=" * 80)
print("\n💡 Use prediction_method='all' to compare analytical, exact, and Hoeffding methods")
print("   All bounds now use LOO-corrected uncertainty quantification")
print("\n" + "=" * 80)
