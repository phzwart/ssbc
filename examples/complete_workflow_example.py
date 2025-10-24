"""Complete uncertainty quantification workflow - all methods integrated.

Demonstrates the COMPLETE rigorous_pac_report with ALL uncertainty analyses:
1. PAC Bounds (LOO-CV + CP) - Always included
2. Bootstrap - Optional, requires simulator
3. Cross-Conformal - Optional, no simulator needed

All from a single function call!
"""

import numpy as np

from ssbc import BinaryClassifierSimulator, generate_rigorous_pac_report

# Create simulator
sim = BinaryClassifierSimulator(p_class1=0.20, beta_params_class0=(1, 7), beta_params_class1=(5, 2), seed=42)

# Generate calibration data
labels, probs = sim.generate(n_samples=100)

print("=" * 80)
print("COMPLETE UNCERTAINTY QUANTIFICATION - SINGLE FUNCTION CALL")
print("=" * 80)
print(f"\nCalibration: {len(labels)} samples ({np.sum(labels == 0)} class 0, {np.sum(labels == 1)} class 1)")
print("\n✓ Generating comprehensive report with ALL uncertainty analyses...")

# Generate complete report with ALL analyses
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
    # Optional: Bootstrap analysis
    run_bootstrap=True,
    n_bootstrap=1000,
    simulator=sim,
    # Optional: Cross-conformal validation
    run_cross_conformal=True,
    n_folds=10,
)

print("\n" + "=" * 80)
print("COMPARATIVE ANALYSIS")
print("=" * 80)

# Extract results for comparison
pac_class_0 = report["pac_bounds_class_0"]
bootstrap_class_0 = report["bootstrap_results"]["class_0"]["singleton"] if report["bootstrap_results"] else None
cross_conf_class_0 = (
    report["cross_conformal_results"]["class_0"]["singleton"] if report["cross_conformal_results"] else None
)

print("\nClass 0 Singleton Rate Summary:")
print("-" * 80)

pac_bounds = pac_class_0["singleton_rate_bounds"]
print(f"\n1. PAC Bounds (LOO-CV + CP): [{pac_bounds[0]:.3f}, {pac_bounds[1]:.3f}]")
print(f"   Expected: {pac_class_0['expected_singleton_rate']:.3f}")
print("   → Rigorous deployment guarantees")

if bootstrap_class_0:
    bs_q = bootstrap_class_0["quantiles"]
    print(f"\n2. Bootstrap:                [{bs_q['q05']:.3f}, {bs_q['q95']:.3f}]")
    print(f"   Mean: {bootstrap_class_0['mean']:.3f} ± {bootstrap_class_0['std']:.3f}")
    print("   → Recalibration uncertainty")

if cross_conf_class_0:
    cc_q = cross_conf_class_0["quantiles"]
    print(f"\n3. Cross-Conformal:          [{cc_q['q05']:.3f}, {cc_q['q95']:.3f}]")
    print(f"   Mean: {cross_conf_class_0['mean']:.3f} ± {cross_conf_class_0['std']:.3f}")
    print("   → Finite-sample diagnostics")

print("\n" + "=" * 80)
print("DEPLOYMENT RECOMMENDATION")
print("=" * 80)

pac_lower, pac_upper = pac_class_0["singleton_rate_bounds"]

print(f"\n✅ Use PAC bounds for SLA: [{pac_lower:.3f}, {pac_upper:.3f}] (90% guarantee)")
print(f"   → Monitor: Alert if observed rate < {pac_lower:.3f}")

if cross_conf_class_0 and cross_conf_class_0["std"] > 0.1:
    print(f"\n⚠️  Cross-conformal std = {cross_conf_class_0['std']:.3f} is high")
    print("   → Consider collecting more calibration data")
elif cross_conf_class_0:
    print(f"\n✓ Cross-conformal std = {cross_conf_class_0['std']:.3f} is acceptable")
    print("   → Current calibration size is adequate")
else:
    print("\n✓ Cross-conformal validation not performed")

if bootstrap_class_0:
    bs_width = bootstrap_class_0["quantiles"]["q95"] - bootstrap_class_0["quantiles"]["q05"]
    pac_width = pac_upper - pac_lower
    if bs_width > pac_width * 1.5:
        print(f"\n⚠️  Bootstrap range ({bs_width:.3f}) >> PAC range ({pac_width:.3f})")
        print("   → High sensitivity to recalibration")
    else:
        print(f"\n✓ Bootstrap range ({bs_width:.3f}) ≈ PAC range ({pac_width:.3f})")
        print("   → Low recalibration sensitivity")

print("\n" + "=" * 80)
print("\n💡 All three methods are now part of the standard rigorous_pac_report!")
print("   Simply set run_bootstrap=True and/or run_cross_conformal=True")
print("\n" + "=" * 80)
