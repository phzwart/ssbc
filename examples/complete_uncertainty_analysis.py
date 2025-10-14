"""Complete uncertainty quantification workflow for conformal prediction.

This example demonstrates ALL THREE complementary methods for uncertainty analysis:
1. LOO-CV + CP Bounds: PAC guarantees for operational rates (fixed calibration)
2. Bootstrap: Recalibration uncertainty (sampling with replacement + fresh test)
3. Cross-Conformal: Finite-sample variability (K-fold splits of calibration)

Each method answers a different question!
"""

import numpy as np
from ssbc import (
    BinaryClassifierSimulator,
    generate_rigorous_pac_report,
    cross_conformal_validation,
    print_cross_conformal_results,
)

# Create simulator
sim = BinaryClassifierSimulator(p_class1=0.20, beta_params_class0=(1, 7), beta_params_class1=(5, 2), seed=42)

# Generate calibration data
labels, probs = sim.generate(n_samples=100)

print("=" * 80)
print("COMPLETE UNCERTAINTY QUANTIFICATION FOR CONFORMAL PREDICTION")
print("=" * 80)
print(f"\nCalibration size: {len(labels)}")
print(f"Class distribution: {np.sum(labels == 0)} class 0, {np.sum(labels == 1)} class 1")
print("\nRunning all three uncertainty analyses...")

# ============================================================================
# METHOD 1: LOO-CV + Clopper-Pearson (PAC Guarantees)
# ============================================================================
print("\n" + "=" * 80)
print("METHOD 1: LOO-CV + CLOPPER-PEARSON BOUNDS (PAC Guarantees)")
print("=" * 80)
print("\nQuestion: 'Given THIS calibration, what rates on future test sets?'")

pac_report = generate_rigorous_pac_report(
    labels=labels,
    probs=probs,
    alpha_target=0.10,
    delta=0.10,
    test_size=1000,
    ci_level=0.95,
    use_union_bound=True,
    n_jobs=-1,
    verbose=False,  # Suppress detailed output for summary
    run_bootstrap=False,  # We'll run separately
)

pac_class_0 = pac_report["pac_bounds_class_0"]
singleton_bounds = pac_class_0["singleton_rate_bounds"]
singleton_expected = pac_class_0["expected_singleton_rate"]

print(f"\n✓ Class 0 Singleton Rate:")
print(f"  PAC Bounds:  [{singleton_bounds[0]:.3f}, {singleton_bounds[1]:.3f}]")
print(f"  Expected:    {singleton_expected:.3f}")
print(f"\n  → These bounds have 90% PAC guarantee (1 - δ = 0.90)")
print(f"  → Valid for ANY future test set from same distribution")
print(f"  → Use for: Deployment guarantees and SLA contracts")

# ============================================================================
# METHOD 2: Bootstrap Calibration Uncertainty
# ============================================================================
print("\n" + "=" * 80)
print("METHOD 2: BOOTSTRAP CALIBRATION UNCERTAINTY")
print("=" * 80)
print("\nQuestion: 'If I recalibrate on similar datasets, how do rates vary?'")

bootstrap_report = generate_rigorous_pac_report(
    labels=labels,
    probs=probs,
    alpha_target=0.10,
    delta=0.10,
    test_size=1000,
    n_jobs=-1,
    verbose=False,
    run_bootstrap=True,
    n_bootstrap=1000,
    simulator=sim,
)

bootstrap_class_0 = bootstrap_report["bootstrap_results"]["class_0"]["singleton"]
bootstrap_mean = bootstrap_class_0["mean"]
bootstrap_std = bootstrap_class_0["std"]
bootstrap_q = bootstrap_class_0["quantiles"]

print(f"\n✓ Class 0 Singleton Rate:")
print(f"  Bootstrap Mean:  {bootstrap_mean:.3f} ± {bootstrap_std:.3f}")
print(f"  [5%, 95%]:       [{bootstrap_q['q05']:.3f}, {bootstrap_q['q95']:.3f}]")
print(f"\n  → Shows variability from recalibrating on similar datasets")
print(f"  → Includes both calibration and test set uncertainties")
print(f"  → Use for: Understanding sensitivity to calibration choice")

# ============================================================================
# METHOD 3: Cross-Conformal Validation
# ============================================================================
print("\n" + "=" * 80)
print("METHOD 3: CROSS-CONFORMAL VALIDATION (K-Fold)")
print("=" * 80)
print("\nQuestion: 'How stable are rates across different calibration subsets?'")

cross_conf_results = cross_conformal_validation(
    labels=labels,
    probs=probs,
    alpha_target=0.10,
    delta=0.10,
    n_folds=10,
    stratify=True,
    seed=123,
)

cc_class_0 = cross_conf_results["class_0"]["singleton"]
cc_mean = cc_class_0["mean"]
cc_std = cc_class_0["std"]
cc_q = cc_class_0["quantiles"]

print(f"\n✓ Class 0 Singleton Rate:")
print(f"  Cross-Conf Mean: {cc_mean:.3f} ± {cc_std:.3f}")
print(f"  [5%, 95%]:       [{cc_q['q05']:.3f}, {cc_q['q95']:.3f}]")
print(f"\n  → Shows finite-sample variability from K-fold splits")
print(f"  → No simulator needed (uses calibration data only)")
print(f"  → Use for: Diagnosing if more calibration data needed")

# ============================================================================
# COMPARATIVE SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("COMPARATIVE SUMMARY: Class 0 Singleton Rate")
print("=" * 80)

print(f"\n{'Method':<30} {'Bounds/Range':<25} {'Width':<10} {'Purpose'}")
print("-" * 80)

pac_width = singleton_bounds[1] - singleton_bounds[0]
bootstrap_width = bootstrap_q["q95"] - bootstrap_q["q05"]
cc_width = cc_q["q95"] - cc_q["q05"]

print(f"{'1. LOO-CV + CP (PAC)':<30} [{singleton_bounds[0]:.3f}, {singleton_bounds[1]:.3f}]        {pac_width:.3f}      PAC guarantees")
print(f"{'2. Bootstrap':<30} [{bootstrap_q['q05']:.3f}, {bootstrap_q['q95']:.3f}]        {bootstrap_width:.3f}      Recal. uncertainty")
print(f"{'3. Cross-Conformal':<30} [{cc_q['q05']:.3f}, {cc_q['q95']:.3f}]        {cc_width:.3f}      Finite-sample diag.")

print("\n" + "=" * 80)
print("KEY INSIGHTS")
print("=" * 80)

print("\n🎯 Method 1 (LOO-CV + CP): DEPLOYMENT GUARANTEES")
print("  • Rigorous PAC bounds for THIS fixed calibration")
print("  • Accounts for estimation uncertainty via CP intervals")
print("  • Use for: SLA contracts, deployment decisions, risk management")

print("\n🔄 Method 2 (Bootstrap): RECALIBRATION SENSITIVITY")
print("  • Shows how rates vary if you recalibrate on similar data")
print("  • Typically wider than PAC bounds (includes recal. variability)")
print("  • Use for: Understanding model/data sensitivity, what-if analysis")

print("\n📊 Method 3 (Cross-Conformal): CALIBRATION DIAGNOSTICS")
print("  • Shows rate stability across K-fold splits")
print("  • Large std → need more calibration data")
print("  • Use for: Sample size planning, stability assessment")

print("\n✅ ALL THREE ARE COMPLEMENTARY:")
print("  → Use LOO-CV for deployment (PAC guarantees)")
print("  → Use Bootstrap to understand recalibration impact")
print("  → Use Cross-Conformal to diagnose calibration quality")

print("\n📈 INTERPRETATION FOR THIS DATASET:")

if cc_std > 0.1:
    print(f"  ⚠️  High cross-conformal std ({cc_std:.3f}) → Consider more calibration data")
else:
    print(f"  ✓ Low cross-conformal std ({cc_std:.3f}) → Calibration is stable")

if bootstrap_width > pac_width * 1.5:
    print(f"  ⚠️  Bootstrap much wider than PAC → High recalibration sensitivity")
else:
    print(f"  ✓ Bootstrap similar to PAC → Low recalibration sensitivity")

print("\n" + "=" * 80)
print("RECOMMENDATION")
print("=" * 80)

print("\nFor production deployment:")
print(f"  1. Report PAC bounds: [{singleton_bounds[0]:.3f}, {singleton_bounds[1]:.3f}] (90% guarantee)")
print(f"  2. Monitor: If observed rate < {singleton_bounds[0]:.3f}, investigate")
print(f"  3. Recalibration: Expect rates in [{bootstrap_q['q05']:.3f}, {bootstrap_q['q95']:.3f}] range")
print(f"  4. Sample size: Cross-conformal std = {cc_std:.3f}")

if cc_std > 0.1:
    print(f"     → Consider increasing calibration size to reduce variability")
else:
    print(f"     → Current calibration size appears adequate")

print("\n" + "=" * 80)

