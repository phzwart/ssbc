"""Example: Service Level Agreement (SLA) Bounds for Conformal Prediction.

This example demonstrates the complete SLA workflow combining:
1. Mondrian conformal calibration (PAC coverage via SSBC)
2. Operational rate bounds (via LOO-CV + Clopper-Pearson)

The workflow is:
1. Generate synthetic binary classification data with probabilities
2. Use mondrian_conformal_calibrate() for PAC coverage guarantees
3. Add rigorous operational rate bounds via compute_mondrian_operational_bounds()
4. Display comprehensive report with report_prediction_stats()

The result provides contract-ready guarantees on:
- Coverage: P(Y ∈ C(X)) ≥ 1 - α with probability ≥ 1 - δ₁ (from Mondrian)
- Operational rates: singleton, doublet, abstention rates with probability ≥ 1 - δ₂ (from LOO-CV)
"""

import numpy as np

from ssbc import (
    BinaryClassifierSimulator,
    compute_marginal_operational_bounds,
    compute_mondrian_operational_bounds,
    mondrian_conformal_calibrate,
    report_prediction_stats,
    split_by_class,
)


def main():
    """Run SLA example with binary classification."""

    print("=" * 80)
    print("CONFORMAL PREDICTION WITH SERVICE LEVEL AGREEMENT (SLA)")
    print("=" * 80)

    # ========== Step 1: Generate Synthetic Binary Classification Data ==========
    print("\n1. Generating synthetic binary classification data...")

    np.random.seed(42)
    n_cal = 200

    # Simulate binary classifier with overlapping distributions
    sim = BinaryClassifierSimulator(p_class1=0.5, beta_params_class0=(3, 7), beta_params_class1=(7, 3), seed=42)

    labels, probs = sim.generate(n_samples=n_cal)

    print(f"   Calibration set size: n = {n_cal}")
    print(f"   Class 0: {np.sum(labels == 0)} samples")
    print(f"   Class 1: {np.sum(labels == 1)} samples")

    # ========== Step 2: Mondrian Calibration (PAC Coverage) ==========
    print("\n2. Mondrian conformal calibration (PAC coverage guarantees)...")

    alpha_target = 0.10  # Target 90% coverage per class
    delta_1 = 0.05  # 95% confidence for coverage

    print(f"   α_target = {alpha_target} (target 90% coverage per class)")
    print(f"   δ₁ = {delta_1} (coverage confidence: {1 - delta_1:.1%})")

    class_data = split_by_class(labels, probs)
    cal_result, pred_stats = mondrian_conformal_calibrate(
        class_data=class_data, alpha_target=alpha_target, delta=delta_1, mode="beta"
    )

    print("\n   Calibration Results (with SSBC correction):")
    for class_label in [0, 1]:
        result = cal_result[class_label]
        print(f"   Class {class_label}:")
        print(f"      α_corrected = {result['alpha_corrected']:.4f}")
        print(f"      Threshold   = {result['threshold']:.4f}")
        print(f"      n           = {result['n']}")
        print(f"      PAC mass    = {result['ssbc_result'].satisfied_mass:.4f}")

    # ========== Step 3: Operational Rate Bounds ==========
    print("\n3. Computing operational rate bounds via LOO-CV...")
    print("   (Leave-one-out cross-validation on calibration data)")

    delta_2 = 0.05  # 95% confidence for rate bounds

    print(f"   δ₂ = {delta_2} (rate bounds confidence: {1 - delta_2:.1%})")
    print("   Using default rate types (singleton, doublet, abstention, conditional rates)")

    # Compute per-class operational bounds using LOO-CV
    operational_bounds = compute_mondrian_operational_bounds(
        calibration_result=cal_result,
        labels=labels,
        probs=probs,
    )

    # Also compute marginal bounds
    marginal_bounds = compute_marginal_operational_bounds(
        labels=labels, probs=probs, alpha_target=alpha_target, delta_coverage=delta_1
    )

    print("   ✓ Operational bounds computation complete")

    # ========== Step 4: Display Comprehensive Report ==========
    print("\n4. Generating comprehensive SLA report...\n")

    report_prediction_stats(
        prediction_stats=pred_stats,
        calibration_result=cal_result,
        operational_bounds_per_class=operational_bounds,
        marginal_operational_bounds=marginal_bounds,
        verbose=True,
    )

    # ========== Summary ==========
    print("\n" + "=" * 80)
    print("KEY TAKEAWAYS")
    print("=" * 80)

    print("\n✓ Complete SLA for Mondrian conformal prediction computed!")
    print()
    print("1. PAC Coverage Guarantees (from SSBC):")
    print(f"   - Each class achieves ≥ {1 - alpha_target:.1%} coverage")
    print(f"   - With probability ≥ {1 - delta_1:.1%}")
    print()
    print("2. Operational Rate Bounds (from LOO-CV):")
    print(f"   - {n_cal} leave-one-out evaluations per class")
    print("   - Rigorous Clopper-Pearson bounds")
    print(f"   - Each rate gets confidence {1 - delta_2:.1%} independently")
    print()
    print(f"3. Joint Confidence (union bound): ≥ {1 - (delta_1 + delta_2):.1%}")
    print()
    print("See the report above for detailed bounds on:")
    print("  • Singleton, doublet, abstention rates (per-class and marginal)")
    print("  • Conditional rates: P(correct | singleton), P(error | singleton)")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
