"""Example: Service Level Agreement (SLA) Bounds for Conformal Prediction.

This example demonstrates the complete SLA workflow combining:
1. Mondrian conformal calibration (PAC coverage via SSBC)
2. Operational rate bounds (via cross-validation + Clopper-Pearson)

The workflow is:
1. Generate synthetic binary classification data with probabilities
2. Use mondrian_conformal_calibrate() for PAC coverage guarantees
3. Add rigorous operational rate bounds via compute_mondrian_operational_bounds()

The result provides contract-ready guarantees on:
- Coverage: P(Y ∈ C(X)) ≥ 1 - α with probability ≥ 1 - δ₁ (from Mondrian)
- Operational rates: singleton, doublet, abstention rates with probability ≥ 1 - δ₂ (from SLA)
"""

import numpy as np

from ssbc import (
    BinaryClassifierSimulator,
    compute_mondrian_operational_bounds,
    mondrian_conformal_calibrate,
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
    print("\n3. Computing operational rate bounds via cross-validation...")
    print("   (Using the SAME class_data from step 2)")

    delta_2 = 0.05  # 95% confidence for rate bounds
    rate_types = ["singleton", "doublet", "abstention"]
    n_folds = 5

    print(f"   δ₂ = {delta_2} (rate bounds confidence: {1 - delta_2:.1%})")
    print(f"   Operational rates: {rate_types}")
    print(f"   Cross-validation folds: {n_folds}")

    # Compute operational bounds using labels and probs
    # Cross-validates on marginal data, then reports per-class
    operational_bounds = compute_mondrian_operational_bounds(
        calibration_result=cal_result,
        labels=labels,
        probs=probs,
        delta=delta_2,
        rate_types=rate_types,
        n_folds=n_folds,
        random_seed=42,
    )

    print("   ✓ Operational bounds computation complete")

    # ========== Step 4: Display Results ==========
    print("\n" + "=" * 80)
    print("COMPLETE SLA RESULTS")
    print("=" * 80)

    for class_label in [0, 1]:
        print(f"\n{'=' * 80}")
        print(f"CLASS {class_label}")
        print(f"{'=' * 80}")

        # Coverage (from Mondrian)
        cal = cal_result[class_label]
        print("\n📊 COVERAGE GUARANTEE (from Mondrian + SSBC):")
        print(f"  Target coverage:       {1 - alpha_target:.1%}")
        print(f"  Adjusted alpha:        {cal['alpha_corrected']:.4f}")
        print(f"  Confidence level:      {1 - delta_1:.1%}")
        print(f"  Conformal threshold:   {cal['threshold']:.4f}")
        print(f"  Calibration set size:  n = {cal['n']}")
        print()
        print(f"  Guarantee: P(Coverage ≥ {1 - alpha_target:.1%}) ≥ {1 - delta_1:.1%}")

        # Operational rates (from SLA)
        if class_label in operational_bounds:
            bounds_result = operational_bounds[class_label]
            print(f"\n📈 OPERATIONAL RATE BOUNDS (confidence: {bounds_result.rate_confidence:.1%}):")
            for rate_name, bounds in bounds_result.rate_bounds.items():
                print(f"\n  {rate_name.upper()}:")
                print(f"    Single-rule bounds:  [{bounds.lower_bound:.3f}, {bounds.upper_bound:.3f}]")
                print(f"    Cross-fit bounds:    [{bounds.cross_fit_lower:.3f}, {bounds.cross_fit_upper:.3f}]")
                print(f"    Transfer cushion:    {bounds.cushion:.4f}")
                print(f"    Number of folds:     {len(bounds.fold_results)}")

    # ========== Step 5: Deployment Interpretation ==========
    print("\n" + "=" * 80)
    print("DEPLOYMENT GUIDE")
    print("=" * 80)

    print("\nTo deploy the Mondrian conformal predictor:")
    print()
    print("1. For each test input x with probabilities [P(0|x), P(1|x)]:")
    print(f"   - Compute score_0 = 1 - P(0|x), check if score_0 ≤ {cal_result[0]['threshold']:.4f}")
    print(f"   - Compute score_1 = 1 - P(1|x), check if score_1 ≤ {cal_result[1]['threshold']:.4f}")
    print("2. Prediction set C(x) = {y : score_y ≤ threshold_y}")
    print()
    print("Expected behavior with these guarantees:")

    for class_label in [0, 1]:
        cal = cal_result[class_label]
        print(f"\n  CLASS {class_label}:")
        print(f"    • Coverage:  ≥ {1 - alpha_target:.1%} (w.p. ≥ {1 - delta_1:.1%})")

        if class_label in operational_bounds:
            singleton_bounds = operational_bounds[class_label].rate_bounds.get("singleton")
            doublet_bounds = operational_bounds[class_label].rate_bounds.get("doublet")

            if singleton_bounds:
                print(
                    f"    • Singleton: [{singleton_bounds.lower_bound:.1%}, {singleton_bounds.upper_bound:.1%}] "
                    f"(w.p. ≥ {1 - delta_2:.1%})"
                )
            if doublet_bounds:
                print(
                    f"    • Doublet:   [{doublet_bounds.lower_bound:.1%}, {doublet_bounds.upper_bound:.1%}] "
                    f"(w.p. ≥ {1 - delta_2:.1%})"
                )

    # ========== Summary ==========
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print("\n✓ Successfully computed complete SLA for Mondrian conformal prediction")
    print()
    print("Key takeaways:")
    print("  1. PAC coverage guarantees from mondrian_conformal_calibrate() with SSBC")
    print(
        f"     - Class 0: α_corrected = {cal_result[0]['alpha_corrected']:.4f}, "
        f"threshold = {cal_result[0]['threshold']:.4f}"
    )
    print(
        f"     - Class 1: α_corrected = {cal_result[1]['alpha_corrected']:.4f}, "
        f"threshold = {cal_result[1]['threshold']:.4f}"
    )
    print("  2. Rigorous operational rate bounds from compute_mondrian_operational_bounds()")
    print(f"     - Computed via {n_folds}-fold cross-validation with Clopper-Pearson CIs")
    print("     - Transfer cushions account for single refit-on-all deployment")
    print(f"  3. Joint confidence: {1 - (delta_1 + delta_2):.1%} (union bound)")
    print()
    print("These guarantees are valid for production deployment of the Mondrian predictor")
    print("with per-class thresholds computed on the full calibration set.")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
