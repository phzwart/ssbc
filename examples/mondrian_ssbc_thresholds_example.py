"""Example: Using existing SSBC code to compute Mondrian thresholds and rigorous error bounds.

This example demonstrates how to use the existing SSBC infrastructure to:
1. Compute SSBC-corrected thresholds for both classes
2. Generate a rigorous PAC report with error rates and bounds
"""

import numpy as np

from ssbc import (
    BinaryClassifierSimulator,
    generate_rigorous_pac_report,
    mondrian_conformal_calibrate,
    split_by_class,
)


def main():
    """Demonstrate SSBC threshold computation and rigorous reporting."""

    print("=" * 80)
    print("Mondrian SSBC Thresholds and Rigorous Error Bounds Report")
    print("=" * 80)

    # ========== Step 1: Generate Simulated Data ==========
    print("\n1. Generating simulated binary classification data...")

    sim = BinaryClassifierSimulator(p_class1=0.50, beta_params_class0=(2, 7), beta_params_class1=(7, 2), seed=42)

    labels, probs = sim.generate(n_samples=1000)

    print(f"   Generated {len(labels)} samples")
    print(f"   Class balance: Class 0: {np.sum(labels == 0)}, Class 1: {np.sum(labels == 1)}")

    # ========== Step 2: Split by Class ==========
    print("\n2. Splitting data by class for Mondrian conformal prediction...")

    class_data = split_by_class(labels, probs)

    for label in [0, 1]:
        print(f"   Class {label}: n = {class_data[label]['n']}")

    # ========== Step 3: Compute SSBC-Corrected Thresholds ==========
    print("\n3. Computing SSBC-corrected thresholds...")

    alpha_target = 0.10
    delta = 0.10

    # Calibrate with SSBC
    cal_result, pred_stats = mondrian_conformal_calibrate(
        class_data=class_data, alpha_target={0: alpha_target, 1: alpha_target}, delta={0: delta, 1: delta}, mode="beta"
    )

    # Extract thresholds (these are the equivalent of mondrian_ssbc_thresholds output)
    print("\n   Thresholds:")
    for label in [0, 1]:
        result = cal_result[label]
        print(f"   Class {label}:")
        print(f"      Threshold: {result['threshold']:.4f}")
        print(f"      α_corrected: {result['alpha_corrected']:.4f}")
        print(f"      Satisfied mass: {result['ssbc_result'].satisfied_mass:.4f}")

    # ========== Step 4: Generate Rigorous PAC Report ==========
    print("\n4. Generating rigorous PAC report with error rates and bounds...")
    print("   (This includes LOO-CV + Clopper-Pearson for estimation uncertainty)\n")

    report = generate_rigorous_pac_report(
        labels=labels, probs=probs, alpha_target=alpha_target, delta=delta, use_loo_correction=True, verbose=True
    )

    # ========== Step 5: Extract Key Error Bounds ==========
    print("\n" + "=" * 80)
    print("Key Error Bounds Summary")
    print("=" * 80)

    # Marginal error bounds
    pac_marg = report["pac_bounds_marginal"]
    se_lower, se_upper = pac_marg["singleton_error_rate_bounds"]

    print("\nMarginal (Overall):")
    print(f"  Singleton error rate: [{se_lower:.3f}, {se_upper:.3f}]")
    print(f"  Expected: {pac_marg['expected_singleton_error_rate']:.3f}")

    # Per-class error bounds
    for class_label in [0, 1]:
        pac_class = report[f"pac_bounds_class_{class_label}"]
        se_lower_c, se_upper_c = pac_class["singleton_error_rate_bounds"]

        print(f"\nClass {class_label}:")
        print(f"  Singleton error rate: [{se_lower_c:.3f}, {se_upper_c:.3f}]")
        print(f"  Expected: {pac_class['expected_singleton_error_rate']:.3f}")

    # ========== Step 6: Operational Bounds ==========
    print("\n" + "=" * 80)
    print("Operational Bounds (Deployment Expectations)")
    print("=" * 80)

    print("\nMarginal Operations:")
    s_lower, s_upper = pac_marg["singleton_rate_bounds"]
    a_lower, a_upper = pac_marg["abstention_rate_bounds"]
    d_lower, d_upper = pac_marg["doublet_rate_bounds"]

    print(f"  Automation (singletons): [{s_lower:.1%}, {s_upper:.1%}]")
    print(f"  Escalation (doublets+abstentions): [{a_lower + d_lower:.1%}, {a_upper + d_upper:.1%}]")

    # ========== Complete! ==========
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print("\n✓ SSBC-corrected thresholds computed for both classes")
    print("✓ PAC-controlled error bounds generated")
    print("✓ Operational bounds computed for deployment planning")
    print("\nThe rigorous PAC report ensures:")
    print("  • Coverage guarantees hold with high probability")
    print("  • Error rates are bounded with statistical confidence")
    print("  • All bounds account for calibration uncertainty via LOO-CV")
    print("=" * 80)


if __name__ == "__main__":
    main()
