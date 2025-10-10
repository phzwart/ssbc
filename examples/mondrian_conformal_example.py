"""Example: Mondrian Conformal Prediction with SSBC Correction.

This example demonstrates the complete workflow:
1. Generate simulated binary classification data
2. Split data by class for Mondrian conformal prediction
3. Calibrate with SSBC correction
4. Generate comprehensive statistics report
"""

import matplotlib.pyplot as plt
import numpy as np

from ssbc import (
    BinaryClassifierSimulator,
    mondrian_conformal_calibrate,
    report_prediction_stats,
    split_by_class,
)


def main():
    """Run Mondrian conformal prediction example."""

    print("=" * 80)
    print("MONDRIAN CONFORMAL PREDICTION WITH SSBC CORRECTION")
    print("=" * 80)

    # ========== Step 1: Generate Simulated Data ==========
    print("\n1. Generating simulated binary classification data...")

    # Create simulator with balanced classes
    # Class 0: Beta(2, 7) → mean p(class=1) ≈ 0.22 (low scores, mostly correct)
    # Class 1: Beta(7, 2) → mean p(class=1) ≈ 0.78 (high scores, mostly correct)
    sim = BinaryClassifierSimulator(p_class1=0.50, beta_params_class0=(2, 7), beta_params_class1=(7, 2), seed=42)

    labels, probs = sim.generate(n_samples=1000)

    print(f"   Generated {len(labels)} samples")
    print(f"   Class balance: Class 0: {np.sum(labels == 0)}, Class 1: {np.sum(labels == 1)}")
    print(f"   Mean P(1) when true=0: {probs[labels == 0, 1].mean():.3f}")
    print(f"   Mean P(1) when true=1: {probs[labels == 1, 1].mean():.3f}")

    # ========== Step 2: Split by Class ==========
    print("\n2. Splitting data by class for Mondrian conformal prediction...")

    class_data = split_by_class(labels, probs)

    for label in [0, 1]:
        data = class_data[label]
        print(f"   Class {label}: n = {data['n']}")

    # ========== Step 3: Visualize Class Distributions ==========
    print("\n3. Visualizing predicted probability distributions...")

    plt.figure(figsize=(10, 5))
    bins = np.linspace(0, 1, 50)

    plt.hist(class_data[0]["probs"][:, 1], bins=bins, alpha=0.6, label="Class 0 (true)", color="blue")
    plt.hist(class_data[1]["probs"][:, 1], bins=bins, alpha=0.6, label="Class 1 (true)", color="red")
    plt.xlabel("P(class=1)")
    plt.ylabel("Count")
    plt.title("Distribution of Predicted Probabilities by True Class")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("examples/mondrian_distributions.png", dpi=150)
    print("   Saved plot to: examples/mondrian_distributions.png")

    # ========== Step 4: Calibrate with SSBC ==========
    print("\n4. Calibrating Mondrian conformal prediction with SSBC...")

    # Use same alpha and delta for both classes
    alpha_target = 0.10
    delta = 0.10

    print(f"   α_target = {alpha_target} (target 10% miscoverage)")
    print(f"   δ = {delta} (90% PAC guarantee)")

    cal_result, pred_stats = mondrian_conformal_calibrate(
        class_data=class_data, alpha_target={0: alpha_target, 1: alpha_target}, delta={0: delta, 1: delta}, mode="beta"
    )

    # Print calibration results
    print("\n   Calibration Results:")
    for label in [0, 1]:
        result = cal_result[label]
        print(f"   Class {label}:")
        print(f"      α_corrected = {result['alpha_corrected']:.4f}")
        print(f"      u* = {result['ssbc_result'].u_star}")
        print(f"      Threshold = {result['threshold']:.4f}")
        print(f"      Satisfied mass = {result['ssbc_result'].satisfied_mass:.4f}")

    # ========== Step 5: Generate Full Statistics Report ==========
    print("\n5. Generating comprehensive statistics report...\n")

    summary = report_prediction_stats(pred_stats, cal_result, verbose=True)

    # ========== Step 6: Summary Insights ==========
    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)

    marg = summary["marginal"]

    print("\nDeployment Performance:")
    print(
        f"  Coverage:        {marg['coverage']['rate']:.1%} "
        f"(CI: [{marg['coverage']['ci_95'][0]:.3f}, {marg['coverage']['ci_95'][1]:.3f}])"
    )
    print(f"  Singleton rate:  {marg['singletons']['rate']:.1%} (automated decisions)")
    print(f"  Escalation rate: {marg['abstentions']['rate'] + marg['doublets']['rate']:.1%} (requires human review)")

    sing_err = marg["singletons"]["errors"]
    print("\nSingleton Error Rate:")
    print(f"  Overall: {sing_err['rate']:.1%} (CI: [{sing_err['ci_95'][0]:.3f}, {sing_err['ci_95'][1]:.3f}])")

    # Check PAC bounds
    if marg["pac_bounds"]["rho"] is not None:
        pac = marg["pac_bounds"]
        print("\nPAC Bound Check:")
        print(f"  Theoretical bound: {pac['alpha_singlet_bound']:.4f}")
        print(f"  Observed rate:     {pac['alpha_singlet_observed']:.4f}")
        if pac["alpha_singlet_observed"] <= pac["alpha_singlet_bound"]:
            print("  ✓ PAC guarantee satisfied!")
        else:
            print(f"  ✗ PAC guarantee violated (may happen with probability δ={delta})")

    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
