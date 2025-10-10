"""Example: Service Level Agreement (SLA) Bounds for Conformal Prediction.

This example demonstrates how to compute PAC coverage guarantees and
operational rate bounds for conformal prediction using the SLA module.

The workflow includes:
1. Generate synthetic multi-class classification data
2. Define a nonconformity score function
3. Compute PAC coverage with SSBC adjustment
4. Compute cross-fit operational rate bounds
5. Transfer bounds to single refit-on-all rule for deployment
6. (Optional) Mondrian class-conditional SLA

The result provides contract-ready guarantees on:
- Coverage: P(Y ∈ C(X)) ≥ 1 - α with probability ≥ 1 - δ₁
- Operational rates: singleton, doublet, abstention rates with probability ≥ 1 - δ₂
"""

import numpy as np

from ssbc import compute_conformal_sla, compute_conformal_sla_mondrian


def main():
    """Run SLA example with synthetic data."""

    print("=" * 80)
    print("CONFORMAL PREDICTION WITH SERVICE LEVEL AGREEMENT (SLA)")
    print("=" * 80)

    # ========== Step 1: Generate Synthetic Data ==========
    print("\n1. Generating synthetic multi-class classification data...")

    np.random.seed(42)
    n_cal = 200
    n_classes = 5
    d = 10  # feature dimension

    # Simulate features and labels
    cal_features = np.random.randn(n_cal, d)
    cal_labels = np.random.randint(0, n_classes, n_cal)

    print(f"   Calibration set size: n = {n_cal}")
    print(f"   Number of classes: {n_classes}")
    print(f"   Feature dimension: {d}")
    print(f"   Class distribution: {np.bincount(cal_labels)}")

    # ========== Step 2: Define Nonconformity Score Function ==========
    print("\n2. Defining nonconformity score function...")

    # Simulate class-specific prototypes/centers
    # In practice, this would be based on your trained model (e.g., softmax scores)
    class_centers = np.random.randn(n_classes, d)

    def score_function(x, y):
        """Nonconformity score: distance from class center.

        Higher score = less conforming = less confident in this label.

        Parameters
        ----------
        x : np.ndarray
            Feature vector
        y : int
            Candidate label

        Returns
        -------
        float
            Nonconformity score
        """
        if y < len(class_centers):
            center = class_centers[y]
        else:
            center = np.zeros(d)
        return float(np.linalg.norm(x - center))

    print("   Score function: distance from class center")
    print("   (In practice, use 1 - P(y|x) from your classifier)")

    # ========== Step 3: Set SLA Parameters ==========
    print("\n3. Setting SLA parameters...")

    alpha_target = 0.10  # Target 90% coverage
    delta_1 = 0.05  # 95% confidence for coverage guarantee
    delta_2 = 0.05  # 95% confidence for rate bounds

    rate_types = ["singleton", "doublet", "abstention"]
    n_folds = 5

    print(f"   α_target = {alpha_target} (target 90% coverage)")
    print(f"   δ₁ = {delta_1} (coverage confidence: {1 - delta_1:.1%})")
    print(f"   δ₂ = {delta_2} (rate bounds confidence: {1 - delta_2:.1%})")
    print(f"   Operational rates: {rate_types}")
    print(f"   Cross-validation folds: {n_folds}")

    # ========== Step 4: Run Complete SLA Pipeline ==========
    print("\n4. Computing conformal SLA...")

    sla_result = compute_conformal_sla(
        cal_features=cal_features,
        cal_labels=cal_labels,
        score_function=score_function,
        alpha_target=alpha_target,
        delta_1=delta_1,
        delta_2=delta_2,
        rate_types=rate_types,
        n_folds=n_folds,
        random_seed=42,
    )

    print("   ✓ SLA computation complete")

    # ========== Step 5: Display Results ==========
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    print("\n📊 COVERAGE GUARANTEE:")
    print(f"  Target coverage:       {sla_result.coverage_guarantee:.1%}")
    print(f"  Adjusted alpha:        {sla_result.alpha_adjusted:.4f}")
    print(f"  Confidence level:      {sla_result.coverage_confidence:.1%}")
    print(f"  Conformal threshold:   {sla_result.threshold:.4f}")
    print(f"  Calibration set size:  n = {sla_result.n_calibration}")
    print()
    print(f"  Guarantee: P(Coverage ≥ {sla_result.coverage_guarantee:.1%}) ≥ {sla_result.coverage_confidence:.1%}")

    print(f"\n📈 OPERATIONAL RATE BOUNDS (confidence: {sla_result.rate_confidence:.1%}):")
    for rate_name, bounds in sla_result.rate_bounds.items():
        print(f"\n  {rate_name.upper().replace('_', ' ')}:")
        print(f"    Single-rule bounds:  [{bounds.lower_bound:.3f}, {bounds.upper_bound:.3f}]")
        print(f"    Cross-fit bounds:    [{bounds.cross_fit_lower:.3f}, {bounds.cross_fit_upper:.3f}]")
        print(f"    Transfer cushion:    {bounds.cushion:.4f}")
        print(f"    Number of folds:     {len(bounds.fold_results)}")

    print("\n🔒 JOINT SLA GUARANTEE:")
    print(f"  Confidence: {sla_result.joint_confidence:.1%}")
    print(f"  All bounds (coverage + rates) hold simultaneously with probability ≥ {sla_result.joint_confidence:.1%}")

    # ========== Step 6: Deployment Interpretation ==========
    print("\n" + "=" * 80)
    print("DEPLOYMENT GUIDE")
    print("=" * 80)

    print(f"\nTo deploy the conformal predictor with threshold q = {sla_result.threshold:.4f}:")
    print()
    print("1. For each test input x, compute scores A(x, y) for all candidate labels y")
    print(f"2. Create prediction set: C(x) = {{y : A(x, y) ≤ {sla_result.threshold:.4f}}}")
    print()
    print("Expected behavior with these guarantees:")

    singleton_bounds = sla_result.rate_bounds["singleton"]
    doublet_bounds = sla_result.rate_bounds["doublet"]
    abstention_bounds = sla_result.rate_bounds["abstention"]

    print(f"  • Coverage:        ≥ {sla_result.coverage_guarantee:.1%} (w.p. ≥ {sla_result.coverage_confidence:.1%})")
    print(
        f"  • Singleton rate:  [{singleton_bounds.lower_bound:.1%}, {singleton_bounds.upper_bound:.1%}] "
        f"(automated decisions)"
    )
    print(
        f"  • Doublet rate:    [{doublet_bounds.lower_bound:.1%}, {doublet_bounds.upper_bound:.1%}] "
        f"(ambiguous, needs review)"
    )
    print(
        f"  • Abstention rate: [{abstention_bounds.lower_bound:.1%}, {abstention_bounds.upper_bound:.1%}] "
        f"(high uncertainty)"
    )

    escalation_lower = doublet_bounds.lower_bound + abstention_bounds.lower_bound
    escalation_upper = doublet_bounds.upper_bound + abstention_bounds.upper_bound
    print(f"  • Escalation rate: [{escalation_lower:.1%}, {escalation_upper:.1%}] (requires human intervention)")

    # ========== Step 7: Mondrian (Class-Conditional) SLA ==========
    print("\n" + "=" * 80)
    print("MONDRIAN (CLASS-CONDITIONAL) SLA")
    print("=" * 80)

    print("\nComputing class-specific SLAs (useful when classes have different difficulties)...")

    mondrian_results = compute_conformal_sla_mondrian(
        cal_features=cal_features,
        cal_labels=cal_labels,
        score_function=score_function,
        alpha_target=alpha_target,
        delta_1=delta_1,
        delta_2=delta_2,
        rate_types=rate_types,
        n_folds=n_folds,
        random_seed=42,
    )

    print(f"\nResults for {len(mondrian_results)} classes:\n")
    for class_label, result in sorted(mondrian_results.items()):
        print(f"CLASS {class_label}:")
        print(f"  Coverage:   {result.coverage_guarantee:.1%} @ {result.coverage_confidence:.1%} confidence")
        print(f"  Threshold:  {result.threshold:.4f}")
        print(f"  n_cal:      {result.n_calibration}")

        singleton_bounds = result.rate_bounds.get("singleton")
        if singleton_bounds:
            print(f"  Singleton:  [{singleton_bounds.lower_bound:.3f}, {singleton_bounds.upper_bound:.3f}]")
        print()

    # ========== Summary ==========
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print("\n✓ Successfully computed SLA bounds for conformal prediction")
    print()
    print("Key takeaways:")
    print(
        f"  1. Coverage guarantee: ≥ {sla_result.coverage_guarantee:.1%} "
        f"with {sla_result.coverage_confidence:.1%} confidence"
    )
    print(f"  2. Operational rate bounds hold with {sla_result.rate_confidence:.1%} confidence")
    print(f"  3. Joint guarantee: {sla_result.joint_confidence:.1%} confidence for all bounds simultaneously")
    print(f"  4. Deploy using threshold q = {sla_result.threshold:.4f}")
    print()
    print("These guarantees are valid for the single refit-on-all conformal predictor")
    print("trained on the full calibration set, ready for production deployment.")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
