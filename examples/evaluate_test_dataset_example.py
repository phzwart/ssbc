"""Example demonstrating evaluate_test_dataset() function.

This shows how to use the new evaluate_test_dataset() function to compute
empirical rates from a test dataset and compare them to theoretical predictions.
"""

import numpy as np

from ssbc import BinaryClassifierSimulator, evaluate_test_dataset, generate_rigorous_pac_report


def main():
    """Demonstrate evaluate_test_dataset() functionality."""
    print("=" * 80)
    print("EVALUATE TEST DATASET EXAMPLE")
    print("=" * 80)

    # Create a simulator for generating data
    sim = BinaryClassifierSimulator(
        p_class1=0.3,
        beta_params_class0=(2, 8),  # Class 0: low p(class=1) scores
        beta_params_class1=(8, 2),  # Class 1: high p(class=1) scores
        seed=42,
    )

    # Generate calibration data
    print("\n1. Generating calibration data...")
    cal_labels, cal_probs = sim.generate(n_samples=100)
    print(f"   Calibration data: {len(cal_labels)} samples")
    print(f"   Class distribution: {np.sum(cal_labels == 0)} class 0, {np.sum(cal_labels == 1)} class 1")

    # Generate PAC report to get thresholds
    print("\n2. Generating PAC report with operational bounds...")
    report = generate_rigorous_pac_report(
        labels=cal_labels, probs=cal_probs, alpha_target=0.10, delta=0.10, test_size=1000, verbose=False
    )

    # Extract thresholds
    threshold_0 = report["calibration_result"][0]["threshold"]
    threshold_1 = report["calibration_result"][1]["threshold"]
    print(f"   Threshold for class 0: {threshold_0:.4f}")
    print(f"   Threshold for class 1: {threshold_1:.4f}")

    # Generate test data
    print("\n3. Generating test data...")
    test_labels, test_probs = sim.generate(n_samples=200)
    print(f"   Test data: {len(test_labels)} samples")
    print(f"   Class distribution: {np.sum(test_labels == 0)} class 0, {np.sum(test_labels == 1)} class 1")

    # Evaluate test dataset
    print("\n4. Evaluating test dataset with empirical rates...")
    results = evaluate_test_dataset(
        test_labels=test_labels, test_probs=test_probs, threshold_0=threshold_0, threshold_1=threshold_1
    )

    # Print results
    print("\n" + "=" * 80)
    print("EMPIRICAL RATES FROM TEST DATASET")
    print("=" * 80)

    print(f"\nMarginal rates (all {results['n_test']} samples):")
    marginal = results["marginal"]
    print(f"  Singleton rate:  {marginal['singleton_rate']:.3f} ({marginal['n_singletons']}/{marginal['n_samples']})")
    print(f"  Doublet rate:    {marginal['doublet_rate']:.3f} ({marginal['n_doublets']}/{marginal['n_samples']})")
    print(f"  Abstention rate: {marginal['abstention_rate']:.3f} ({marginal['n_abstentions']}/{marginal['n_samples']})")
    if not np.isnan(marginal["singleton_error_rate"]):
        print(f"  Singleton error: {marginal['singleton_error_rate']:.3f}")
    else:
        print("  Singleton error: N/A (no singletons)")

    print(f"\nClass 0 rates ({results['class_0']['n_samples']} samples):")
    class_0 = results["class_0"]
    print(f"  Singleton rate:  {class_0['singleton_rate']:.3f} ({class_0['n_singletons']}/{class_0['n_samples']})")
    print(f"  Doublet rate:    {class_0['doublet_rate']:.3f} ({class_0['n_doublets']}/{class_0['n_samples']})")
    print(f"  Abstention rate: {class_0['abstention_rate']:.3f} ({class_0['n_abstentions']}/{class_0['n_samples']})")

    print(f"\nClass 1 rates ({results['class_1']['n_samples']} samples):")
    class_1 = results["class_1"]
    print(f"  Singleton rate:  {class_1['singleton_rate']:.3f} ({class_1['n_singletons']}/{class_1['n_samples']})")
    print(f"  Doublet rate:    {class_1['doublet_rate']:.3f} ({class_1['n_doublets']}/{class_1['n_samples']})")
    print(f"  Abstention rate: {class_1['abstention_rate']:.3f} ({class_1['n_abstentions']}/{class_1['n_samples']})")

    # Compare to theoretical bounds
    print("\n" + "=" * 80)
    print("COMPARISON TO THEORETICAL BOUNDS")
    print("=" * 80)

    pac_marg = report["pac_bounds_marginal"]
    print("\nMarginal bounds from PAC report:")
    print(
        f"  Singleton rate bounds: "
        f"[{pac_marg['singleton_rate_bounds'][0]:.3f}, {pac_marg['singleton_rate_bounds'][1]:.3f}]"
    )
    print(f"  Expected singleton rate: {pac_marg['expected_singleton_rate']:.3f}")
    print(f"  Empirical singleton rate: {marginal['singleton_rate']:.3f}")

    # Check if empirical rate is within bounds
    lower, upper = pac_marg["singleton_rate_bounds"]
    within_bounds = lower <= marginal["singleton_rate"] <= upper
    status = "✅ WITHIN BOUNDS" if within_bounds else "❌ OUTSIDE BOUNDS"
    print(f"  Status: {status}")

    print("\n" + "=" * 80)
    print("USAGE SUMMARY")
    print("=" * 80)
    print("\nThe evaluate_test_dataset() function provides:")
    print("✓ Easy evaluation of test datasets with conformal prediction")
    print("✓ Comprehensive rate computation (marginal and per-class)")
    print("✓ Direct comparison with theoretical PAC bounds")
    print("✓ Clean, structured output for reporting")
    print("\nUse this function to:")
    print("• Validate that your conformal prediction is working correctly")
    print("• Compare empirical performance to theoretical guarantees")
    print("• Generate operational statistics for reports")
    print("• Debug threshold selection and calibration issues")


if __name__ == "__main__":
    main()
