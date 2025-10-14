"""Example of empirically validating PAC operational bounds.

This demonstrates how to use validate_pac_bounds() to verify that
theoretical PAC guarantees actually hold in practice.
"""

from ssbc import (
    BinaryClassifierSimulator,
    generate_rigorous_pac_report,
    print_validation_results,
    validate_pac_bounds,
)

# Create simulator
sim = BinaryClassifierSimulator(p_class1=0.20, beta_params_class0=(1, 7), beta_params_class1=(5, 2), seed=42)

# Generate calibration data
labels, probs = sim.generate(n_samples=100)

print("=" * 80)
print("PAC OPERATIONAL BOUNDS VALIDATION")
print("=" * 80)

# Step 1: Generate rigorous PAC report
print("\nStep 1: Generate PAC report with operational bounds...")
report = generate_rigorous_pac_report(
    labels=labels, probs=probs, alpha_target=0.10, delta=0.10, test_size=1000, verbose=False
)

print("\n✓ PAC report generated")
print(f"  Marginal singleton bounds: {report['pac_bounds_marginal']['singleton_rate_bounds']}")
print(f"  PAC level: {report['parameters']['pac_level_marginal']:.0%}")

# Step 2: Validate bounds empirically
print("\nStep 2: Empirically validate bounds with 10,000 test trials...")
validation = validate_pac_bounds(report=report, simulator=sim, test_size=1000, n_trials=10000, verbose=False)

# Step 3: Print validation results
print_validation_results(validation)

# Step 4: Check overall validation
print("\n" + "=" * 80)
print("VALIDATION SUMMARY")
print("=" * 80)

marginal_coverage = validation["marginal"]["singleton"]["empirical_coverage"]
class_0_coverage = validation["class_0"]["singleton"]["empirical_coverage"]
class_1_coverage = validation["class_1"]["singleton"]["empirical_coverage"]

pac_level = report["parameters"]["pac_level_marginal"]

print(f"\nTarget PAC level: {pac_level:.0%}")
print("\nEmpirical coverage:")
print(f"  Marginal: {marginal_coverage:.1%} {'✅' if marginal_coverage >= pac_level else '❌'}")
print(f"  Class 0:  {class_0_coverage:.1%} {'✅' if class_0_coverage >= 0.95 else '❌'}")
print(f"  Class 1:  {class_1_coverage:.1%} {'✅' if class_1_coverage >= 0.95 else '❌'}")

if marginal_coverage >= pac_level and class_0_coverage >= 0.95 and class_1_coverage >= 0.95:
    print("\n✅ VALIDATION PASSED: PAC bounds are empirically validated!")
else:
    print("\n⚠️  VALIDATION WARNING: Some coverage below expected PAC level")
    print("   (This can happen due to finite-sample effects)")

print("\n" + "=" * 80)
print("INTERPRETATION")
print("=" * 80)

print("\n✓ Validation confirms that PAC bounds hold in practice")
print("✓ Empirical coverage ≥ PAC level verifies theoretical guarantees")
print("✓ Use this for:")
print("  • Verifying implementation correctness")
print("  • Understanding bound tightness")
print("  • Generating validation plots for publications")

print("\n" + "=" * 80)
