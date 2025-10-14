"""Service Level Agreement (SLA) example using rigorous PAC-controlled operational bounds.

This example demonstrates how to use SSBC for deployment scenarios where you need:
1. PAC-guaranteed coverage (via SSBC)
2. Rigorous operational rate bounds (via LOO-CV + Clopper-Pearson)
3. Contract-ready guarantees for production deployment

Updated for v0.2.0: Uses unified generate_rigorous_pac_report() workflow.
"""

import numpy as np

from ssbc import BinaryClassifierSimulator, generate_rigorous_pac_report

# Simulate a classifier for deployment
print("=" * 80)
print("SERVICE LEVEL AGREEMENT (SLA) - RIGOROUS OPERATIONAL BOUNDS")
print("=" * 80)

# Create simulator
sim = BinaryClassifierSimulator(
    p_class1=0.3,
    beta_params_class0=(2, 5),  # Moderate quality for class 0
    beta_params_class1=(6, 2),  # Good quality for class 1
    seed=42,
)

# Generate calibration data
labels, probs = sim.generate(n_samples=200)

print(f"\nCalibration data: {len(labels)} samples")
print(f"  Class 0: {np.sum(labels == 0)} samples")
print(f"  Class 1: {np.sum(labels == 1)} samples")

# Define SLA requirements
ALPHA_TARGET = 0.10  # Maximum 10% miscoverage
DELTA = 0.05  # 95% PAC confidence (1 - δ)
TEST_SIZE = 1000  # Expected deployment test set size

print("\nSLA Requirements:")
print(f"  Maximum miscoverage: {ALPHA_TARGET:.1%}")
print(f"  PAC confidence: {1-DELTA:.0%} (δ = {DELTA})")
print(f"  Expected test set size: {TEST_SIZE}")

# Generate rigorous PAC report with operational bounds
print("\n" + "=" * 80)
print("GENERATING RIGOROUS PAC REPORT WITH OPERATIONAL BOUNDS")
print("=" * 80)

report = generate_rigorous_pac_report(
    labels=labels,
    probs=probs,
    alpha_target=ALPHA_TARGET,
    delta=DELTA,
    test_size=TEST_SIZE,
    ci_level=0.95,
    use_union_bound=True,
    n_jobs=-1,
    verbose=True,
)

# Extract operational bounds for SLA contract
print("\n" + "=" * 80)
print("SLA CONTRACT - OPERATIONAL GUARANTEES")
print("=" * 80)

# Marginal operational bounds (deployment view)
marginal_bounds = report["pac_bounds_marginal"]

print("\n📋 MARGINAL OPERATIONAL GUARANTEES (All Samples):")
print(f"   PAC Confidence Level: {report['parameters']['pac_level_marginal']:.0%}")

singleton_lower, singleton_upper = marginal_bounds["singleton_rate_bounds"]
doublet_lower, doublet_upper = marginal_bounds["doublet_rate_bounds"]
abstention_lower, abstention_upper = marginal_bounds["abstention_rate_bounds"]
error_lower, error_upper = marginal_bounds["singleton_error_rate_bounds"]

print("\n   AUTOMATION RATE (Singletons):")
print(f"     Guaranteed range: [{singleton_lower:.1%}, {singleton_upper:.1%}]")
print(f"     Expected: {marginal_bounds['expected_singleton_rate']:.1%}")

print("\n   ESCALATION RATE (Doublets + Abstentions):")
total_escalation_lower = doublet_lower + abstention_lower
total_escalation_upper = doublet_upper + abstention_upper
print(f"     Guaranteed range: [{total_escalation_lower:.1%}, {total_escalation_upper:.1%}]")
print(f"     Expected: {marginal_bounds['expected_doublet_rate'] + marginal_bounds['expected_abstention_rate']:.1%}")

print("\n   ERROR RATE (Among Automated Decisions):")
print(f"     Guaranteed range: [{error_lower:.1%}, {error_upper:.1%}]")
print(f"     Expected: {marginal_bounds['expected_singleton_error_rate']:.1%}")

# Per-class guarantees
print("\n📋 PER-CLASS OPERATIONAL GUARANTEES:")

for class_label in [0, 1]:
    class_bounds = report[f"pac_bounds_class_{class_label}"]
    ssbc = report[f"ssbc_class_{class_label}"]
    pac_level = report["parameters"][f"pac_level_{class_label}"]

    s_lower, s_upper = class_bounds["singleton_rate_bounds"]
    d_lower, d_upper = class_bounds["doublet_rate_bounds"]
    a_lower, a_upper = class_bounds["abstention_rate_bounds"]
    e_lower, e_upper = class_bounds["singleton_error_rate_bounds"]

    print(f"\n   CLASS {class_label} (PAC Confidence: {pac_level:.0%}):")
    print(f"     Singleton:  [{s_lower:.1%}, {s_upper:.1%}]")
    print(f"     Doublet:    [{d_lower:.1%}, {d_upper:.1%}]")
    print(f"     Abstention: [{a_lower:.1%}, {a_upper:.1%}]")
    print(f"     Error:      [{e_lower:.1%}, {e_upper:.1%}]")

# SLA contract summary
print("\n" + "=" * 80)
print("SLA CONTRACT SUMMARY")
print("=" * 80)

print("\n✅ COVERAGE GUARANTEE:")
print(f"   With {1-DELTA:.0%} confidence, coverage ≥ {1-ALPHA_TARGET:.0%} for both classes")

print(f"\n✅ OPERATIONAL GUARANTEES (Marginal, {report['parameters']['pac_level_marginal']:.0%} confidence):")
print(f"   • Automation rate: {singleton_lower:.1%} - {singleton_upper:.1%}")
print(f"   • Escalation rate: {total_escalation_lower:.1%} - {total_escalation_upper:.1%}")
print(f"   • Error rate (automated): ≤ {error_upper:.1%}")

print("\n✅ MONITORING THRESHOLDS:")
print(f"   Alert if observed automation rate < {singleton_lower:.1%}")
print(f"   Alert if observed error rate > {error_upper:.1%}")

print("\n✅ DEPLOYMENT RECOMMENDATION:")
if singleton_lower >= 0.70 and error_upper <= 0.10:
    print("   ✓ APPROVED: Meets automation (≥70%) and error (≤10%) requirements")
elif singleton_lower >= 0.70:
    print(f"   ⚠️  CONDITIONAL: Good automation but error rate up to {error_upper:.1%}")
elif error_upper <= 0.10:
    print(f"   ⚠️  CONDITIONAL: Low error but automation only {singleton_lower:.1%}")
else:
    print("   ❌ REJECTED: Does not meet minimum requirements")

print("\n" + "=" * 80)
print("TECHNICAL NOTES")
print("=" * 80)

print("\n✓ All bounds are rigorous:")
print("  • Coverage: PAC guarantee via SSBC")
print("  • Operational rates: LOO-CV + Clopper-Pearson")
print("  • Union bound: All metrics hold simultaneously")

print("\n✓ Deployment-ready:")
print("  • Bounds valid for any future test set from same distribution")
print("  • Can be used in legal/contractual SLA agreements")
print("  • Monitoring thresholds provide early warning system")

print("\n" + "=" * 80)
