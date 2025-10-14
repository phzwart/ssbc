"""Cross-conformal validation example.

Demonstrates K-fold cross-validation for estimating rate variability in
Mondrian conformal prediction due to finite calibration samples.
"""

import numpy as np

from ssbc import BinaryClassifierSimulator, cross_conformal_validation, print_cross_conformal_results

# Create simulator
sim = BinaryClassifierSimulator(p_class1=0.20, beta_params_class0=(1, 7), beta_params_class1=(5, 2), seed=42)

# Generate calibration data
print("=" * 80)
print("CROSS-CONFORMAL VALIDATION")
print("=" * 80)
print("\nGenerating calibration data...")

labels, probs = sim.generate(n_samples=100)

print(f"Calibration size: {len(labels)}")
print(f"Class 0: {np.sum(labels == 0)}, Class 1: {np.sum(labels == 1)}")

# Run cross-conformal validation
print("\nRunning 10-fold cross-conformal validation...")

results = cross_conformal_validation(
    labels=labels,
    probs=probs,
    alpha_target=0.10,
    delta=0.10,
    n_folds=10,
    stratify=True,
    seed=123,
)

# Print results
print_cross_conformal_results(results)

# Compare to expectations
print("\n" + "=" * 80)
print("COMPARISON: Cross-Conformal vs Other Methods")
print("=" * 80)

print("\n📊 Cross-Conformal (K-fold splits of calibration):")
singleton_cc = results["marginal"]["singleton"]
print(f"  Singleton rate: {singleton_cc['mean']:.3f} ± {singleton_cc['std']:.3f}")
print(f"  [5%, 95%] range: [{singleton_cc['quantiles']['q05']:.3f}, {singleton_cc['quantiles']['q95']:.3f}]")
print("  → Shows variability from finite calibration splits")

print("\n🔄 How this compares to other methods:")
print("  • LOO-CV bounds: Account for estimation uncertainty, aggregate counts")
print("  • Bootstrap: Recalibration uncertainty with fresh test sets")
print("  • Cross-conformal: Rate distribution from K-fold calibration splits")

print("\n✓ Use cross-conformal to:")
print("  1. Understand rate stability across different calibration subsets")
print("  2. Estimate variability without needing a data simulator")
print("  3. Diagnose if more calibration data is needed (large std → need more data)")

print("\n" + "=" * 80)
