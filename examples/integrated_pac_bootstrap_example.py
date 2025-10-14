"""Integrated example showing both PAC bounds and bootstrap uncertainty.

This demonstrates the COMPLETE workflow:
1. PAC-controlled operational bounds (LOO-CV + CP) - Given THIS calibration
2. Bootstrap calibration uncertainty - If we RECALIBRATE on similar data

Both analyses are complementary and answer different questions!
"""

from ssbc import BinaryClassifierSimulator, generate_rigorous_pac_report, plot_bootstrap_distributions

# Create simulator
sim = BinaryClassifierSimulator(p_class1=0.20, beta_params_class0=(1, 7), beta_params_class1=(5, 2), seed=42)

# Generate calibration data
labels, probs = sim.generate(n_samples=100)

print("=" * 80)
print("INTEGRATED PAC + BOOTSTRAP ANALYSIS")
print("=" * 80)

# Generate comprehensive report with BOTH analyses
report = generate_rigorous_pac_report(
    labels=labels,
    probs=probs,
    alpha_target=0.10,
    delta=0.10,
    test_size=1000,
    ci_level=0.95,
    use_union_bound=True,
    n_jobs=-1,
    verbose=True,
    run_bootstrap=True,  # Enable bootstrap analysis
    n_bootstrap=1000,  # Number of bootstrap trials
    simulator=sim,  # Required for bootstrap
)

print("\n" + "=" * 80)
print("INTERPRETATION")
print("=" * 80)

# Extract bounds
pac_class_0 = report["pac_bounds_class_0"]
bootstrap_class_0 = report["bootstrap_results"]["class_0"]

pac_singleton = pac_class_0["singleton_rate_bounds"]
bootstrap_singleton = bootstrap_class_0["singleton"]["quantiles"]

print("\nCLASS 0 SINGLETON RATE:")
print(f"  PAC bounds (LOO-CV + CP):    [{pac_singleton[0]:.3f}, {pac_singleton[1]:.3f}]")
print(f"  Bootstrap [5%, 95%]:         [{bootstrap_singleton['q05']:.3f}, {bootstrap_singleton['q95']:.3f}]")

print("\n✓ PAC bounds are TIGHTER:")
print("  → Model: 'Given THIS specific calibration, what rates on future test sets?'")
print("  → Account for: Test set sampling volatility")
print("  → Valid for: Any test set from same distribution")

print("\n✓ Bootstrap intervals are WIDER:")
print("  → Model: 'If I recalibrate on similar datasets, how do rates vary?'")
print("  → Account for: Calibration-to-calibration variability")
print("  → Valid for: Understanding recalibration uncertainty")

print("\n✓ Both are RIGOROUS and COMPLEMENTARY!")
print("  → Use PAC bounds for deployment guarantees")
print("  → Use bootstrap to understand calibration sensitivity")

# Optional: Plot bootstrap distributions
print("\n" + "=" * 80)
print("OPTIONAL: Bootstrap Visualization")
print("=" * 80)

try:
    # Extract bootstrap results from report
    bootstrap_results = report["bootstrap_results"]
    if bootstrap_results is not None:
        print("\nCreating bootstrap visualization...")
        plot_bootstrap_distributions(
            bootstrap_results,
            save_path="integrated_bootstrap_results.png"
        )
        print("✅ Visualization saved!")
    else:
        print("❌ No bootstrap results found (run_bootstrap=False)")
except ImportError as e:
    print(f"⚠️  Could not create visualization: {e}")
    print("   Install matplotlib to enable plotting: pip install matplotlib")

print("\n" + "=" * 80)

