"""Example: Hyperparameter Sweep with Interactive Visualization.

This example demonstrates hyperparameter tuning for Mondrian conformal
prediction with SSBC correction using parallel coordinates plots.
"""

import numpy as np

from ssbc import (
    BinaryClassifierSimulator,
    split_by_class,
    sweep_and_plot_parallel_plotly,
)


def main():
    """Run hyperparameter sweep example."""

    print("=" * 80)
    print("HYPERPARAMETER SWEEP FOR MONDRIAN CONFORMAL PREDICTION")
    print("=" * 80)

    # ========== Step 1: Generate Data ==========
    print("\n1. Generating simulated binary classification data...")

    # Imbalanced classes with good classifier
    sim = BinaryClassifierSimulator(
        p_class1=0.10,  # 10% positive class
        beta_params_class0=(2, 8),  # Class 0: low P(1) scores
        beta_params_class1=(8, 2),  # Class 1: high P(1) scores
        seed=42,
    )

    labels, probs = sim.generate(n_samples=1000)
    class_data = split_by_class(labels, probs)

    print(f"   Generated {len(labels)} samples")
    print(f"   Class 0: {class_data[0]['n']} samples")
    print(f"   Class 1: {class_data[1]['n']} samples")

    # ========== Step 2: Define Hyperparameter Grid ==========
    print("\n2. Defining hyperparameter grid...")

    alpha_grid = np.arange(0.05, 0.20, 0.05)
    delta_grid = np.arange(0.05, 0.20, 0.05)

    print(f"   α values: {alpha_grid}")
    print(f"   δ values: {delta_grid}")
    print(f"   Total configurations: {len(alpha_grid) ** 2 * len(delta_grid) ** 2}")

    # ========== Step 3: Run Sweep ==========
    print("\n3. Running hyperparameter sweep...")
    print("   (This may take a moment...)")

    df, fig = sweep_and_plot_parallel_plotly(
        class_data=class_data,
        delta_0=delta_grid,
        delta_1=delta_grid,
        alpha_0=alpha_grid,
        alpha_1=alpha_grid,
        mode="beta",
        color="err_all",  # Color by overall singleton error rate
        title="Mondrian Conformal Prediction - Hyperparameter Sweep",
    )

    print(f"   ✓ Sweep completed: {len(df)} configurations evaluated")

    # ========== Step 4: Analyze Results ==========
    print("\n4. Analyzing results...")

    print("\nKey Metrics Summary:")
    print(f"   Coverage:      mean={df['cov'].mean():.3f}, min={df['cov'].min():.3f}, max={df['cov'].max():.3f}")
    print(
        f"   Singleton rate: mean={df['sing_rate'].mean():.3f}, "
        f"min={df['sing_rate'].min():.3f}, max={df['sing_rate'].max():.3f}"
    )
    print(
        f"   Error rate:    mean={df['err_all'].mean():.3f}, "
        f"min={df['err_all'].min():.3f}, max={df['err_all'].max():.3f}"
    )
    print(
        f"   Escalation:    mean={df['esc_rate'].mean():.3f}, "
        f"min={df['esc_rate'].min():.3f}, max={df['esc_rate'].max():.3f}"
    )

    # Find best configurations based on different criteria
    print("\n5. Best Configurations:")

    # Highest singleton rate (most automation)
    best_automation = df.loc[df["sing_rate"].idxmax()]
    print("\n   a) Highest automation (singleton rate):")
    print(
        f"      α0={best_automation['a0']:.2f}, δ0={best_automation['d0']:.2f}, "
        f"α1={best_automation['a1']:.2f}, δ1={best_automation['d1']:.2f}"
    )
    print(f"      Singleton rate: {best_automation['sing_rate']:.3f}")
    print(f"      Error rate:     {best_automation['err_all']:.3f}")
    print(f"      Coverage:       {best_automation['cov']:.3f}")

    # Lowest error rate among configs with >70% singletons
    high_sing = df[df["sing_rate"] > 0.70]
    if len(high_sing) > 0:
        best_quality = high_sing.loc[high_sing["err_all"].idxmin()]
        print("\n   b) Best quality (lowest error, >70% singleton rate):")
        print(
            f"      α0={best_quality['a0']:.2f}, δ0={best_quality['d0']:.2f}, "
            f"α1={best_quality['a1']:.2f}, δ1={best_quality['d1']:.2f}"
        )
        print(f"      Singleton rate: {best_quality['sing_rate']:.3f}")
        print(f"      Error rate:     {best_quality['err_all']:.3f}")
        print(f"      Coverage:       {best_quality['cov']:.3f}")

    # Balanced: good coverage with reasonable automation
    # Target: coverage > 0.90, maximize singletons
    balanced = df[df["cov"] >= 0.90]
    if len(balanced) > 0:
        best_balanced = balanced.loc[balanced["sing_rate"].idxmax()]
        print("\n   c) Balanced (coverage ≥ 90%, maximize automation):")
        print(
            f"      α0={best_balanced['a0']:.2f}, δ0={best_balanced['d0']:.2f}, "
            f"α1={best_balanced['a1']:.2f}, δ1={best_balanced['d1']:.2f}"
        )
        print(f"      Singleton rate: {best_balanced['sing_rate']:.3f}")
        print(f"      Error rate:     {best_balanced['err_all']:.3f}")
        print(f"      Coverage:       {best_balanced['cov']:.3f}")

    # ========== Step 6: Save Results ==========
    print("\n6. Saving results...")

    # Save dataframe
    csv_path = "examples/hyperparameter_sweep_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"   CSV saved to: {csv_path}")

    # Save interactive plot
    html_path = "examples/hyperparameter_sweep_plot.html"
    fig.write_html(html_path)
    print(f"   Interactive plot saved to: {html_path}")
    print("\n   Open the HTML file in a browser to:")
    print("   - Brush (select) ranges on any axis to filter configurations")
    print("   - See how different hyperparameters affect metrics")
    print("   - Identify Pareto-optimal configurations")

    # ========== Step 7: Display in Notebook (if running in Jupyter) ==========
    try:
        # Check if running in Jupyter/IPython
        get_ipython()  # type: ignore  # noqa: F821
        print("\n7. Displaying interactive plot...")
        fig.show()
    except NameError:
        # Not in Jupyter
        print("\n   Note: To view the plot interactively, either:")
        print(f"         - Open {html_path} in a web browser")
        print("         - Run this script in a Jupyter notebook")

    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Open the HTML plot to explore the hyperparameter space")
    print("  2. Analyze the CSV file for detailed results")
    print("  3. Choose hyperparameters based on your deployment requirements:")
    print("     - High coverage: minimize false negatives")
    print("     - High automation: maximize singleton predictions")
    print("     - Low errors: minimize singleton error rate")


if __name__ == "__main__":
    main()
