# Usage Guide

## Overview

SSBC (Small-Sample Beta Correction) provides tools for:
- **PAC coverage guarantees** for conformal prediction with finite samples
- **Mondrian conformal prediction** for class-conditional guarantees
- **PAC operational bounds** for deployment rate estimates (LOO-CV + Clopper-Pearson)
- **Uncertainty quantification** via bootstrap and cross-conformal validation
- **Statistical utilities** for exact binomial confidence intervals

## Installation

```bash
pip install ssbc
```

## Package Organization

SSBC is organized into focused packages that group related functionality:

- **`ssbc.core_pkg`**: Core SSBC algorithm (`ssbc_correct`, `SSBCResult`)
- **`ssbc.bounds`**: Statistical bounds computation (Clopper-Pearson, prediction bounds)
- **`ssbc.calibration`**: Conformal prediction and calibration (Mondrian CP, bootstrap, cross-conformal)
- **`ssbc.metrics`**: Operational metrics and uncertainty quantification (LOO-CV, operational bounds)
- **`ssbc.reporting`**: Reporting and visualization utilities
- **`ssbc.validation_pkg`**: Validation and empirical testing utilities

For most use cases, the recommended approach is to import from the top-level `ssbc` package:

```python
from ssbc import (
    ssbc_correct,
    mondrian_conformal_calibrate,
    generate_rigorous_pac_report,
    # ... other functions
)
```

This provides a stable API regardless of internal organization. For specialized use cases, you may also import directly from specific packages:

```python
from ssbc.calibration import split_by_class
from ssbc.metrics import compute_pac_operational_bounds_marginal
from ssbc.bounds import clopper_pearson_intervals
```

## Quick Start

### Unified Workflow (Recommended)

The complete rigorous workflow is available through a single function:

```python
from ssbc import BinaryClassifierSimulator, generate_rigorous_pac_report

# Generate or load calibration data
sim = BinaryClassifierSimulator(
    p_class1=0.2,
    beta_params_class0=(1, 7),
    beta_params_class1=(5, 2),
    seed=42
)
labels, probs = sim.generate(n_samples=100)

# Generate comprehensive PAC report
report = generate_rigorous_pac_report(
    labels=labels,
    probs=probs,
    alpha_target=0.10,     # Target 90% coverage
    delta=0.10,            # 90% PAC confidence
    test_size=1000,        # Expected deployment size
    use_union_bound=True,  # Simultaneous guarantees
    verbose=True,
)

# Access PAC bounds
marginal_bounds = report['pac_bounds_marginal']
class_0_bounds = report['pac_bounds_class_0']
class_1_bounds = report['pac_bounds_class_1']

print(f"Singleton rate: {marginal_bounds['singleton_rate_bounds']}")
print(f"Expected: {marginal_bounds['expected_singleton_rate']:.3f}")
```

### LOO-CV Uncertainty Methods

The rigorous report supports multiple methods for small-sample uncertainty quantification:

```python
report = generate_rigorous_pac_report(
    labels=labels,
    probs=probs,
    alpha_target=0.10,
    delta=0.10,
    test_size=1000,
    prediction_method="all",  # Compare analytical, exact, and Hoeffding methods
    use_loo_correction=True,
    loo_inflation_factor=2.0,  # Override default LOO variance inflation
)

# Access method comparison
marginal = report['pac_bounds_marginal']
if 'loo_diagnostics' in marginal:
    singleton_diag = marginal['loo_diagnostics'].get('singleton', {})
    if 'comparison' in singleton_diag:
        comparison = singleton_diag['comparison']
        print("Method comparison:")
        for method, lower, upper, width in zip(
            comparison['method'],
            comparison['lower'],
            comparison['upper'],
            comparison['width']
        ):
            print(f"  {method}: [{lower:.3f}, {upper:.3f}] (width: {width:.3f})")
```

## Core SSBC Algorithm

### Basic Correction

```python
from ssbc import ssbc_correct

# Correct miscoverage rate for finite-sample PAC guarantee
result = ssbc_correct(
    alpha_target=0.10,  # Target 10% miscoverage
    n=100,              # Calibration set size
    delta=0.05,         # 95% PAC guarantee
    mode="beta"         # Infinite test window
)

print(f"Corrected alpha: {result.alpha_corrected:.4f}")
print(f"Use u* = {result.u_star} as threshold index")
```

### Parameters

- `alpha_target`: Target miscoverage rate (e.g., 0.10 for 90% coverage)
- `n`: Calibration set size
- `delta`: PAC risk tolerance (probability of violating guarantee)
- `mode`: "beta" (infinite test) or "beta-binomial" (finite test)

## Mondrian Conformal Prediction

### Basic Workflow

```python
from ssbc import split_by_class, mondrian_conformal_calibrate

# Split data by class for Mondrian CP
class_data = split_by_class(labels, probs)

# Calibrate with SSBC correction
cal_result, pred_stats = mondrian_conformal_calibrate(
    class_data=class_data,
    alpha_target=0.10,  # Target 90% coverage per class
    delta=0.10,         # 90% PAC guarantee
    mode="beta"
)

# View thresholds
for label in [0, 1]:
    print(f"Class {label}:")
    print(f"  Threshold: {cal_result[label]['threshold']:.4f}")
    print(f"  Corrected α: {cal_result[label]['alpha_corrected']:.4f}")
```

## Alpha Scan Analysis

Analyze how prediction set statistics vary across all possible alpha thresholds:

```python
from ssbc import alpha_scan

# Scan all possible alpha thresholds
df = alpha_scan(labels, probs)

print(f"Scanned {len(df)} alpha values")
print(df.head())

# Find optimal operating point
max_singleton_idx = df['n_singletons'].idxmax()
optimal = df.loc[max_singleton_idx]
print(f"\nMaximum singleton rate at alpha={optimal['alpha']:.4f}:")
print(f"  Singletons: {optimal['n_singletons']}")
print(f"  Singleton coverage: {optimal['singleton_coverage']:.4f}")
```

**DataFrame columns:**
- `alpha`: miscoverage rate
- `qhat_0`, `qhat_1`: per-class thresholds
- `n_abstentions`, `n_singletons`, `n_doublets`: prediction set counts
- `singleton_coverage`: fraction of singletons that are correct
- `singleton_coverage_0`, `singleton_coverage_1`: per-class singleton coverage

## Uncertainty Quantification

### Standalone Diagnostic Tools

Bootstrap and cross-conformal validation are available as standalone diagnostic tools (not integrated into the main PAC bounds computation):

### Bootstrap Calibration Uncertainty

Analyze how operational rates vary if you recalibrate on similar datasets:

```python
from ssbc import bootstrap_calibration_uncertainty, plot_bootstrap_distributions

results = bootstrap_calibration_uncertainty(
    labels=labels,
    probs=probs,
    simulator=sim,
    n_bootstrap=1000,
    test_size=1000
)

# Visualize distributions
plot_bootstrap_distributions(results, save_path='bootstrap_results.png')
```

### Cross-Conformal Validation

K-fold cross-validation for finite-sample diagnostics:

```python
from ssbc import cross_conformal_validation

results = cross_conformal_validation(
    labels=labels,
    probs=probs,
    n_folds=10,
    alpha_target=0.10,
    delta=0.10
)

print(f"Singleton rate: {results['marginal']['singleton']['mean']:.3f} ± {results['marginal']['singleton']['std']:.3f}")
```

### Validation with Class-Conditional Error Metrics

The validation framework includes class-conditional singleton error metrics:

```python
from ssbc import validate_pac_bounds, validate_prediction_interval_calibration

# Single validation
validation = validate_pac_bounds(report, simulator=sim, test_size=1000, n_trials=10000)

# Access class-conditional metrics (marginal scope only)
marginal = validation['marginal']
print(f"Error rate (class 0, normalized): {marginal['singleton_error_class0']['mean']:.3f}")
print(f"Error rate (class 1, normalized): {marginal['singleton_error_class1']['mean']:.3f}")
print(f"P(error | singleton & class=0): {marginal['singleton_error_cond_class0']['mean']:.3f}")
print(f"P(error | singleton & class=1): {marginal['singleton_error_cond_class1']['mean']:.3f}")

# Meta-validation across many calibrations
results = validate_prediction_interval_calibration(
    simulator=sim,
    n_calibration=100,
    BigN=500,
    n_trials=1000,
    prediction_method="all",
)
```

### Empirical Validation

Verify theoretical PAC guarantees empirically:

```python
from ssbc import validate_pac_bounds, print_validation_results

# Generate report
report = generate_rigorous_pac_report(labels, probs, delta=0.10)

# Validate with many test trials
validation = validate_pac_bounds(
    report=report,
    simulator=sim,
    test_size=1000,
    n_trials=10000,
)

# Print validation results
print_validation_results(validation)

# Check coverage
coverage = validation['marginal']['singleton']['empirical_coverage']
pac_level = report['parameters']['pac_level_marginal']
if coverage >= pac_level:
    print(f"✅ Validation passed: {coverage:.1%} >= {pac_level:.1%}")
```

## Statistical Utilities

### Clopper-Pearson Confidence Intervals

```python
from ssbc import clopper_pearson_lower, clopper_pearson_upper, cp_interval

# One-sided bounds
lower = clopper_pearson_lower(k=45, n=100, confidence=0.95)
upper = clopper_pearson_upper(k=45, n=100, confidence=0.95)

# Two-sided interval
interval = cp_interval(count=45, total=100, confidence=0.95)
print(f"Rate: {interval['proportion']:.3f}")
print(f"95% CI: [{interval['lower']:.3f}, {interval['upper']:.3f}]")
```

### Operational Rate Computation

```python
from ssbc import compute_operational_rate
import numpy as np

# Example prediction sets
pred_sets = [{0}, {0, 1}, set(), {1}, {0}]
true_labels = np.array([0, 0, 1, 1, 0])

# Compute indicators for different rates
singleton_indicators = compute_operational_rate(
    pred_sets, true_labels, "singleton"
)
error_indicators = compute_operational_rate(
    pred_sets, true_labels, "error_in_singleton"
)

print(f"Singleton rate: {np.mean(singleton_indicators):.2%}")
print(f"Error rate: {np.mean(error_indicators):.2%}")
```

**Supported rate types:**
- `"singleton"`: Single predicted label
- `"doublet"`: Two predicted labels
- `"abstention"`: Empty prediction set
- `"error_in_singleton"`: Singleton with incorrect prediction
- `"correct_in_singleton"`: Singleton with correct prediction

## Hyperparameter Tuning

Sweep over α and δ values to find optimal configurations:

```python
from ssbc import sweep_and_plot_parallel_plotly
import numpy as np

# Define grid
alpha_grid = np.arange(0.05, 0.20, 0.05)
delta_grid = np.arange(0.05, 0.20, 0.05)

# Split data by class
class_data = split_by_class(labels, probs)

# Run sweep and visualize
df, fig = sweep_and_plot_parallel_plotly(
    class_data=class_data,
    alpha_0=alpha_grid, delta_0=delta_grid,
    alpha_1=alpha_grid, delta_1=delta_grid,
    color='err_all'  # Color by error rate
)

# Save interactive plot
fig.write_html("sweep_results.html")

# Analyze results
print(df[['a0', 'd0', 'cov', 'sing_rate', 'err_all']].head())
```

The interactive plot allows you to:
- Brush (select) ranges on any axis to filter configurations
- Explore trade-offs between coverage, automation, and error rates
- Identify Pareto-optimal hyperparameter settings

## Understanding Report Components

### PAC Report Structure

```python
report = generate_rigorous_pac_report(labels, probs)

# SSBC results for each class
ssbc_0 = report['ssbc_class_0']  # SSBCResult
ssbc_1 = report['ssbc_class_1']  # SSBCResult

# PAC operational bounds
marginal_bounds = report['pac_bounds_marginal']  # Marginal statistics
class_0_bounds = report['pac_bounds_class_0']    # Class 0 conditional
class_1_bounds = report['pac_bounds_class_1']    # Class 1 conditional

# Calibration results
cal_result = report['calibration_result']  # Thresholds per class
pred_stats = report['prediction_stats']    # Prediction statistics

# LOO diagnostics and method comparison (if prediction_method="all")
loo_diagnostics = bounds.get('loo_diagnostics', {})

# Parameters used
params = report['parameters']
```

### PAC Bounds Dictionary

Each PAC bounds dictionary contains:

```python
bounds = report['pac_bounds_marginal']

# Bounds (as lists [lower, upper])
bounds['singleton_rate_bounds']       # [lower, upper]
bounds['doublet_rate_bounds']         # [lower, upper]
bounds['abstention_rate_bounds']      # [lower, upper]
bounds['singleton_error_rate_bounds'] # [lower, upper]

# Class-conditional error metrics (marginal bounds only)
bounds.get('singleton_error_rate_class0_bounds', None)      # [lower, upper]
bounds.get('singleton_error_rate_class1_bounds', None)      # [lower, upper]
bounds.get('singleton_error_rate_cond_class0_bounds', None) # [lower, upper]
bounds.get('singleton_error_rate_cond_class1_bounds', None) # [lower, upper]

# Expected values (from LOO-CV)
bounds['expected_singleton_rate']
bounds['expected_doublet_rate']
bounds['expected_abstention_rate']
bounds['expected_singleton_error_rate']
bounds.get('expected_singleton_error_rate_class0', None)
bounds.get('expected_singleton_error_rate_class1', None)
bounds.get('expected_singleton_error_rate_cond_class0', None)
bounds.get('expected_singleton_error_rate_cond_class1', None)

# Metadata
bounds['n_grid_points']  # Number of grid points evaluated
bounds['pac_level']      # PAC confidence level
bounds['ci_level']       # Clopper-Pearson CI level
```

## Key Concepts

### PAC Coverage (from SSBC)

**Guarantee:** With probability ≥ 1-δ over calibration sets, the conformal predictor
achieves coverage ≥ 1-α_target on future data.

**Properties:**
- Valid for ANY sample size n
- Distribution-free
- Frequentist (no priors)

### PAC Operational Bounds (LOO-CV + Clopper-Pearson)

**Estimates:** Rigorous bounds on deployment rates accounting for estimation uncertainty.

**Procedure:**
1. For each calibration point i, compute threshold using all OTHER points (LOO-CV)
2. Evaluate point i with that threshold (unbiased evaluation)
3. Aggregate counts across all n evaluations
4. Apply Clopper-Pearson confidence intervals to bound the true rate

**Properties:**
- Unbiased estimates (LOO ensures no data leakage)
- Exact binomial CIs (Clopper-Pearson)
- Accounts for estimation uncertainty from finite calibration
- Valid for any future test set from same distribution

### PAC Bounds (LOO-CV + Prediction Bounds)

**PAC Bounds (LOO-CV + Clopper-Pearson):**
- Question: "Given THIS calibration, what rates on future test sets?"
- Accounts for: Estimation uncertainty and test set sampling variability
- Methods: Analytical (recommended n≥40), Exact (n=20-40), Hoeffding (ultra-conservative)
- Use for: Deployment guarantees, SLA contracts
- Validates: Operational rates (singleton, doublet, abstention, error rates) and class-conditional error metrics

### Marginal vs Per-Class

**Marginal bounds** (ignore true labels):
- "What will a user see?"
- Deployment view
- Overall automation rate

**Per-class bounds** (conditioned on true label):
- "How does performance differ by ground truth?"
- Class-specific rates
- Identifies minority class challenges

## Examples

Complete examples are available in the `examples/` directory:

### 1. Core SSBC Algorithm
```bash
python examples/ssbc_core_example.py
```
Demonstrates the SSBC algorithm for different calibration set sizes.

### 2. Mondrian Conformal Prediction
```bash
python examples/mondrian_conformal_example.py
```
Complete workflow: simulation → calibration → per-class reporting.

### 3. Complete Workflow with PAC Operational Bounds
```bash
python examples/complete_workflow_example.py
```
Shows the complete PAC bounds workflow using `generate_rigorous_pac_report()`.

### 4. SLA/Deployment Contracts
```bash
python examples/sla_example.py
```
Full deployment pipeline with contract-ready operational guarantees.

### 5. Alpha Scan Analysis
```bash
python examples/alpha_scan_example.py
```
Scan across all possible alpha thresholds to find optimal operating points.

### 6. PAC Bounds Validation
```bash
python examples/pac_validation_example.py
```
Empirically validate that theoretical PAC guarantees hold in practice.

### 7. Bootstrap Demo (Standalone Diagnostic)
```bash
python examples/bootstrap_calibration_demo.py
```
Standalone bootstrap analysis with detailed visualization. This is a diagnostic tool, not integrated into PAC bounds.

### 8. Cross-Conformal Validation (Standalone Diagnostic)
```bash
python examples/cross_conformal_example.py
```
K-fold cross-validation for finite-sample diagnostics. This is a diagnostic tool, not integrated into PAC bounds.

## References

### Key Statistical Properties

- **Distribution-Free**: No P(X,Y) assumptions
- **Model-Agnostic**: Works with any classifier
- **Frequentist**: Valid frequentist guarantees
- **Non-Bayesian**: No priors required
- **Finite-Sample**: Exact guarantees for small n (not asymptotic)
- **Exchangeability Only**: Minimal assumption

### Further Reading

- See [theory.md](theory.md) for detailed theoretical background
- See [installation.md](installation.md) for setup instructions
- See `examples/` directory for complete working examples
