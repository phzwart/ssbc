# Usage Guide

## Overview

SSBC (Small-Sample Beta Correction) provides tools for:
- **PAC coverage guarantees** for conformal prediction with finite samples
- **Mondrian conformal prediction** for class-conditional guarantees
- **LOO-CV operational bounds** for deployment rate estimates
- **Statistical utilities** for exact binomial confidence intervals

## Installation

```bash
pip install ssbc
```

## Quick Start

### Basic SSBC Correction

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

### Mondrian Conformal Prediction

```python
import numpy as np
from ssbc import split_by_class, mondrian_conformal_calibrate

# Prepare data (labels and probabilities from your classifier)
labels = np.array([0, 1, 0, 1, 0, 1])
probs = np.array([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1],
                   [0.2, 0.8], [0.85, 0.15], [0.1, 0.9]])

# Split by class for Mondrian CP
class_data = split_by_class(labels, probs)

# Calibrate with SSBC correction
cal_result, pred_stats = mondrian_conformal_calibrate(
    class_data=class_data,
    alpha_target=0.10,  # Target 90% coverage per class
    delta=0.10,         # 90% PAC guarantee
    mode="beta"
)

# View results
for label in [0, 1]:
    print(f"Class {label}:")
    print(f"  Threshold: {cal_result[label]['threshold']:.4f}")
    print(f"  Corrected α: {cal_result[label]['alpha_corrected']:.4f}")
```

## Complete Deployment Pipeline

### Step 1: Calibrate with SSBC (PAC Coverage)

```python
from ssbc import (
    BinaryClassifierSimulator,
    split_by_class,
    mondrian_conformal_calibrate,
    compute_mondrian_operational_bounds,
    compute_marginal_operational_bounds,
    report_prediction_stats,
)

# Generate or load calibration data
sim = BinaryClassifierSimulator(p_class1=0.5, seed=42)
labels, probs = sim.generate(n_samples=200)

# Split by class
class_data = split_by_class(labels, probs)

# Get PAC coverage guarantees
cal_result, pred_stats = mondrian_conformal_calibrate(
    class_data=class_data,
    alpha_target=0.10,  # Target 90% coverage
    delta=0.05,         # 95% PAC confidence
)
```

### Step 2: Estimate Operational Rates (LOO-CV)

```python
# Per-class operational bounds via Leave-One-Out CV
per_class_bounds = compute_mondrian_operational_bounds(
    calibration_result=cal_result,
    labels=labels,
    probs=probs,
    ci_width=0.95,  # 95% confidence intervals
    n_jobs=-1,      # Use all CPU cores for speed
)

# Marginal operational bounds (deployment view)
marginal_bounds = compute_marginal_operational_bounds(
    labels=labels,
    probs=probs,
    alpha_target=0.10,
    delta_coverage=0.05,
    ci_width=0.95,
    n_jobs=-1,
)
```

### Step 3: Generate Deployment Report

```python
# Comprehensive report with all guarantees
report_prediction_stats(
    prediction_stats=pred_stats,
    calibration_result=cal_result,
    operational_bounds_per_class=per_class_bounds,
    marginal_operational_bounds=marginal_bounds,
    verbose=True,
)
```

**Report includes:**
- ✅ PAC coverage guarantees (SSBC)
- ✅ Operational rate bounds (LOO-CV)
- ✅ Singleton/doublet/abstention rates
- ✅ Conditional error rates P(error | singleton)
- ✅ Per-class and marginal statistics

## Operational Bounds API

### Marginal Operational Bounds

Estimates for overall deployment (ignores true labels):

```python
from ssbc import compute_marginal_operational_bounds

marginal_bounds = compute_marginal_operational_bounds(
    labels=labels,                    # True labels
    probs=probs,                      # Probability matrix (n, 2)
    alpha_target=0.10,                # Target miscoverage
    delta_coverage=0.05,              # PAC risk for coverage
    rate_types=None,                  # All rates (or specify list)
    ci_width=0.95,                    # 95% CI width
    n_jobs=1,                         # Parallel jobs (-1 = all cores)
)

# Access bounds
singleton_bounds = marginal_bounds.rate_bounds["singleton"]
print(f"Singleton rate: [{singleton_bounds.lower_bound:.3f}, "
      f"{singleton_bounds.upper_bound:.3f}]")
print(f"Count: {singleton_bounds.n_successes}/{singleton_bounds.n_evaluations}")
```

### Per-Class Operational Bounds

Estimates conditioned on true label (performance by class):

```python
from ssbc import compute_mondrian_operational_bounds

per_class_bounds = compute_mondrian_operational_bounds(
    calibration_result=cal_result,   # From mondrian_conformal_calibrate()
    labels=labels,                    # True labels
    probs=probs,                      # Probability matrix (n, 2)
    rate_types=None,                  # All rates (or specify list)
    ci_width=0.95,                    # 95% CI width
    n_jobs=1,                         # Parallel jobs (-1 = all cores)
)

# Access per-class bounds
for class_label in [0, 1]:
    class_bounds = per_class_bounds[class_label]
    singleton = class_bounds.rate_bounds["singleton"]
    print(f"Class {class_label} singleton rate: "
          f"[{singleton.lower_bound:.3f}, {singleton.upper_bound:.3f}]")
```

### Performance Optimization

For large datasets (n > 250), use multiprocessing:

```python
# Single-threaded (default)
bounds = compute_mondrian_operational_bounds(cal_result, labels, probs, n_jobs=1)

# Use all CPU cores (recommended for n > 250)
bounds = compute_mondrian_operational_bounds(cal_result, labels, probs, n_jobs=-1)

# Use specific number of cores
bounds = compute_mondrian_operational_bounds(cal_result, labels, probs, n_jobs=4)
```

**Speedup examples:**
- n=250: ~3.6x speedup with `n_jobs=-1`
- n=500: ~7.2x speedup with `n_jobs=-1`
- n=1000: ~10-15x speedup with `n_jobs=-1`

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

### Supported Rate Types

- **`"singleton"`**: |C(x)| = 1 (single predicted label)
- **`"doublet"`**: |C(x)| = 2 (two predicted labels)
- **`"abstention"`**: |C(x)| = 0 (no prediction)
- **`"error_in_singleton"`**: |C(x)| = 1 and y ∉ C(x) (singleton error)
- **`"correct_in_singleton"`**: |C(x)| = 1 and y ∈ C(x) (singleton correct)

## Understanding the Results

### OperationalRateBounds Dataclass

Each rate bound contains:

```python
bounds = marginal_bounds.rate_bounds["singleton"]

print(bounds.rate_name)       # "singleton"
print(bounds.lower_bound)     # Lower CP bound
print(bounds.upper_bound)     # Upper CP bound
print(bounds.ci_width)        # CI width (e.g., 0.95)
print(bounds.n_evaluations)   # Total LOO evaluations (n)
print(bounds.n_successes)     # Count (K)
```

### OperationalRateBoundsResult Dataclass

```python
result = marginal_bounds

print(result.rate_bounds)      # Dict of OperationalRateBounds
print(result.ci_width)         # CI width for all bounds
print(result.thresholds)       # Reference thresholds (display only)
print(result.n_calibration)    # Calibration set size
```

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

### 3. Complete SLA/Deployment Workflow ⭐
```bash
python examples/sla_example.py
```
**Full deployment pipeline**: PAC coverage + LOO-CV operational bounds + comprehensive reporting.

## Key Concepts

### PAC Coverage (from SSBC)

**Guarantee:** With probability ≥ 1-δ over calibration sets, the conformal predictor
achieves coverage ≥ 1-α_target on future data.

**Properties:**
- Valid for ANY sample size n
- Distribution-free
- Frequentist (no priors)

### Operational Bounds (from LOO-CV)

**Estimates:** Confidence intervals on deployment rates (singleton, doublet, abstention, error).

**Procedure:**
1. For each calibration point i, train on all OTHER points
2. Apply predictor to point i (unbiased evaluation)
3. Aggregate n evaluations
4. Apply Clopper-Pearson CIs

**Properties:**
- Unbiased estimates (LOO ensures no data leakage)
- Exact binomial CIs (Clopper-Pearson)
- Standard frequentist interpretation

### Marginal vs Per-Class

**Marginal bounds** (ignore true labels):
- "What will a user see?"
- Deployment view
- Overall automation rate

**Per-class bounds** (conditioned on true label):
- "How does performance differ by ground truth?"
- Class-specific rates
- Identifies minority class challenges

## References

### Key Statistical Properties

- **Distribution-Free**: No P(X,Y) assumptions
- **Model-Agnostic**: Works with any classifier
- **Frequentist**: Valid frequentist guarantees
- **Non-Bayesian**: No priors required
- **Finite-Sample**: Exact for small n (not asymptotic)
- **Exchangeability Only**: Minimal assumption

### Further Reading

- See [theory.md](theory.md) for detailed theoretical background
- See [installation.md](installation.md) for setup instructions
- See `examples/` directory for complete working examples
