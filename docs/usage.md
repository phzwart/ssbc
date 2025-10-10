# Usage

## Overview

SSBC (Small-Sample Beta Correction) provides tools for:
- **PAC coverage guarantees** for conformal prediction with finite samples
- **Mondrian conformal prediction** for class-conditional guarantees
- **Service Level Agreement (SLA) bounds** for operational rates
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

# Prepare data
labels = np.array([0, 1, 0, 1, 0, 1])
probs = np.array([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1],
                   [0.2, 0.8], [0.85, 0.15], [0.1, 0.9]])

# Split by class
class_data = split_by_class(labels, probs)

# Calibrate with SSBC
cal_result, pred_stats = mondrian_conformal_calibrate(
    class_data=class_data,
    alpha_target=0.10,
    delta=0.10,
    mode="beta"
)

# View results
for label in [0, 1]:
    print(f"Class {label}: threshold = {cal_result[label]['threshold']:.4f}")
```

## Service Level Agreement (SLA) Bounds

The SLA module provides contract-ready guarantees for conformal prediction deployment, including coverage and operational rate bounds.

### Complete SLA Pipeline

```python
import numpy as np
from ssbc import compute_conformal_sla

# Generate or load calibration data
n_cal = 200
cal_features = np.random.randn(n_cal, 10)
cal_labels = np.random.randint(0, 5, n_cal)

# Define nonconformity score function
def score_function(x, y):
    """Score based on your trained model.

    In practice, use: 1 - P(y|x) from your classifier.
    Higher score = less conforming.
    """
    # Example: distance-based score
    class_centers = get_class_centers()  # Your model
    return np.linalg.norm(x - class_centers[y])

# Compute SLA
sla_result = compute_conformal_sla(
    cal_features=cal_features,
    cal_labels=cal_labels,
    score_function=score_function,
    alpha_target=0.10,      # Target 90% coverage
    delta_1=0.05,           # Coverage confidence: 95%
    delta_2=0.05,           # Rate bounds confidence: 95%
    rate_types=["singleton", "doublet", "abstention"],
    n_folds=5,
    random_seed=42
)

# View guarantees
print(f"Coverage: ≥ {sla_result.coverage_guarantee:.1%} "
      f"w.p. ≥ {sla_result.coverage_confidence:.1%}")
print(f"Threshold: {sla_result.threshold:.4f}")

# Operational rate bounds
for rate_name, bounds in sla_result.rate_bounds.items():
    print(f"{rate_name}: [{bounds.lower_bound:.3f}, {bounds.upper_bound:.3f}]")
```

### Mondrian SLA (Class-Conditional)

```python
from ssbc import compute_conformal_sla_mondrian

# Compute class-specific SLAs
mondrian_results = compute_conformal_sla_mondrian(
    cal_features=cal_features,
    cal_labels=cal_labels,
    score_function=score_function,
    alpha_target=0.10,
    delta_1=0.05,
    delta_2=0.05,
    n_folds=5,
    random_seed=42
)

# Each class gets its own threshold and bounds
for class_label, result in mondrian_results.items():
    print(f"Class {class_label}:")
    print(f"  Threshold: {result.threshold:.4f}")
    print(f"  Coverage: {result.coverage_guarantee:.1%}")
```

### Understanding SLA Results

The `ConformalSLAResult` provides:

- **Coverage Guarantee**: P(Y ∈ C(X)) ≥ 1 - α with probability ≥ 1 - δ₁
- **Operational Rate Bounds**:
  - `singleton`: Single predicted label (automated decision)
  - `doublet`: Two predicted labels (needs review)
  - `abstention`: No predicted labels (high uncertainty)
  - `error_in_singleton`: Singleton prediction that's incorrect
- **Joint Confidence**: 1 - (δ₁ + δ₂) - all bounds hold simultaneously

## Statistical Utilities

### Clopper-Pearson Confidence Intervals

```python
from ssbc import clopper_pearson_lower, clopper_pearson_upper, cp_interval

# One-sided bounds (for PAC guarantees)
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

## Advanced Topics

### Custom Rate Types

The SLA module supports custom operational rates:
- `"singleton"`: |C(x)| = 1
- `"doublet"`: |C(x)| = 2
- `"abstention"`: |C(x)| = 0
- `"error_in_singleton"`: |C(x)| = 1 and y ∉ C(x)

You can compute bounds for any combination:

```python
sla_result = compute_conformal_sla(
    cal_features, cal_labels, score_function,
    alpha_target=0.10, delta_1=0.05, delta_2=0.05,
    rate_types=["singleton", "error_in_singleton"],
    n_folds=5
)
```

### Cross-Fit Analysis

For detailed fold-by-fold analysis:

```python
from ssbc import cross_fit_cp_bounds

# Compute cross-fit bounds separately
cf_results = cross_fit_cp_bounds(
    cal_features, cal_labels, score_function,
    alpha_adj=0.08,  # From SSBC correction
    rate_types=["singleton"],
    n_folds=5,
    delta_2=0.05,
    random_seed=42
)

# Access fold-level details
for fold_data in cf_results["singleton"]["fold_results"]:
    print(f"Fold {fold_data['fold']}: "
          f"L={fold_data['L_fr']:.3f}, U={fold_data['U_fr']:.3f}")
```

### Transfer Cushion

The transfer cushion accounts for differences between cross-fit and single-rule thresholds:

```python
from ssbc import compute_transfer_cushion, transfer_bounds_to_single_rule

# Compute cushions
cushions = {}
for rate_type in ["singleton", "doublet"]:
    cushions[rate_type] = compute_transfer_cushion(
        cal_features, cal_labels, score_function,
        cf_results[rate_type], rate_type
    )

# Transfer bounds to single rule
transferred = transfer_bounds_to_single_rule(cf_results, cushions)

print(f"Singleton cushion: {cushions['singleton']:.4f}")
```

## Examples

Complete examples are available in the `examples/` directory:

- `examples/basic_usage.py` - Basic SSBC correction
- `examples/mondrian_conformal_example.py` - Mondrian conformal prediction
- `examples/sla_example.py` - Complete SLA pipeline with synthetic data
- `examples/hyperparameter_sweep_example.py` - Hyperparameter optimization

Run an example:

```bash
python examples/sla_example.py
```

## API Reference

For detailed API documentation, see the module docstrings:

```python
from ssbc import compute_conformal_sla
help(compute_conformal_sla)
```

## References

- **SSBC**: Finite-sample PAC guarantees via Beta distribution bounds
- **Conformal Prediction**: Distribution-free prediction intervals
- **Mondrian CP**: Class-conditional conformal prediction
- **SLA Bounds**: Cross-fit operational rate bounds with transfer to single rule
