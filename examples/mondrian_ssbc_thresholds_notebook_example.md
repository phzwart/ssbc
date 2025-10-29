# Using Existing SSBC Code for Mondrian Thresholds and Error Bounds

This guide shows how to use the existing SSBC infrastructure to compute Mondrian thresholds and generate rigorous error bounds reports **without modifying any code**.

## Cell 1: Import and Setup

```python
import numpy as np
from ssbc import (
    BinaryClassifierSimulator,
    generate_rigorous_pac_report,
    split_by_class,
    mondrian_conformal_calibrate,
)
```

## Cell 2: Load Your Data

Replace this with your actual data:

```python
# If you have your own data:
# labels = your_labels_array  # shape (n,)
# probs = your_probabilities_array  # shape (n, 2) with columns [P(class=0), P(class=1)]

# OR use simulated data for testing:
sim = BinaryClassifierSimulator(
    p_class1=0.50, 
    beta_params_class0=(2, 7), 
    beta_params_class1=(7, 2), 
    seed=42
)
labels, probs = sim.generate(n_samples=1000)
```

## Cell 3: Split by Class (Mondrian Approach)

```python
class_data = split_by_class(labels, probs)

for label in [0, 1]:
    print(f"Class {label}: n = {class_data[label]['n']}")
```

## Cell 4: Compute SSBC-Corrected Thresholds

```python
alpha_target = 0.10  # Target 10% miscoverage
delta = 0.10        # 90% PAC confidence

# Calibrate with SSBC to get corrected thresholds
cal_result, pred_stats = mondrian_conformal_calibrate(
    class_data=class_data,
    alpha_target={0: alpha_target, 1: alpha_target},
    delta={0: delta, 1: delta},
    mode="beta"
)

# Extract the thresholds (this replaces the mondrian_ssbc_thresholds function)
print("\nSSBC-Corrected Thresholds:")
for label in [0, 1]:
    result = cal_result[label]
    print(f"  Class {label}:")
    print(f"    Threshold: {result['threshold']:.4f}")
    print(f"    α_corrected: {result['alpha_corrected']:.4f}")
    print(f"    PAC guarantee satisfied: {result['ssbc_result'].satisfied_mass:.4f}")
```

## Cell 5: Generate Rigorous Error Bounds Report

```python
# Generate complete PAC report with error bounds
report = generate_rigorous_pac_report(
    labels=labels,
    probs=probs,
    alpha_target=alpha_target,
    delta=delta,
    use_loo_correction=True,  # Uses LOO-CV + Clopper-Pearson
    verbose=True  # Set to False if you don't want the full printout
)
```

## Cell 6: Extract Error Bounds

```python
# Marginal error bounds (overall)
pac_marg = report["pac_bounds_marginal"]
se_lower, se_upper = pac_marg["singleton_error_rate_bounds"]

print("Marginal Singleton Error Rate Bounds:")
print(f"  [{se_lower:.3f}, {se_upper:.3f}]")
print(f"  Expected: {pac_marg['expected_singleton_error_rate']:.3f}")

# Per-class error bounds
for class_label in [0, 1]:
    pac_class = report[f"pac_bounds_class_{class_label}"]
    se_lower_c, se_upper_c = pac_class["singleton_error_rate_bounds"]
    
    print(f"\nClass {class_label} Singleton Error Rate Bounds:")
    print(f"  [{se_lower_c:.3f}, {se_upper_c:.3f}]")
    print(f"  Expected: {pac_class['expected_singleton_error_rate']:.3f}")
```

## Cell 7: Extract Operational Bounds (Deployment Planning)

```python
pac_marg = report["pac_bounds_marginal"]

s_lower, s_upper = pac_marg["singleton_rate_bounds"]
a_lower, a_upper = pac_marg["abstention_rate_bounds"]
d_lower, d_upper = pac_marg["doublet_rate_bounds"]

print("Deployment Expectations:")
print(f"  Automation (singletons): [{s_lower:.1%}, {s_upper:.1%}]")
print(f"  Escalation needed: [{a_lower + d_lower:.1%}, {a_upper + d_upper:.1%}]")
```

## Cell 8: Access All Report Data

```python
# The report dictionary contains comprehensive information:

# SSBC results for each class
ssbc_class_0 = report["ssbc_class_0"]
ssbc_class_1 = report["ssbc_class_1"]

# PAC bounds (marginal and per-class)
pac_bounds_marginal = report["pac_bounds_marginal"]
pac_bounds_class_0 = report["pac_bounds_class_0"]
pac_bounds_class_1 = report["pac_bounds_class_1"]

# Calibration results (thresholds, scores, etc.)
calibration_result = report["calibration_result"]

# Prediction statistics from calibration data
prediction_stats = report["prediction_stats"]

# Parameters used
parameters = report["parameters"]
```

## Summary

This workflow uses the existing SSBC infrastructure to:
1. **Compute SSBC-corrected thresholds** via `mondrian_conformal_calibrate()`
2. **Generate rigorous PAC reports** with error bounds via `generate_rigorous_pac_report()`
3. **Extract error rates and bounds** from the report dictionary
4. **Account for calibration uncertainty** via LOO-CV + Clopper-Pearson

**No code modifications required** - everything uses the existing SSBC codebase!
