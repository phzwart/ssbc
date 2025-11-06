# Code Audit Report: Operational Bounds Mathematical Correctness

## Summary

This audit verifies that the operational bounds implementation matches the mathematical framework defined in `operational_bounds_mathematical_framework.md`.

## Issues Found

### 1. CRITICAL: Approach B Ratio Estimation (Lines 1063-1099 in `operational_bounds_simple.py`)

**Location**: `compute_pac_operational_bounds_perclass_loo_corrected`

**Problem**: The "Approach B" code constructs prediction bounds by multiplying intervals:
```python
approach_b_singleton_bounds = [
    singleton_lower * min_class_rate / class_rate_cal,
    singleton_upper * max_class_rate / class_rate_cal,
]
```

This violates the principle that bounds should be computed from a **single well-defined Bernoulli event**, not by combining two separate intervals.

**Fix**: Remove Approach B code entirely. Approach A is correct (joint rates with fixed denominator). If conditional rates are needed, they should be computed directly using the conditional subpopulation, not by ratio estimation.

### 2. MINOR: Marginal Rates Still Computed (Lines 243-249, 314-316)

**Location**: `compute_pac_operational_bounds_marginal` and `compute_pac_operational_bounds_marginal_loo_corrected`

**Problem**: Global marginal rates (P(singleton), P(doublet), P(abstention)) are computed and returned, even though the framework states these should not be reported because they mix classes with different cost structures.

**Status**: These are currently returned for backward compatibility. Consider deprecating or removing them in a future version. The class-specific rates are the primary metrics.

### 3. DOCUMENTATION: Missing Mathematical Comments

**Problem**: The code lacks explicit comments mapping each metric to its Bernoulli event definition.

**Fix**: Add comments above each `prediction_bounds` call explaining:
- The Bernoulli event being modeled
- What `k_cal`, `n_cal`, and `n_test` represent
- Whether it's a joint (full sample) or conditional rate

### 4. DOCUMENTATION: Docstring Clarity

**Problem**: The `prediction_bounds` docstring doesn't explicitly state that `k_cal` and `n_cal` must be for a **single well-defined Bernoulli event**.

**Fix**: Update docstring to emphasize this requirement.

## Verification Results

### ✅ CORRECT: Joint Per-Class Rates (Full Sample)

**Example**: `singleton_rate_class0_bounds`

**Code** (lines 252-254):
```python
singleton_class0_lower, singleton_class0_upper = prediction_bounds(
    n_singletons_class0_total, n, test_size, adjusted_ci_level, prediction_method
)
```

**Verification**:
- ✅ `k_cal = n_singletons_class0_total` = count of `Z_i^{sing,0} = 1{Y_i=0, S_i=singleton}`
- ✅ `n_cal = n` = total calibration size (fixed denominator)
- ✅ `n_test = test_size` = planned test size (fixed)
- ✅ Matches framework: Joint rate with fixed denominator

### ✅ CORRECT: Conditional Error Rates

**Example**: `singleton_error_rate_cond_class0_bounds`

**Code** (lines 628-635):
```python
error_cond_class0_lower, error_cond_class0_upper, error_cond_class0_report = compute_robust_prediction_bounds(
    error_cond_class0_loo_preds[singleton_class0_mask],
    expected_n_singletons_class0_test,
    1 - adjusted_ci_level,
    method=prediction_method,
    inflation_factor=loo_inflation_factor,
    verbose=False,
)
```

**Verification**:
- ✅ `loo_predictions` = array of `W_i^{err|0}` indicators (restricted to Y=0, S=singleton)
- ✅ `n_cal` = size of conditional subpopulation (implicit in array length)
- ✅ `n_test = expected_n_singletons_class0_test` = estimated future conditional test size
- ✅ Matches framework: Option B (predictive fraction for test run)
- ⚠️ **Limitation**: Denominator is random, making bounds conservative (documented in stability note)

### ❌ INCORRECT: Approach B Ratio Estimation

**Location**: Lines 1063-1099

**Problem**: Constructs bounds by multiplying two intervals:
```python
approach_b_singleton_bounds = [
    singleton_lower * min_class_rate / class_rate_cal,
    singleton_upper * max_class_rate / class_rate_cal,
]
```

This is mathematically incorrect because:
1. It combines uncertainty from two separate random quantities
2. The resulting interval is not a valid prediction interval for any well-defined Bernoulli event
3. It violates the principle that bounds should come from a single Bernoulli formulation

**Fix**: Remove Approach B entirely. Approach A already provides the correct joint rates.

## Recommended Fixes

### Priority 1: Remove Approach B Code

Remove lines 1063-1099 and all references to `*_bounds_class_samples` in the return dictionary.

### Priority 2: Add Mathematical Comments

Add comments above each `prediction_bounds` call mapping to the Bernoulli event definition.

### Priority 3: Update Docstrings

Clarify that `prediction_bounds` requires a single well-defined Bernoulli event.

### Priority 4: Consider Deprecating Marginal Rates

Document that global marginal rates (P(singleton), etc.) are deprecated in favor of class-specific rates.
