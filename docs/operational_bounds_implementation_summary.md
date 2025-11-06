# Operational Bounds Implementation Summary

## Completed Changes

### 1. Removed Approach B Ratio Estimation (CRITICAL FIX)

**Location**: `compute_pac_operational_bounds_perclass_loo_corrected` (lines 1063-1099)

**What was removed**: Code that constructed prediction bounds by multiplying two separate intervals:
```python
approach_b_singleton_bounds = [
    singleton_lower * min_class_rate / class_rate_cal,
    singleton_upper * max_class_rate / class_rate_cal,
]
```

**Why removed**: This violated the principle that bounds must come from a **single well-defined Bernoulli event**, not by combining uncertainty from multiple random quantities.

**Result**: The function now only returns Approach A bounds (joint rates with fixed denominators), which are mathematically correct.

### 2. Added Mathematical Comments

**Locations**: Throughout `operational_bounds_simple.py`

**What was added**: Comments mapping each metric to its Bernoulli event definition:
- Joint per-class rates: `Z_i^{sing,0} = 1{Y_i=0, S_i=singleton}`
- Conditional error rates: `W_i^{err|0} = 1{E_i=1}` given `Y_i=0, S_i=singleton`
- Explicit documentation of `k_cal`, `n_cal`, and `n_test` meanings

### 3. Updated Docstrings

**Location**: `statistical.py` - `prediction_bounds()` function

**What was updated**:
- Added explicit requirement that `k_cal` and `n_cal` must be for a **single well-defined Bernoulli event**
- Clarified that `n_test` is the number of future trials for the same event
- Added warning against constructing bounds by dividing two separate intervals

### 4. Documented Conditional Rate Limitations

**Locations**: Comments above conditional error rate computations

**What was added**: Notes explaining that conditional rates have random denominators, making bounds conservative. This is also documented in the stability note in the report.

## Verification Status

### ✅ CORRECT: Joint Per-Class Rates

All joint per-class rates (singleton, doublet, abstention, error for class 0 and class 1) are computed correctly:
- Use fixed denominators (`n_cal = total calibration size`)
- Use fixed test sizes (`n_test = planned test size`)
- Match the mathematical framework exactly

### ✅ CORRECT: Conditional Error Rates

Conditional error rates are computed using Option B (predictive fraction for test run):
- Use conditional subpopulation for calibration
- Estimate future conditional test size
- Bounds are conservative due to random denominator (documented)

### ⚠️ NOTE: Marginal Rates Still Computed

Global marginal rates (P(singleton), P(doublet), P(abstention)) are still computed and returned for backward compatibility. These mix classes with different cost structures and should be considered deprecated. The class-specific rates are the primary metrics.

## Files Modified

1. `src/ssbc/metrics/operational_bounds_simple.py`:
   - Removed Approach B code (~50 lines)
   - Added mathematical comments (~30 lines)
   - Simplified return dictionary

2. `src/ssbc/bounds/statistical.py`:
   - Enhanced `prediction_bounds()` docstring with Bernoulli event requirements

3. Documentation:
   - `docs/operational_bounds_mathematical_framework.md` (new)
   - `docs/operational_bounds_code_audit.md` (new)
   - `docs/operational_bounds_implementation_summary.md` (this file)

## Next Steps (Optional)

1. **Consider deprecating marginal rates**: Document that `singleton_rate_bounds`, `doublet_rate_bounds`, `abstention_rate_bounds` are deprecated in favor of class-specific versions.

2. **Add unit tests**: Verify that all bounds match the mathematical framework in simulation tests.

3. **Update validation**: Ensure validation routines check that bounds are computed from single Bernoulli events, not ratios.

## Mathematical Correctness

All bounds are now computed from **single well-defined Bernoulli events**:
- Joint rates: Fixed denominators, mathematically exact
- Conditional rates: Conservative due to random denominator, properly documented

The implementation is now consistent with the mathematical framework defined in `operational_bounds_mathematical_framework.md`.
