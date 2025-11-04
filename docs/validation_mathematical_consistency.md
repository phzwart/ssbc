# Mathematical Consistency Validation Framework

## Overview

The validation framework for `validate_pac_bounds` and `validate_prediction_interval_calibration` has been enhanced to enforce strict mathematical consistency between the generative model, calibration statistics, and predictive validation.

## Implementation

### New Module: `validation_math.py`

This module provides:

1. **Bernoulli Event Definitions**: Dictionary mapping each metric to its exact Bernoulli event definition
2. **Calibration Count Extraction**: Functions to extract `k_cal`, `n_cal`, and `n_test` from reports
3. **Denominator Alignment Validation**: Checks that denominators match event definitions
4. **Probability Consistency Validation**: Verifies that joint rates sum to class prevalences
5. **Beta-Binomial Predictive Validation**: Validates that test rates follow the Beta-Binomial predictive distribution

### Enhanced Validation Functions

Both `validate_pac_bounds` and `validate_prediction_interval_calibration` now:

1. **Document Event Definitions**: Each validated metric includes its Bernoulli event definition
2. **Validate Denominator Alignment**: Check that `n_cal` and `n_test` match the event type
3. **Check Probability Consistency**: Verify `q_y^sing + q_y^dbl + q_y^abs = p_y ± ε` for each class
4. **Validate Beta-Binomial Predictions**: Simulate and compare empirical vs. theoretical quantiles
5. **Report Pass/Fail Indicators**: Clear ✅/❌ outcomes for each check

### Enhanced Reporting

The `print_validation_results` function now displays:

- **Event Definition**: Exact Bernoulli event for each metric
- **Calibration Counts**: `k_cal`, `n_cal`, `n_test` values
- **Denominator Alignment**: Validation status and issues
- **Coverage**: Empirical coverage vs. nominal confidence
- **Beta-Binomial Validation**: Comparison of empirical and theoretical quantiles
- **Probability Consistency**: Per-class checks that joint rates sum correctly

### Validation Rules Enforced

#### 1. Bernoulli Event Definition
- Each metric must correspond to exactly one Bernoulli random variable
- No ratio of two random quantities allowed
- Event definition explicitly documented

#### 2. Denominator Alignment
- Joint (full-sample) metrics: `n_cal = N_cal` (total calibration size)
- Conditional metrics: `n_cal = #{i: condition holds}` (conditional subpopulation)
- Test denominator `n_test` must be fixed and equal to future draws

#### 3. Predictive Law Validation
- Validates coverage of `K_test ~ BetaBinomial(n_test, k_cal+1, n_cal-k_cal+1)`
- Records empirical quantiles and compares to theoretical
- Requires coverage ≥ nominal confidence

#### 4. Internal Probability Consistency
- For each class `y`: `q_y^sing + q_y^dbl + q_y^abs = p_y ± ε` where `ε < 10^-3`
- Flags violations as internal accounting errors

#### 5. Reporting and Diagnostics
- Event definition displayed
- `k_cal`, `n_cal`, `n_test` values shown
- Empirical coverage of each interval type
- Clear ✅/❌ outcome for mathematical consistency

## Usage

The validation framework is automatically integrated into `validate_pac_bounds`. When you call:

```python
validation = validate_pac_bounds(report, simulator, test_size=1000, n_trials=1000)
print_validation_results(validation)
```

The output will include:
- Mathematical consistency checks for each metric
- Probability consistency validation
- Clear pass/fail indicators

## Limitations

1. **Calibration Count Estimation**: The framework estimates `k_cal` from expected rates because the report doesn't store raw calibration counts. For exact validation, calibration counts should be stored in the report.

2. **Conditional Rates**: For conditional rates with random denominators, full validation is limited because we cannot determine the exact conditional subpopulation size from the report alone.

3. **LOO Correlation**: The Beta-Binomial validation may show differences for LOO-corrected bounds due to correlation structure, which is expected and documented.

## Future Enhancements

1. Store actual calibration counts in PAC reports for exact validation
2. Add Monte Carlo validation for conditional rates with random denominators
3. Enhance Beta-Binomial validation to account for LOO correlation structure
