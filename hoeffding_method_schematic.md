# Schematic: Hoeffding Method for Operational Rate Bounds

## Overview
Compute PAC prediction bounds using the Hoeffding concentration inequality, accounting for LOO-CV correlation via an inflation factor.

## Inputs
- **LOO predictions**: `loo_predictions` (np.ndarray) - Binary LOO predictions from calibration set
- **Test size**: `n_test` (int) - Number of test samples  
- **Confidence level**: `alpha` (float) - Significance level (e.g., 0.05 for 95% confidence)
- **Inflation factor**: `inflation_factor` (float, optional) - Accounts for LOO correlation

## Step-by-Step Computation

### Step 1: Estimate Calibration Rate
```
p̂ = mean(loo_predictions)
```
This is the empirical rate on the calibration set (e.g., singleton/doublet/abstention rate).

### Step 2: Account for LOO Correlation
```
n_cal = len(loo_predictions)
n_effective_cal = n_cal / inflation_factor
n_effective_test = n_test
```
- `inflation_factor` accounts for correlation in LOO predictions
- If not provided, use conservative default: `inflation_factor = 2.0`
- Reduces effective calibration sample size to account for dependence

### Step 3: Compute Hoeffding Epsilons

**For Calibration Set:**
```
ε_cal = sqrt(log(4/α) / (2 × n_effective_cal))
```

**For Test Set:**
```
ε_test = sqrt(log(4/α) / (2 × n_effective_test))
```

Derived from Hoeffding's inequality: `P(|p̂ - p| > ε) ≤ 2 exp(-2nε²)`

### Step 4: Apply Union Bound
```
ε_total = ε_cal + ε_test
```
Combines uncertainty from both calibration and test sets.

### Step 5: Compute Final Bounds

**Lower Bound:**
```
L = max(0.0, p̂ - ε_total)
```

**Upper Bound:**
```
U = min(1.0, p̂ + ε_total)
```

Clips bounds to valid probability range [0, 1].

## Visual Summary

```
Input: loo_predictions, n_test, α, inflation_factor
  │
  ├─→ Compute p̂ = mean(loo_predictions)
  │
  ├─→ Compute effective sample sizes:
  │     n_effective_cal = n_cal / inflation_factor
  │     n_effective_test = n_test
  │
  ├─→ Compute Hoeffding epsilons:
  │     ε_cal = sqrt(log(4/α) / (2 × n_effective_cal))
  │     ε_test = sqrt(log(4/α) / (2 × n_effective_test))
  │
  ├─→ Combine uncertainties:
  │     ε_total = ε_cal + ε_test
  │
  └─→ Final bounds:
        L = max(0, p̂ - ε_total)
        U = min(1, p̂ + ε_total)

Output: Lower bound (L), Upper bound (U), Diagnostics
```

## Key Properties

- **Distribution-free**: No assumptions about data distribution
- **Ultra-conservative**: Provides worst-case bounds
- **PAC guarantee**: `P(p ∈ [L, U]) ≥ 1 - α` with probability ≥ 1 - α
- **LOO-corrected**: Accounts for correlation via inflation factor
- **Adaptive**: Uses estimated inflation factor (typically 1.0-6.0 range)

## Example

**Inputs:**
- `loo_predictions = [1, 0, 1, 1, 0, 1, ...]` (n_cal = 100)
- `n_test = 1000`
- `alpha = 0.05`
- `inflation_factor = 1.85` (estimated from data)

**Computation:**
```
p̂ = 0.75 (75% singleton rate)
n_effective_cal = 100 / 1.85 ≈ 54.05
n_effective_test = 1000

ε_cal = sqrt(log(4/0.05) / (2 × 54.05)) ≈ 0.158
ε_test = sqrt(log(4/0.05) / (2 × 1000)) ≈ 0.037

ε_total = 0.158 + 0.037 = 0.195

L = max(0, 0.75 - 0.195) = 0.555
U = min(1, 0.75 + 0.195) = 0.945
```

**Result:** [L=0.555, U=0.945] - We're 95% confident the true operational rate is between 55.5% and 94.5%.

## When to Use

- ✅ Need guaranteed PAC bounds (no distributional assumptions)
- ✅ Small sample sizes where asymptotic methods fail
- ✅ Want ultra-conservative (worst-case) bounds
- ✅ Dealing with non-IID data (LOO correlation)

**Trade-off:** Widest bounds, but mathematically rigorous guarantee.
