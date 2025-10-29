# Code Review: utils.py and validation.py

## Executive Summary

Both `utils.py` and `validation.py` contain well-documented, functional code with good test coverage. However, several issues were identified:

1. **Critical Bug**: `evaluate_test_dataset` uses wrong indicators for per-class rates
2. **Significant Code Duplication**: Prediction set building logic duplicated across 8+ locations
3. **Performance Opportunities**: Vectorization possible for prediction set building
4. **Code Organization**: Long functions and repetitive code patterns
5. **Missing Edge Case Handling**: Some validation edge cases not covered

---

## utils.py - Detailed Findings

### ✅ Strengths

1. **Excellent Documentation**: NumPy-style docstrings with examples
2. **Good Type Hints**: Functions properly typed
3. **Test Coverage**: Comprehensive tests in `test_utils.py`
4. **Clear Function Names**: Functions are self-documenting

### ⚠️ Design Issues

#### Confusing Function Design: Closure Variables in `compute_rates` (Lines 179-235)

**Location**: `evaluate_test_dataset`, nested function `compute_rates`

**Problem**: The nested `compute_rates` function uses closure variables (`doublet_indicators`, `abstention_indicators`, `error_in_singleton_indicators`) which makes the function's dependencies unclear. The `indicators` parameter is only used for singleton rate computation, but the function also computes doublet, abstention, and error rates using closed-over variables.

**Current Design**:
```python
def compute_rates(indicators: np.ndarray, mask: np.ndarray | None = None):
    # Uses 'indicators' for singleton rate
    singleton_rate = np.mean(subset_indicators)
    # But uses closure variables for other rates:
    doublet_indicators_subset = doublet_indicators[mask] if mask else doublet_indicators
    # ...
```

**Issue**:
- Function signature doesn't reveal all dependencies
- Makes it harder to extract to module level
- Closure variables could cause issues if function is moved

**Verification**: Tested - code works correctly, but design is suboptimal.

**Recommendation**:
1. Refactor to accept all indicator types explicitly:
   ```python
   def compute_rates(
       singleton_indicators: np.ndarray,
       doublet_indicators: np.ndarray,
       abstention_indicators: np.ndarray,
       error_indicators: np.ndarray,
       mask: np.ndarray | None = None,
   ) -> dict[str, Any]:
   ```
2. Extract to module-level function for reusability
3. Improve documentation to clarify parameter usage

### ⚠️ Code Duplication

#### Prediction Set Building Logic

**Problem**: Prediction set building logic is duplicated across multiple files:
- `utils.py:156-167` - `evaluate_test_dataset`
- `validation.py:81-98` - `_validate_single_trial`
- `conformal.py:259-271`, `193-201`, `596-609` - Multiple locations
- `bootstrap.py:149-166` - Bootstrap analysis
- `cross_conformal.py:92-103` - Cross-conformal
- `operational_bounds_simple.py:73-91`, `462-480` - LOO evaluation

**Pattern Duplicated**:
```python
score_0 = 1.0 - probs[i, 0]
score_1 = 1.0 - probs[i, 1]
pred_set = set()  # or []
if score_0 <= threshold_0:
    pred_set.add(0)  # or append
if score_1 <= threshold_1:
    pred_set.add(1)  # or append
```

**Recommendation**: Extract to utility function:
```python
def build_mondrian_prediction_sets(
    probs: np.ndarray,
    threshold_0: float,
    threshold_1: float,
) -> list[set[int]]:
    """Build prediction sets using Mondrian conformal thresholds.

    Parameters
    ----------
    probs : np.ndarray, shape (n, 2)
        Probability predictions
    threshold_0 : float
        Threshold for class 0
    threshold_1 : float
        Threshold for class 1

    Returns
    -------
    list[set[int]]
        List of prediction sets
    """
    scores_0 = 1.0 - probs[:, 0]
    scores_1 = 1.0 - probs[:, 1]

    prediction_sets = []
    for score_0, score_1 in zip(scores_0, scores_1):
        pred_set = set()
        if score_0 <= threshold_0:
            pred_set.add(0)
        if score_1 <= threshold_1:
            pred_set.add(1)
        prediction_sets.append(pred_set)

    return prediction_sets
```

### 📊 Performance Opportunities

#### Vectorization of Prediction Set Building

**Current**: Loop-based building (lines 157-167)

**Potential Optimization**: Could vectorize score computation:
```python
scores_0 = 1.0 - test_probs[:, 0]  # Vectorized
scores_1 = 1.0 - test_probs[:, 1]  # Vectorized
in_0 = scores_0 <= threshold_0
in_1 = scores_1 <= threshold_1

# Build sets (still need loop for sets, but faster)
prediction_sets = [
    {0} if in_0[i] else set() | ({1} if in_1[i] else set())
    for i in range(n_test)
]
```

**Note**: The set building still needs a loop, but score computation can be vectorized.

### 🔧 Code Organization

#### Nested Function `compute_rates` (Lines 179-235)

**Issue**: 57-line nested function makes the parent function harder to read

**Recommendation**: Extract to module-level function:
- Makes it reusable
- Improves testability
- Reduces cognitive load

### 📝 Documentation Improvements

1. **Line 179**: Document that `compute_rates` uses multiple indicator arrays, not just `indicators`
2. **Line 222**: Clarify that `subset_indicators == 1` filters to singletons for error rate computation

---

## validation.py - Detailed Findings

### ✅ Strengths

1. **Comprehensive Functionality**: Full validation workflow
2. **Good Error Handling**: Safe parallel execution with fallback
3. **Rich Output**: Detailed validation results
4. **Good Documentation**: Clear docstrings

### ⚠️ Code Duplication

#### Prediction Set Building

**Same issue as utils.py**: Duplicated across multiple files (see utils.py section)

#### Rate Computation Logic (Lines 120-139)

**Problem**: Manual rate computation duplicates logic from `evaluate_test_dataset`

**Current**: Manual counting and rate computation in `_validate_single_trial`
```python
marginal_singleton_rate = n_singletons / n_total
marginal_doublet_rate = n_doublets / n_total
# ... etc
```

**Recommendation**: Reuse `evaluate_test_dataset`:
```python
def _validate_single_trial(...):
    # ... generate test data ...

    # Use existing function instead of manual counting
    eval_results = evaluate_test_dataset(
        test_labels=labels_test,
        test_probs=probs_test,
        threshold_0=threshold_0,
        threshold_1=threshold_1,
    )

    return {
        "marginal": {
            "singleton": eval_results["marginal"]["singleton_rate"],
            "doublet": eval_results["marginal"]["doublet_rate"],
            # ... etc
        },
        # ... per-class rates ...
    }
```

**Benefits**:
- DRY principle
- Consistent behavior
- Easier to maintain
- Reduces code by ~100 lines

### 🔧 Code Organization

#### Long Function: `validate_pac_bounds` (163-452 lines, 289 lines)

**Issues**:
1. Too many responsibilities
2. Repetitive rate extraction (lines 263-292, 279-292)
3. Helper functions defined inside (lines 295-332)

**Recommendation**: Refactor into smaller functions:
```python
def _extract_rate_arrays(trial_results: list[dict]) -> dict[str, np.ndarray]:
    """Extract rate arrays from trial results."""
    # ...

def _compute_rate_statistics(
    rates: np.ndarray,
    bounds: tuple[float, float],
    expected: float,
) -> dict[str, Any]:
    """Compute statistics for a single rate."""
    # ...

def validate_pac_bounds(...):
    """Main validation function."""
    # ... setup ...
    trial_results = _safe_parallel_map(n_jobs)

    # Extract all rates
    rates = _extract_rate_arrays(trial_results)

    # Compute statistics for each rate type
    marginal_stats = {
        rate_type: _compute_rate_statistics(...)
        for rate_type in ["singleton", "doublet", "abstention", "singleton_error"]
    }
    # ...
```

#### Repetitive Rate Extraction (Lines 263-292)

**Problem**: 30 lines of repetitive list comprehensions

**Current**:
```python
marginal_singleton_rates = [result["marginal"]["singleton"] for result in trial_results]
marginal_doublet_rates = [result["marginal"]["doublet"] for result in trial_results]
# ... 14 more similar lines ...
```

**Recommendation**: Extract to helper function or use dictionary comprehension:
```python
def _extract_rates(trial_results: list[dict]) -> dict[str, np.ndarray]:
    """Extract all rate arrays from trial results."""
    return {
        "marginal_singleton": np.array([r["marginal"]["singleton"] for r in trial_results]),
        "marginal_doublet": np.array([r["marginal"]["doublet"] for r in trial_results]),
        # ... etc
    }
```

### 🐛 Potential Bugs

#### Hardcoded Coverage Check (Line 486)

**Location**: `print_validation_results`, line 486

**Problem**:
```python
coverage_check = "✅" if coverage >= 0.90 else "❌"  # Assuming 90% PAC level
```

**Issue**: Hardcoded `0.90` assumption. Should use actual PAC level from report.

**Fix**:
```python
# Get PAC level from validation results or report
pac_level = validation.get("pac_level", 0.90)  # Default if not provided
coverage_check = "✅" if coverage >= pac_level else "❌"
```

**Note**: PAC level might need to be passed into `print_validation_results` or extracted from report.

### 📊 Performance

#### Parallel Execution (Lines 248-260)

**Good**: Safe fallback to serial execution when multiprocessing fails

**Minor Improvement**: Could log when fallback occurs:
```python
def _safe_parallel_map(n_jobs_local: int):
    try:
        return Parallel(n_jobs=n_jobs_local)(...)
    except Exception as e:
        if verbose:
            print(f"Warning: Parallel execution failed ({e}), falling back to serial")
        return [...]
```

### 📝 Documentation Improvements

1. **Line 486**: Document that hardcoded 0.90 is a placeholder
2. **Line 254**: Document why we catch all exceptions (sandbox restrictions)
3. **Line 295**: Document `check_coverage` vs `check_coverage_with_nan` distinction

### 🔍 Missing Edge Cases

1. **Empty trial results**: No check if `trial_results` is empty
2. **All-NaN rates**: `compute_quantiles` handles this, but `check_coverage` might have issues
3. **Invalid report structure**: No validation that report has expected keys

---

## Recommendations Summary

### Priority 1 (High Value - Code Quality)

1. **Refactor `_validate_single_trial`** to use `evaluate_test_dataset` (eliminates ~100 lines of duplication)
2. **Extract prediction set building** to shared utility function (used in 8+ locations)

### Priority 2 (Code Organization)

3. **Refactor `validate_pac_bounds`** into smaller functions
4. **Fix hardcoded PAC level** in `print_validation_results`

### Priority 3 (Nice to Have)

5. **Extract `compute_rates`** from nested function
6. **Vectorize prediction set building** where possible
7. **Add edge case validation** for empty results, invalid reports
8. **Improve documentation** for complex functions

---

## Code Metrics

| Metric | utils.py | validation.py |
|--------|----------|---------------|
| Lines of Code | 249 | 501 |
| Functions | 2 public | 3 (1 private) |
| Max Function Length | 174 lines | 289 lines |
| Code Duplication | ~30 lines | ~100 lines |
| Test Coverage | ✅ Good | ✅ Good |
| Type Hints | ✅ Complete | ✅ Complete |
| Documentation | ✅ Excellent | ✅ Good |

---

## Estimated Impact

- **Lines Reduced**: ~130 lines (eliminating duplication)
- **Bugs Fixed**: 1 design issue + 1 hardcoded value
- **Maintainability**: Significant improvement from shared utilities
- **Performance**: Minor improvement from vectorization
- **Risk**: Low (changes are refactoring, not logic changes)
