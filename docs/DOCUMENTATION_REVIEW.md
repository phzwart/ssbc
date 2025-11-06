# Documentation Review - Outdated Content Report

**Date**: 2025-01-30
**Status**: ✅ All issues fixed (2025-01-30)

This document identifies outdated, obsolete, or contradictory content in the SSBC documentation.

## Critical Issues

### 1. CONTRIBUTING.md - Outdated Development Setup

**Location**: `CONTRIBUTING.md` lines 52-77

**Issues**:
- ❌ References `virtualenvwrapper` and `mkvirtualenv` (outdated, less common)
- ❌ Uses `python setup.py develop` (deprecated, use `pip install -e .` or `uv pip install -e .`)
- ❌ References `make lint` and `make test` (no Makefile exists, project uses `justfile`)
- ❌ References `bump2version` (not used, versioning is manual in `pyproject.toml`)

**Current State**:
- Project uses `uv` for package management (see `justfile`)
- Project uses `just` commands: `just qa`, `just test`, `just coverage`
- Project uses `pyproject.toml` for build configuration (no `setup.py`)

**Recommendation**: Update to reflect actual workflow:
```bash
# Install for development
uv pip install -e .

# Run quality checks
just qa

# Run tests
just test

# Version is managed in pyproject.toml directly
```

### 2. docs/rigorous_report_math.md - Contradictory Statement

**Location**: `docs/rigorous_report_math.md` line 179

**Issue**:
- ❌ States: "No K-fold CV or transfer cushions are used"
- ✅ **BUT**: `cross_conformal_validation()` IS K-fold CV (see `src/ssbc/calibration/cross_conformal.py`)

**Current State**:
- Cross-conformal validation IS implemented and exported
- It's a K-fold cross-validation method
- It's used in examples and documented in `docs/usage.md`

**Recommendation**: Update to:
```
- Distribution-free, finite-sample guarantees are prioritized throughout (SSBC, CP, Hoeffding-style options).
- LOO-CV is used for unbiased operational estimates in the fixed-calibration setting.
- K-fold cross-conformal validation is available as a standalone diagnostic tool (not integrated into PAC bounds).
```

### 3. src/ssbc/reporting/rigorous_report.py - Obsolete Code Example

**Location**: `src/ssbc/reporting/rigorous_report.py` lines 156-157

**Issue**:
- ❌ References non-existent functions in docstring:
  - `compute_mondrian_operational_bounds()` (removed in v1.1.0)
  - `compute_marginal_operational_bounds()` (removed in v1.1.0)

**Current State**:
- These functions were removed in v1.1.0 (see HISTORY.md)
- Replaced by `generate_rigorous_pac_report()`

**Recommendation**: Update the "OLD" example to reflect what was actually removed:
```python
# OLD (removed in v1.1.0):
# op_bounds = compute_mondrian_operational_bounds(...)  # Removed
# marginal_bounds = compute_marginal_operational_bounds(...)  # Removed
```

### 4. src/ssbc/reporting/visualization.py - Obsolete Function References

**Location**: `src/ssbc/reporting/visualization.py` lines 31-33, 51, 55, 334-339

**Issues**:
- ❌ Docstrings reference removed functions:
  - `compute_mondrian_operational_bounds()`
  - `compute_marginal_operational_bounds()`
- ❌ Error messages reference these functions

**Recommendation**: Update all references to point to `generate_rigorous_pac_report()` instead.

## Medium Priority Issues

### 5. HISTORY.md - Outdated Migration Guide

**Location**: `HISTORY.md` lines 168-179

**Issue**:
- Contains migration guide showing removed functions
- This is historical record, but could be confusing for new users

**Recommendation**: Keep as-is (it's historical), but add note: "This migration guide is for historical reference only. Current users should use `generate_rigorous_pac_report()`."

### 6. CODE_REVIEW_IMPROVEMENTS.md - Status Check

**Location**: `CODE_REVIEW_IMPROVEMENTS.md`

**Status**: ✅ Mostly current, but:
- Progress bars section (line 133) marked as "Pending" - should check if this is still pending or if it's been implemented
- Future enhancements section might be outdated

**Recommendation**: Review and update status of "Pending" items.

### 7. LOO_UNCERTAINTY_IMPLEMENTATION.md - Migration Checklist

**Location**: `LOO_UNCERTAINTY_IMPLEMENTATION.md` lines 184-195

**Issue**:
- Contains migration checklist with checkboxes
- Unclear if all items are completed

**Recommendation**:
- Review checklist items
- Mark completed items as ✅
- Remove or archive if all items are complete

### 8. docs/usage.md - Potential Inconsistency

**Location**: `docs/usage.md` line 9

**Issue**:
- Mentions "bootstrap and cross-conformal validation" as uncertainty quantification
- But these are standalone tools, not integrated into `generate_rigorous_pac_report()`

**Status**: ✅ Actually correct - they ARE available as standalone functions
- `bootstrap_calibration_uncertainty()` exists
- `cross_conformal_validation()` exists
- They're just not integrated into the main report

**Recommendation**: Keep as-is, but could clarify: "Available as standalone diagnostic tools (not integrated into PAC bounds)."

## Minor Issues

### 9. README.md - Python Version References

**Location**: `CONTRIBUTING.md` line 95

**Issue**:
- Says "Python 3.12 and 3.13"
- But `pyproject.toml` shows support for 3.10, 3.11, 3.12, 3.13

**Recommendation**: Update to "Python 3.10+"

### 10. docs/installation.md - Installation Method

**Location**: `docs/installation.md` line 8

**Issue**:
- Shows `uv add ssbc` first (modern)
- But most users likely use `pip`

**Recommendation**: Keep `pip` as primary, with `uv` as alternative:
```bash
pip install ssbc
```

Or if `uv` is preferred:
```bash
# Using uv (recommended)
uv add ssbc

# Or using pip
pip install ssbc
```

## Summary of Required Actions

### High Priority (Breaking/Misleading)
1. ✅ Update `CONTRIBUTING.md` - Replace `setup.py`/`make` with `uv`/`just`
2. ✅ Fix `docs/rigorous_report_math.md` - Correct K-fold CV statement
3. ✅ Update `src/ssbc/reporting/rigorous_report.py` - Fix obsolete function references
4. ✅ Update `src/ssbc/reporting/visualization.py` - Fix obsolete function references

### Medium Priority (Clarification)
5. ⚠️ Review `CODE_REVIEW_IMPROVEMENTS.md` - Update pending items status
6. ⚠️ Review `LOO_UNCERTAINTY_IMPLEMENTATION.md` - Complete migration checklist
7. ⚠️ Clarify `docs/usage.md` - Bootstrap/cross-conformal are standalone tools

### Low Priority (Cosmetic)
8. ⚠️ Update `CONTRIBUTING.md` - Python version range
9. ⚠️ Update `docs/installation.md` - Installation method priority

## Files to Review/Update

1. `CONTRIBUTING.md` - Major rewrite needed
2. `docs/rigorous_report_math.md` - Fix contradiction
3. `src/ssbc/reporting/rigorous_report.py` - Fix docstring examples
4. `src/ssbc/reporting/visualization.py` - Fix function references
5. `CODE_REVIEW_IMPROVEMENTS.md` - Review pending items
6. `LOO_UNCERTAINTY_IMPLEMENTATION.md` - Complete/archive checklist
7. `docs/usage.md` - Minor clarification
8. `docs/installation.md` - Minor update

## Notes

- Bootstrap and cross-conformal ARE implemented and available - they're just not integrated into the main report workflow
- The removal of old functions in v1.1.0 is well-documented in HISTORY.md
- Most documentation is current - these are edge cases and inconsistencies
