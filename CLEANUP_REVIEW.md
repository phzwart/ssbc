# Code Cleanup Review for SSBC v0.2.0

## ✅ Settings Already Correct
- `run_bootstrap=False` by default ✓
- `run_cross_conformal=False` by default ✓

---

## 🗑️ DEAD CODE - Recommend REMOVAL

### 1. **`src/ssbc/coverage_distribution.py`** (1400 lines) ⚠️ HIGH PRIORITY
- **Status**: NOT imported anywhere in main codebase
- **Reason**: Superseded by `operational_bounds_simple.py`
- **History**: This was the complex Beta-Binomial approach that was removed
- **Contains**: Old probability-weighted operational bounds, coverage distribution functions
- **Action**: **DELETE** - No longer used, replaced by simpler LOO-CV + CP approach
- **Note**: Has 1 self-referential import in docstring example

### 2. **`src/ssbc/blakers_confidence_interval.py`** (388 lines) ⚠️ HIGH PRIORITY
- **Status**: NOT imported anywhere
- **Reason**: Blaker's CI was removed from workflow (too conservative/degenerate bounds)
- **History**: Implemented, tested, removed per user request "REMOVE THE BLAKER CODE FROM THE WORKFLOW"
- **Action**: **DELETE** - Explicitly removed from workflow
- **Note**: Well-implemented but not needed

---

## ⚠️ DEPRECATED CODE - Recommend DEPRECATION WARNING

### 3. **`src/ssbc/sla.py`** (541 lines)
- **Status**: Exported in `__init__.py`, used in `examples/sla_example.py`
- **Reason**: OLD operational bounds WITHOUT coverage volatility
- **Exports**:
  - `compute_mondrian_operational_bounds()` ← OLD
  - `compute_marginal_operational_bounds()` ← OLD
- **Replaced by**:
  - `compute_pac_operational_bounds_perclass()` (in `operational_bounds_simple.py`)
  - `compute_pac_operational_bounds_marginal()` (in `operational_bounds_simple.py`)
  - Integrated into `generate_rigorous_pac_report()`
- **Action**:
  - **Option A**: DELETE and update `sla_example.py` to use new workflow
  - **Option B**: Add deprecation warning, keep for backward compatibility
  - **Recommendation**: DELETE (clean break for v0.2.0)

**Files to update if deleted**:
- `src/ssbc/__init__.py` - Remove sla exports
- `examples/sla_example.py` - Update to use `generate_rigorous_pac_report()`
- `examples/mondrian_conformal_example.py` - Remove reference to old bounds

---

## 📝 EXAMPLES TO UPDATE/CONSOLIDATE

### Standalone examples using OLD APIs:
1. **`examples/sla_example.py`** - Uses old `sla.py` functions
   - **Action**: Rewrite to use `generate_rigorous_pac_report()`

2. **`examples/mondrian_conformal_example.py`** - References old bounds in comments
   - **Action**: Update comment to reference new workflow

### Potentially redundant examples:
3. **`examples/integrated_pac_bootstrap_example.py`**
4. **`examples/complete_uncertainty_analysis.py`**
5. **`examples/complete_workflow_example.py`**

These three are VERY similar - all show PAC + Bootstrap + Cross-conformal
- **Action**: Consider consolidating into ONE canonical example
- **Recommendation**: Keep `complete_workflow_example.py`, archive the others

---

## 🔍 DUPLICATE/OVERLAPPING FUNCTIONALITY

### Test file confusion:
Multiple test approaches for similar things:
- `tests/test_sla.py` - Tests OLD sla.py functions
- No test file for `operational_bounds_simple.py`

**Action**:
- Create `tests/test_operational_bounds.py` for new functions
- Remove/update `tests/test_sla.py` if sla.py is deleted

---

## 📦 UNUSED EXPORTS in `__init__.py`

Current exports that may not be needed in public API:

```python
# From sla.py (if we delete it)
"OperationalRateBounds",
"OperationalRateBoundsResult",
"compute_marginal_operational_bounds",
"compute_mondrian_operational_bounds",

# These are probably fine to keep:
"compute_operational_rate",  # Utility, used internally
"cp_interval",  # Utility, might be used externally
```

---

## 🎯 RECOMMENDED CLEANUP ACTIONS

### Phase 1: Delete Dead Code (Safe)
```bash
# These are NOT imported anywhere
git rm src/ssbc/coverage_distribution.py
git rm src/ssbc/blakers_confidence_interval.py
```

### Phase 2: Remove Deprecated Code (Breaking)
```bash
# Old operational bounds - superseded by new workflow
git rm src/ssbc/sla.py
git rm tests/test_sla.py
```

### Phase 3: Update __init__.py
Remove exports:
- `OperationalRateBounds`
- `OperationalRateBoundsResult`
- `compute_marginal_operational_bounds`
- `compute_mondrian_operational_bounds`

### Phase 4: Update Examples
- Rewrite `examples/sla_example.py` to use new workflow
- Update `examples/mondrian_conformal_example.py` comment
- Consider consolidating the 3 "complete" examples

### Phase 5: Add Tests
- Create `tests/test_operational_bounds.py`
- Test `compute_pac_operational_bounds_perclass()`
- Test `compute_pac_operational_bounds_marginal()`

---

## 📊 CLEANUP SUMMARY

| File | Lines | Status | Action |
|------|-------|--------|--------|
| `coverage_distribution.py` | 1400 | Dead | DELETE ✓ |
| `blakers_confidence_interval.py` | 388 | Dead | DELETE ✓ |
| `sla.py` | 541 | Deprecated | DELETE or DEPRECATE |
| `test_sla.py` | ? | Tests deprecated code | DELETE or UPDATE |

**Total removal**: ~2300+ lines of dead/deprecated code

---

## 🚀 RECOMMENDATION FOR v0.2.0

**Clean break approach:**
1. Delete all dead code (coverage_distribution, blakers)
2. Delete deprecated sla.py
3. Update all examples to new workflow
4. Update __init__.py exports
5. Add tests for new operational bounds
6. Update HISTORY.md with breaking changes

**Result**: Clean, focused codebase with single unified workflow via `generate_rigorous_pac_report()`

**Migration guide for users:**
```python
# OLD (v0.1.x)
from ssbc import compute_mondrian_operational_bounds
bounds = compute_mondrian_operational_bounds(...)

# NEW (v0.2.0)
from ssbc import generate_rigorous_pac_report
report = generate_rigorous_pac_report(...)
pac_bounds = report['pac_bounds_class_0']
```
