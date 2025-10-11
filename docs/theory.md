# Theory and Deployment Guide

═══════════════════════════════════════════════════════════════════════════
## SMALL SAMPLE BETA CORRECTION (SSBC) AND OPERATIONAL PROPERTIES
═══════════════════════════════════════════════════════════════════════════

SSBC provides finite-sample PAC coverage guarantees for conformal prediction by
adjusting the miscoverage level α based on the exact beta distribution of coverage
rates. Unlike asymptotic methods that assume large samples, SSBC accounts for the
discrete nature of order statistics and provides rigorous guarantees even when
calibration data is limited.

## The Coverage Guarantee

With probability ≥ 1-δ over the calibration set, the deployed conformal predictor
achieves coverage ≥ 1-α_target on future exchangeable data. This holds for ANY
sample size n, without relying on asymptotic approximations.

**Mathematical Formulation:**

```
P(Coverage(α') ≥ 1 - α_target) ≥ 1 - δ
```

Where:
- `α_target`: Desired miscoverage rate
- `α'`: SSBC-corrected miscoverage rate (α' < α_target)
- `δ`: PAC risk parameter
- `n`: Calibration set size

## Induced Operational Properties

SSBC correction makes the conformal predictor MORE CONSERVATIVE than naive split
conformal, especially for small samples. The corrected miscoverage level α'
satisfies α' < α_target, leading to a more stringent threshold.

### Operational Consequences

- **LARGER prediction sets** (fewer singletons, more doublets/abstentions)
- **LOWER error rates** within singletons (higher precision when making predictions)
- **HONEST finite-sample behavior** (no optimistic bias from asymptotics)
- **GRACEFUL degradation** with small n (explicitly accounts for uncertainty)

### Example with Class Imbalance

For class-imbalanced Mondrian conformal prediction, SSBC is particularly critical.
The minority class often has limited calibration data (e.g., n=92 vs n=908), making
asymptotic assumptions invalid. SSBC automatically adapts: the minority class gets
a more conservative correction (α' = 0.0645 vs majority's α' = 0.0869 for α=0.1),
inducing wider prediction sets and more abstentions where uncertainty is highest.

## Predicting Deployment Behavior with Operational Rate Estimates

A critical question for deploying conformal prediction in classification systems is:
"What fraction of predictions will be singletons (actionable), doublets (ambiguous),
or abstentions (rejected)?" These operational rates determine system throughput,
human workload, and practical utility.

We provide rigorous confidence interval estimates for these deployment rates via
leave-one-out cross-validation (LOO-CV) with Clopper-Pearson intervals.

### LOO-CV Procedure

1. For each calibration point i, train the conformal predictor on all OTHER points
2. Apply the predictor to point i and record: singleton? doublet? abstention?
3. After n evaluations, apply Clopper-Pearson to get exact binomial CIs

**Example Output:**

```
"With 95% confidence, the deployed system will produce:
 - 92-99% singletons (automated decisions)
 - 0-3% doublets (ambiguous, need review)
 - 0-8% abstentions (rejected, need manual processing)"
```

Critically, LOO-CV ensures these estimates are UNBIASED—each point is evaluated by
a predictor that never saw it during training, mimicking true deployment conditions.

## Mondrian Operational Estimates

For Mondrian conformal prediction, operational rates vary by class, and naively
computing per-class estimates would use stale thresholds. The correct approach:

1. Leave out point i from ANY class
2. Train BOTH class thresholds on remaining n-1 points (split by class)
3. Apply the COMPLETE Mondrian predictor (using both thresholds) to point i
4. Condition results on point i's true class for per-class reporting

This ensures that per-class operational estimates respect the coupled nature of
Mondrian thresholds.

### Example: Per-Class vs Marginal Rates

- **Class 0** (n=908): 92-99% singleton rate (high confidence, large sample)
- **Class 1** (n=92): 61-100% singleton rate (uncertainty reflected, small sample)
- **Marginal** (mixed): 85-97% singleton rate (user's view, ignores true labels)

**Marginal estimates** answer: "What will a user see?"
**Per-class estimates** answer: "How does performance differ by ground truth?"

## Beyond Coverage: Conditional Error Rates

We also estimate P(error | singleton)—the error rate WITHIN singleton predictions.
This is crucial for deployment: users care not just about coverage, but about
precision when the system makes a definitive prediction.

LOO-CV + Clopper-Pearson provides confidence intervals like:

```
"Among singleton predictions, 5-9% will be incorrect with 95% confidence."
```

This enables risk assessment: if 95% of cases are singletons with 7% error, the
system can automate 88% of cases correctly while escalating 12%.

## Why This Matters for Deployment

In deployment scenarios with safety or regulatory requirements, practitioners need
to predict operational behavior BEFORE going live. Questions like:

- "Will we achieve 90% automation?"
- "What human oversight capacity do we need?"
- "What error rate should we expect in automated decisions?"

...require quantitative answers with statistical guarantees.

### Our Framework Provides

- **PAC coverage guarantees** (SSBC): "≥90% of predictions will include true label"
- **Confidence interval estimates** (LOO-CV): "92-99% will be singletons, 5-9% error"
- **Distribution-free**, finite-sample, exact (no asymptotic approximations)
- **Handles class imbalance** (Mondrian) and small samples gracefully

This enables trustworthy deployment planning for conformal prediction in
classification, providing contract-ready guarantees for coverage and honest
statistical estimates for operational rates in automated decision systems, even with
limited calibration data.

## Complete Deployment Workflow

```python
from ssbc import (
    BinaryClassifierSimulator,
    split_by_class,
    mondrian_conformal_calibrate,
    compute_mondrian_operational_bounds,
    compute_marginal_operational_bounds,
    report_prediction_stats,
)

# 1. Generate or load your calibration data
labels, probs = ...  # shape: (n,), (n, 2)
class_data = split_by_class(labels, probs)

# 2. Get PAC coverage guarantees with SSBC
cal_result, pred_stats = mondrian_conformal_calibrate(
    class_data=class_data,
    alpha_target=0.10,  # Target 90% coverage
    delta=0.05,         # 95% PAC confidence
)

# 3. Estimate operational rates with LOO-CV
per_class_bounds = compute_mondrian_operational_bounds(
    calibration_result=cal_result,
    labels=labels,
    probs=probs,
    ci_width=0.95,  # 95% confidence intervals
)

marginal_bounds = compute_marginal_operational_bounds(
    labels=labels,
    probs=probs,
    alpha_target=0.10,
    delta_coverage=0.05,
    ci_width=0.95,
)

# 4. Generate comprehensive deployment report
report_prediction_stats(
    prediction_stats=pred_stats,
    calibration_result=cal_result,
    operational_bounds_per_class=per_class_bounds,
    marginal_operational_bounds=marginal_bounds,
    verbose=True,
)
```

## The Transformation: Theory to Deployment

The combination of SSBC PAC coverage and LOO-CV operational estimates transforms
conformal prediction from a theoretical framework into a deployable technology with
rigorous, actionable guarantees.

**Before:** "Conformal prediction provides coverage guarantees (asymptotically)"

**After:** "Our deployed system will:
- Achieve ≥90% coverage with 95% probability (SSBC)
- Automate 85-97% of decisions with 95% confidence (LOO-CV)
- Have 5-9% error rate in automated decisions with 95% confidence (LOO-CV)
- Require human review for 3-15% of cases with 95% confidence (LOO-CV)"

This level of specificity enables:
- **Resource planning**: How many human reviewers do we need?
- **Risk assessment**: What's our worst-case error rate?
- **SLA guarantees**: Can we contractually promise 90% automation?
- **Regulatory approval**: Demonstrable safety bounds for automated decisions

═══════════════════════════════════════════════════════════════════════════
