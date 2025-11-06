### Rigorous PAC Report: Mathematical Flow and Implementation Map

This document details the mathematics and end-to-end flow used by the rigorous report generator, with precise pointers to the implementation. It is intended for automated inspection and as a base for the manuscript Methods section.

- Public entrypoint: `ssbc.reporting.rigorous_report.generate_rigorous_pac_report`
- Core math dependencies: SSBC beta correction, Mondrian conformal thresholds, LOO-CV unbiased rate estimation, prediction bounds (calibration + test uncertainty), Clopper–Pearson exact CIs, optional Bonferroni/union bounds.

### End-to-end flow (high level)

```mermaid
flowchart TD
    A[Inputs: labels, probs, alpha_target, delta, test_size, ci_level] --> B[Split by true class]
    B --> C1[SSBC per class: alpha -> alpha']
    B --> C2[Mondrian conformal calibration: thresholds u*_0, u*_1]
    C1 --> D[Quantile indices k_0, k_1 from alpha']
    C2 --> D
    D --> E[LOO-CV across calibration set using k_0, k_1]
    E --> F[Unbiased point estimates of rates on calibration distribution]
    F --> G1[Marginal operational bounds]
    F --> G2[Per-class operational bounds]
    G1 --> H1[Prediction bounds (calibration + test sampling) at CI level]
    G2 --> H2[Prediction bounds (calibration + test sampling) at CI level]
    H1 --> I1[Optional union/Bonferroni across metrics]
    H2 --> I2[Optional union/Bonferroni across metrics]
    I1 --> J[Report: bounds + diagnostics (optional LOO methods)]
    I2 --> J
```

### Notation

- Calibration data: size n, split into classes with counts n_0, n_1
- Target miscoverage per class: α_c, PAC risk per class: δ_c, confidence for CIs/prediction bounds: γ (e.g., 0.95)
- SSBC-corrected miscoverage: α'_c = SSBC(α_c, n_c, δ_c)
- Mondrian conformal threshold per class: u*_c (quantile threshold for class c)
- Quantile index from α'_c: k_c = ceil((n_c + 1) · (1 − α'_c))
- LOO-CV prediction-set categories for sample i: singleton_i, doublet_i, abstention_i ∈ {0,1}
- Singleton correctness: singleton_correct_i ∈ {0,1}
- Point rates: r̂_singleton, r̂_doublet, r̂_abstention, r̂_error|singleton

### Step-by-step math with implementation pointers

1) Class split and PAC levels
- Observed class split: `(n_0, n_1)` from labels.
- PAC levels used:
  - Per-class: 1 − δ_c
  - Marginal (deployment): (1 − δ_0)(1 − δ_1) since the split is observed.
- Code: `generate_rigorous_pac_report` in `src/ssbc/reporting/rigorous_report.py` lines 155–165 and 289–299.

2) SSBC beta correction per class
- Goal: obtain α'_c ensuring finite-sample PAC coverage for each class with risk δ_c.
- Implementation: `ssbc.core_pkg.ssbc_correct(..., mode="beta")` used twice (for c=0,1).
- Code: `src/ssbc/reporting/rigorous_report.py` lines 166–169.

3) Mondrian conformal calibration (per class)
- Produces thresholds `u*_0, u*_1` and calibration prediction stats (counts of abstentions/singletons/doublets and conditional error summaries).
- Implementation: `ssbc.calibration.mondrian_conformal_calibrate` with `mode="beta"`.
- Code: `src/ssbc/reporting/rigorous_report.py` lines 170–174.

4) Quantile index from SSBC-corrected α
- For each class c: k_c = ceil((n_c + 1) · (1 − α'_c)).
- Used to form LOO-CV prediction sets deterministically with fixed thresholds.
- Code (marginal + LOO-corrected paths): `src/ssbc/metrics/operational_bounds_simple.py` lines 165–171 and 369–374.

5) Leave-one-out CV for unbiased fixed-calibration rates
- For each i, remove i, compute whether sample i would be singleton/doublet/abstention using k_0, k_1 and fixed thresholds; compute singleton correctness.
- Aggregate across i to obtain counts:
  - n_singletons = Σ_i singleton_i
  - n_doublets = Σ_i doublet_i
  - n_abstentions = Σ_i abstention_i
  - n_singletons_correct = Σ_i singleton_correct_i
- Point estimates (calibration distribution):
  - r̂_singleton = n_singletons / n
  - r̂_doublet = n_doublets / n
  - r̂_abstention = n_abstentions / n
  - r̂_error|singleton = (n_singletons − n_singletons_correct) / n_singletons (if n_singletons>0)
- Code: `_evaluate_loo_single_sample_marginal` and aggregation in
  - `src/ssbc/metrics/operational_bounds_simple.py` lines 90–108, 172–193, 396–405.

6) Prediction bounds: accounting for calibration and test uncertainty
- We bound future test-set operational rates for a specified test size N_test.
- Simple method (distribution-free, finite-sample conservative):
  - For a calibration count k_cal out of n_cal, the standard error
    SE = sqrt(p̂(1 − p̂) · (1/n_cal + 1/N_test)), where p̂ = k_cal / n_cal.
  - One-sided lower/upper prediction bounds derived via normal quantiles at level γ.
  - Code: `prediction_bounds_lower`, `prediction_bounds_upper`, `prediction_bounds` in
    `src/ssbc/bounds/statistical.py` lines 193–307 and 389–472.
- Beta–Binomial method (small-sample, more accurate):
  - p ~ Beta(k_cal + 1, n_cal − k_cal + 1), r|p ~ Binomial(N_test, p)/N_test
  - Marginal r ~ BetaBinomial(N_test, ·)/N_test, bound via Beta quantiles + Binomial margin.
  - Code: `prediction_bounds_beta_binomial` in `src/ssbc/bounds/statistical.py` lines 309–386.
- Selection in operational bounds functions via `prediction_method` argument (e.g., "simple", "beta_binomial", "exact", "analytical", "hoeffding", or method comparison "all" in LOO-corrected flows).

7) Clopper–Pearson exact intervals (calibration-only diagnostic summaries)
- Used to summarize calibration-data rates with exact binomial coverage.
- Code: `clopper_pearson_lower`, `clopper_pearson_upper`, `cp_interval` in `src/ssbc/bounds/statistical.py` lines 19–90, 148–191.
- Referenced for printing/reporting statistics in `generate_rigorous_pac_report`.

8) Union/Bonferroni adjustment across multiple metrics (optional)
- When reporting simultaneous guarantees across M metrics, use adjusted γ': 1 − (1 − γ)/M.
- In the marginal path we currently bound up to eight metrics: singleton, doublet, abstention, error|singleton, class-specific and conditional error variants.
- Code: marginal path union adjustment in `src/ssbc/metrics/operational_bounds_simple.py` lines 214–219.

9) Marginal vs. per-class paths
- Marginal: deployment view ignoring true labels, PAC level (1 − δ_0)(1 − δ_1).
  - Without LOO-correction: `compute_pac_operational_bounds_marginal`.
  - With LOO-correction and method diagnostics: `compute_pac_operational_bounds_marginal_loo_corrected`.
- Per-class: conditioned on true class, each with PAC level (1 − δ_c).
  - Without LOO-correction: `compute_pac_operational_bounds_perclass`.
  - With LOO-correction and method diagnostics: `compute_pac_operational_bounds_perclass_loo_corrected`.
- Code: `src/ssbc/reporting/rigorous_report.py` lines 175–275 (marginal/per-class dispatch), and implementations in `src/ssbc/metrics/operational_bounds_simple.py`.

10) LOO-corrected uncertainty methods and diagnostics
- Goal: address correlation structure of LOO folds and small-sample effects by inflating variance or using exact/analytical constructions. Methods include:
  - "analytical": closed-form approximations (recommended n ≥ 40)
  - "exact": enumeration/exact small-sample method (recommended n ≈ 20–40)
  - "hoeffding": distribution-free Hoeffding-style bounds (ultra-conservative)
  - "all": compare and select; diagnostics expose per-method bounds and widths, plus the selected method
- An optional `loo_inflation_factor` can override variance inflation (typical ≈ 2 for LOO).
- Code: `compute_pac_operational_bounds_marginal_loo_corrected` docstring and logic in `src/ssbc/metrics/operational_bounds_simple.py` starting at line 302; analogous per-class function at line 789.

### Mathematical guarantees and assumptions

- Distribution-free frequentist guarantees. No parametric assumptions on score distributions.
- Finite-sample guarantees are maintained via:
  - SSBC beta correction for per-class coverage with PAC risk δ_c.
  - Exact Clopper–Pearson binomial CIs for calibration summaries.
  - Prediction bounds that compound calibration and test sampling uncertainty at specified test size N_test.
  - Optional Bonferroni for simultaneous validity across multiple metrics.
- LOO-CV is used to remove data leakage and produce unbiased estimates of fixed-calibration performance; correlation across LOO folds is handled by conservative variance inflation or exact/analytical methods.

### Report structure (outputs)

The top-level report dictionary assembled by `generate_rigorous_pac_report` contains:
- `ssbc_class_0`, `ssbc_class_1`: SSBC results per class (n_c, α'_c, etc.)
- `pac_bounds_marginal`: Marginal operational bounds and diagnostics
- `pac_bounds_class_0`, `pac_bounds_class_1`: Per-class operational bounds and diagnostics
- `calibration_result`: Mondrian calibration thresholds and per-class stats
- `prediction_stats`: Calibration statistics including Clopper–Pearson summaries
- `parameters`: Echo of α, δ, PAC levels, CI level, test size, union bound flag

Code: `src/ssbc/reporting/rigorous_report.py` lines 277–305.

### Equations (for manuscript)

- SSBC correction per class c: obtain α'_c such that with probability ≥ 1 − δ_c, the conformal set achieves coverage ≥ 1 − α_c for class c on future data.

- Quantile selection: k_c = ceil((n_c + 1) · (1 − α'_c)). Thresholds u*_c determined by Mondrian calibration using class-conditional scores.

- LOO-CV unbiased rates:
  - r̂_singleton = (1/n) Σ_i 1{prediction_set_i is singleton}
  - r̂_doublet = (1/n) Σ_i 1{prediction_set_i is doublet}
  - r̂_abstention = (1/n) Σ_i 1{prediction_set_i is abstention}
  - r̂_error|singleton = (Σ_i 1{singleton and incorrect}) / (Σ_i 1{singleton})

- Prediction bounds (simple method) for future test size N_test:
  - p̂ = k_cal / n_cal, SE = sqrt(p̂(1 − p̂) · (1/n_cal + 1/N_test))
  - lower = max(0, p̂ + z_{α/2} · SE), upper = min(1, p̂ + z_{1−α/2} · SE)

- Beta–Binomial method uses p ~ Beta(k_cal+1, n_cal−k_cal+1), r|p ~ Binomial(N_test, p)/N_test, and inflates bounds with binomial sampling margin at N_test via normal quantiles.

- Bonferroni/union: to guarantee all M metrics simultaneously at confidence γ, use γ' = 1 − (1 − γ)/M for each metric’s bound.

### Key implementation anchors (file:line)

- Report orchestration: `src/ssbc/reporting/rigorous_report.py` 21–305
- Marginal bounds (fixed thresholds): `src/ssbc/metrics/operational_bounds_simple.py` 111–299
- Marginal bounds (LOO-corrected): `src/ssbc/metrics/operational_bounds_simple.py` 302–...
- Per-class bounds (fixed thresholds): `src/ssbc/metrics/operational_bounds_simple.py` 653–...
- Per-class bounds (LOO-corrected): `src/ssbc/metrics/operational_bounds_simple.py` 789–...
- Prediction bounds (simple/Beta–Binomial): `src/ssbc/bounds/statistical.py` 193–307, 309–386, 389–472
- Exact CP intervals: `src/ssbc/bounds/statistical.py` 19–90, 148–191

### Usage linkage (for docs integration)

- This math spec documents the underlying computations used by `examples/complete_workflow_example.py` and the rigorous report API. See also `examples/mondrian_ssbc_thresholds_example.py` for thresholding behavior and `examples/pac_validation_example.py` for validation usage.

### Reproducibility and finite-sample policy

- Distribution-free, finite-sample guarantees are prioritized throughout (SSBC, CP, Hoeffding-style options).
- LOO-CV is used to maintain unbiasedness of operational estimates in the fixed-calibration setting.
- K-fold cross-conformal validation is available as a standalone diagnostic tool (not integrated into PAC bounds computation).
