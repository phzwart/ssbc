# Mathematical Framework for Operational Bounds

## Step 1: Formal Generative Model and Primitive Probabilities

### Generative Model for a Single Sample

For a single sample under **fixed thresholds** learned from calibration data:

- **Y ∈ {0, 1}**: True class label (random variable)
- **S ∈ {singleton, doublet, abstention}**: Prediction set type (determined by conformal prediction thresholds)
- **E ∈ {0, 1}**: Singleton error indicator, defined as:
  - E = 1 if S = singleton AND predicted label ≠ true label
  - E = 0 otherwise (including when S ≠ singleton)

The joint distribution is fully specified by the fixed thresholds and the underlying class distribution.

### Primitive Probabilities

#### Class Prevalences
- \( p_0 = \mathbb{P}(Y=0) \)
- \( p_1 = \mathbb{P}(Y=1) = 1 - p_0 \)

#### Joint Per-Class Operational Rates (Full Sample)

These are the **primary quantities of interest**. They represent the probability of joint events across the entire population:

**Class 0:**
- \( q^{\text{sing}}_0 = \mathbb{P}(Y=0, S=\text{singleton}) \)
- \( q^{\text{dbl}}_0 = \mathbb{P}(Y=0, S=\text{doublet}) \)
- \( q^{\text{abs}}_0 = \mathbb{P}(Y=0, S=\text{abstention}) \)
- \( q^{\text{err}}_0 = \mathbb{P}(Y=0, S=\text{singleton}, E=1) \)

**Class 1:**
- \( q^{\text{sing}}_1 = \mathbb{P}(Y=1, S=\text{singleton}) \)
- \( q^{\text{dbl}}_1 = \mathbb{P}(Y=1, S=\text{doublet}) \)
- \( q^{\text{abs}}_1 = \mathbb{P}(Y=1, S=\text{abstention}) \)
- \( q^{\text{err}}_1 = \mathbb{P}(Y=1, S=\text{singleton}, E=1) \)

#### Conditional Per-Class Rates (Optional)

These are useful for interpretation but are **derived quantities**:

**Class 0:**
- \( r^{\text{sing}}_0 = \mathbb{P}(S=\text{singleton} \mid Y=0) = q^{\text{sing}}_0 / p_0 \)
- \( r^{\text{dbl}}_0 = \mathbb{P}(S=\text{doublet} \mid Y=0) = q^{\text{dbl}}_0 / p_0 \)
- \( r^{\text{abs}}_0 = \mathbb{P}(S=\text{abstention} \mid Y=0) = q^{\text{abs}}_0 / p_0 \)
- \( r^{\text{err}}_0 = \mathbb{P}(E=1 \mid Y=0, S=\text{singleton}) = q^{\text{err}}_0 / q^{\text{sing}}_0 \)

**Class 1:**
- \( r^{\text{sing}}_1 = \mathbb{P}(S=\text{singleton} \mid Y=1) = q^{\text{sing}}_1 / p_1 \)
- \( r^{\text{dbl}}_1 = \mathbb{P}(S=\text{doublet} \mid Y=1) = q^{\text{dbl}}_1 / p_1 \)
- \( r^{\text{abs}}_1 = \mathbb{P}(S=\text{abstention} \mid Y=1) = q^{\text{abs}}_1 / p_1 \)
- \( r^{\text{err}}_1 = \mathbb{P}(E=1 \mid Y=1, S=\text{singleton}) = q^{\text{err}}_1 / q^{\text{sing}}_1 \)

### Exclusion: Global Marginal Rates

We **do not** compute or report:
- \( \mathbb{P}(S=\text{singleton}) \) (mixes classes with different cost structures)
- \( \mathbb{P}(S=\text{doublet}) \) (mixes classes)
- \( \mathbb{P}(E=1 \mid S=\text{singleton}) \) (mixes class-specific error distributions)

These cannot be justified statistically when classes have different cost structures.

---

## Step 2: Mapping Metrics to Bernoulli Events

### Joint Per-Class Rates (Full Sample)

For each joint rate, define a **Bernoulli indicator** on each sample:

#### Class 0 Singleton Rate (Full Sample)
- **Event**: \( Z_i^{\text{sing},0} = \mathbf{1}\{Y_i=0, S_i=\text{singleton}\} \)
- **Mean**: \( \theta^{\text{sing}}_0 = q^{\text{sing}}_0 = \mathbb{E}[Z_i^{\text{sing},0}] \)
- **Calibration**:
  - \( k_{\text{cal}}^{\text{sing},0} = \sum_{i=1}^{N_{\text{cal}}} Z_i^{\text{sing},0} \)
  - \( n_{\text{cal}} = N_{\text{cal}} \) (total calibration size)
- **Test**:
  - \( K_{\text{test}}^{\text{sing},0} = \sum_{j=1}^{N_{\text{test}}} Z_j^{\text{sing},0*} \sim \text{Binomial}(N_{\text{test}}, \theta^{\text{sing}}_0) \)
  - where \( Z_j^{\text{sing},0*} \) are i.i.d. Bernoulli(\( \theta^{\text{sing}}_0 \))

#### Class 0 Singleton Error Rate (Full Sample)
- **Event**: \( Z_i^{\text{err},0} = \mathbf{1}\{Y_i=0, S_i=\text{singleton}, E_i=1\} \)
- **Mean**: \( \theta^{\text{err}}_0 = q^{\text{err}}_0 = \mathbb{E}[Z_i^{\text{err},0}] \)
- **Calibration**:
  - \( k_{\text{cal}}^{\text{err},0} = \sum_{i=1}^{N_{\text{cal}}} Z_i^{\text{err},0} \)
  - \( n_{\text{cal}} = N_{\text{cal}} \)
- **Test**:
  - \( K_{\text{test}}^{\text{err},0} \sim \text{Binomial}(N_{\text{test}}, \theta^{\text{err}}_0) \)

#### Analogous Definitions for:
- Class 0 doublet: \( Z_i^{\text{dbl},0} = \mathbf{1}\{Y_i=0, S_i=\text{doublet}\} \)
- Class 0 abstention: \( Z_i^{\text{abs},0} = \mathbf{1}\{Y_i=0, S_i=\text{abstention}\} \)
- Class 1 singleton, doublet, abstention, error (same pattern)

### Conditional Per-Class Rates

For conditional rates, we restrict to a **subpopulation**:

#### Class 0 Singleton Rate (Conditional on Class 0)
- **Restriction**: Only samples with \( Y_i = 0 \)
- **Event**: \( W_i^{\text{sing}|0} = \mathbf{1}\{S_i = \text{singleton}\} \) (only defined when \( Y_i = 0 \))
- **Mean**: \( r^{\text{sing}}_0 = \mathbb{E}[W_i^{\text{sing}|0} \mid Y_i=0] \)
- **Calibration**:
  - \( n_{\text{cal},0} = \#\{i : Y_i=0\} \) (number of class-0 samples in calibration)
  - \( k_{\text{cal}}^{\text{sing}|0} = \#\{i : Y_i=0, S_i=\text{singleton}\} \)
- **Test**:
  - \( N_{\text{test},0} = \sum_{j=1}^{N_{\text{test}}} \mathbf{1}\{Y_j^*=0\} \) (random number of class-0 samples in test)
  - \( K_{\text{test}}^{\text{sing}|0} = \sum_{j: Y_j^*=0} W_j^{\text{sing}|0*} \sim \text{Binomial}(N_{\text{test},0}, r^{\text{sing}}_0) \)
  - **Note**: The denominator \( N_{\text{test},0} \) is random, which complicates prediction bounds.

#### Class 0 Conditional Error Rate (Conditional on Singleton & Class 0)
- **Restriction**: Only samples with \( Y_i=0 \) AND \( S_i=\text{singleton} \)
- **Event**: \( W_i^{\text{err}|0} = \mathbf{1}\{E_i=1\} \) (only defined when \( Y_i=0, S_i=\text{singleton} \))
- **Mean**: \( r^{\text{err}}_0 = \mathbb{E}[W_i^{\text{err}|0} \mid Y_i=0, S_i=\text{singleton}] \)
- **Calibration**:
  - \( n_{\text{cal},0,\text{sing}} = \#\{i : Y_i=0, S_i=\text{singleton}\} \)
  - \( k_{\text{cal}}^{\text{err}|0} = \#\{i : Y_i=0, S_i=\text{singleton}, E_i=1\} \)
- **Test**:
  - \( N_{\text{test},0,\text{sing}} \) (random number of class-0 singletons in test)
  - \( K_{\text{test}}^{\text{err}|0} \sim \text{Binomial}(N_{\text{test},0,\text{sing}}, r^{\text{err}}_0) \)
  - **Note**: The denominator \( N_{\text{test},0,\text{sing}} \) is random.

---

## Step 3: Predictive Distribution and Use of `prediction_bounds`

### For Joint Per-Class Rates (Full Sample)

**Mathematical Model:**
- Parameter: \( \theta = \mathbb{P}(Z=1) \) where \( Z \in \{Z^{\text{sing},0}, Z^{\text{dbl},0}, Z^{\text{abs},0}, Z^{\text{err},0}, \ldots\} \)
- Calibration: \( K_{\text{cal}} \sim \text{Binomial}(n_{\text{cal}}, \theta) \)
- Test: \( K_{\text{test}} \mid \theta \sim \text{Binomial}(n_{\text{test}}, \theta) \)

**Use of `prediction_bounds(k_cal, n_cal, n_test, confidence, method)`:**
- `k_cal = K_cal` (number of successes in calibration)
- `n_cal = N_cal` (total calibration size)
- `n_test = N_test` (planned test size - **fixed**)

**Method Interpretation:**
- `method="beta_binomial"`: Exact Beta-Binomial predictive interval
  - Posterior: \( p \mid K_{\text{cal}} \sim \text{Beta}(K_{\text{cal}} + 1, N_{\text{cal}} - K_{\text{cal}} + 1) \) (uniform prior)
  - Predictive: \( K_{\text{test}} \mid K_{\text{cal}} \sim \text{Beta-Binomial}(N_{\text{test}}, K_{\text{cal}}+1, N_{\text{cal}}-K_{\text{cal}}+1) \)
  - Bounds: \( [L, U] \) such that \( \mathbb{P}(K_{\text{test}}/N_{\text{test}} \in [L, U] \mid K_{\text{cal}}) \geq \text{confidence} \)

- `method="simple"`: Normal approximation
  - Standard error: \( SE = \sqrt{\hat{p}(1-\hat{p}) \cdot (1/n_{\text{cal}} + 1/n_{\text{test}})} \) where \( \hat{p} = K_{\text{cal}}/n_{\text{cal}} \)
  - Bounds: \( \hat{p} \pm z_{\alpha/2} \cdot SE \)

**LOO-CV with `compute_robust_prediction_bounds`:**
- Input: Array of LOO indicators \( \{Z_i^{\text{LOO}}\}_{i=1}^{N_{\text{cal}}} \) where \( Z_i^{\text{LOO}} = \mathbf{1}\{\text{event on sample } i \text{ with LOO threshold}\} \)
- These are **correlated** (not independent) due to LOO-CV structure
- The function accounts for this correlation via an inflation factor
- Result: Conservative approximation to the Beta-Binomial predictive interval

### For Conditional Per-Class Rates

**Decision**: We use **Option B** (predictive fraction for a test run)

**Mathematical Model:**
- Parameter: \( r = \mathbb{P}(W=1 \mid \text{condition}) \) where condition restricts to a subpopulation
- Calibration: \( K_{\text{cal,cond}} \sim \text{Binomial}(n_{\text{cal,cond}}, r) \)
- Test: Need to estimate future conditional test size:
  - \( \hat{N}_{\text{test,cond}} = N_{\text{test}} \cdot (n_{\text{cal,cond}} / N_{\text{cal}}) \) (point estimate)
  - Then: \( K_{\text{test,cond}} \mid r, \hat{N}_{\text{test,cond}} \sim \text{Binomial}(\hat{N}_{\text{test,cond}}, r) \)

**Use of `prediction_bounds` or `compute_robust_prediction_bounds`:**
- `k_cal = K_{\text{cal,cond}}` (number of successes in conditional subpopulation)
- `n_cal = n_{\text{cal,cond}}` (size of conditional subpopulation in calibration)
- `n_test = \hat{N}_{\text{test,cond}}` (estimated future conditional test size)

**Limitation**: This treats \( \hat{N}_{\text{test,cond}} \) as fixed, but it's actually random. The bounds are **conservative** because they don't account for the additional uncertainty in the denominator. This is documented in the stability note.

**Alternative (Not Used)**: Option A would bound the parameter \( r \) itself, but we want predictive intervals for the empirical rate in a future test run, so Option B is appropriate.

---

## Step 4: Code Audit Checklist

### Operational Bounds Functions

For every call to `prediction_bounds(k_cal, n_cal, n_test, ...)`:
- [ ] Verify `k_cal` counts a **single well-defined Bernoulli event**
- [ ] Verify `n_cal` is the total number of trials for that event
- [ ] Verify `n_test` is the planned number of **future draws of the same event**

For every call to `compute_robust_prediction_bounds(loo_predictions, n_test, ...)`:
- [ ] Verify `loo_predictions` is an array of indicators for the **same Bernoulli event**
- [ ] Verify `n_test` is aligned as above

**Remove "Approach B" rescaling code:**
- [ ] Find any code that constructs bounds by multiplying two separate intervals
- [ ] Replace with direct Bernoulli formulation if needed, or remove if redundant

### Statistical Utilities

- [ ] Clarify docstrings to emphasize that `k_cal` and `n_cal` are for a **single well-defined Bernoulli event**
- [ ] Make explicit that `n_test` is the number of future Bernoulli trials
- [ ] Emphasize that "operational rate" = empirical mean of a binary indicator, not an arbitrary ratio

---

## Step 5: Metric Mapping Table

| Metric Name | Bernoulli Event | Type | `k_cal` | `n_cal` | `n_test` |
|-------------|----------------|------|---------|---------|----------|
| `singleton_rate_class0_bounds` | \( Z_i^{\text{sing},0} = \mathbf{1}\{Y_i=0, S_i=\text{singleton}\} \) | Joint (full sample) | \( \#\{i: Y_i=0, S_i=\text{singleton}\} \) | \( N_{\text{cal}} \) | \( N_{\text{test}} \) |
| `doublet_rate_class0_bounds` | \( Z_i^{\text{dbl},0} = \mathbf{1}\{Y_i=0, S_i=\text{doublet}\} \) | Joint (full sample) | \( \#\{i: Y_i=0, S_i=\text{doublet}\} \) | \( N_{\text{cal}} \) | \( N_{\text{test}} \) |
| `abstention_rate_class0_bounds` | \( Z_i^{\text{abs},0} = \mathbf{1}\{Y_i=0, S_i=\text{abstention}\} \) | Joint (full sample) | \( \#\{i: Y_i=0, S_i=\text{abstention}\} \) | \( N_{\text{cal}} \) | \( N_{\text{test}} \) |
| `singleton_error_rate_class0_bounds` | \( Z_i^{\text{err},0} = \mathbf{1}\{Y_i=0, S_i=\text{singleton}, E_i=1\} \) | Joint (full sample) | \( \#\{i: Y_i=0, S_i=\text{singleton}, E_i=1\} \) | \( N_{\text{cal}} \) | \( N_{\text{test}} \) |
| (Class 1 analogues) | (Same pattern) | Joint (full sample) | (Same pattern) | \( N_{\text{cal}} \) | \( N_{\text{test}} \) |
| `singleton_error_rate_cond_class0_bounds` | \( W_i^{\text{err}\|0} = \mathbf{1}\{E_i=1\} \) given \( Y_i=0, S_i=\text{singleton} \) | Conditional | \( \#\{i: Y_i=0, S_i=\text{singleton}, E_i=1\} \) | \( \#\{i: Y_i=0, S_i=\text{singleton}\} \) | \( \hat{N}_{\text{test},0,\text{sing}} \) |

**Note**: Conditional rates have random denominators, making bounds conservative. This is documented in stability notes.
