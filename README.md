# SSBC: Small-Sample Beta Correction

![PyPI version](https://img.shields.io/pypi/v/ssbc.svg)
[![Documentation Status](https://readthedocs.org/projects/ssbc/badge/?version=latest)](https://ssbc.readthedocs.io/en/latest/?version=latest)

**Small-Sample Beta Correction** provides PAC (Probably Approximately Correct) guarantees for conformal prediction with small calibration sets.

* PyPI package: https://pypi.org/project/ssbc/
* Free software: MIT License
* Documentation: https://ssbc.readthedocs.io.

## Overview

SSBC addresses the challenge of constructing valid prediction sets when you have limited calibration data. Traditional conformal prediction assumes large calibration sets, but in practice, data is often scarce. SSBC provides **finite-sample PAC guarantees** and **rigorous operational bounds** for deployment.

### What Makes SSBC Unique?

Unlike asymptotic methods, SSBC provides:

1. **Finite-Sample PAC Coverage** (via SSBC algorithm)
   - Rigorous guarantees that hold for ANY sample size
   - Automatically adapts to class imbalance via Mondrian conformal prediction
   - Example: "≥90% coverage with 95% probability" even with n=50

2. **Rigorous Operational Bounds** (via LOO-CV + Clopper-Pearson)
   - PAC-controlled bounds on automation rates, error rates, escalation rates
   - Confidence intervals account for estimation uncertainty
   - Example: "Singleton rate [0.85, 0.97] with 90% PAC guarantee"

3. **Uncertainty Quantification**
   - Bootstrap analysis for recalibration uncertainty
   - Cross-conformal validation for finite-sample diagnostics
   - Empirical validation for verifying theoretical guarantees

4. **Contract-Ready Guarantees**
   - Transform theory into deployable systems
   - Resource planning (human oversight needs)
   - SLA compliance (performance bounds)

### Core Statistical Properties

🎯 **Distribution-Free**: No assumptions about data distribution
🎯 **Model-Agnostic**: Works with ANY probabilistic classifier
🎯 **Frequentist**: Valid frequentist guarantees, no prior needed
🎯 **Non-Bayesian**: No Bayesian assumptions or hyperpriors
🎯 **Finite-Sample**: Exact guarantees for small n, not asymptotic
🎯 **Exchangeability Only**: Minimal assumption (test/calibration exchangeable)

**📖 For detailed theory and deployment guide, see [docs/theory.md](docs/theory.md)**

## Installation

```bash
pip install ssbc
```

Or from source:

```bash
git clone https://github.com/phzwart/ssbc.git
cd ssbc
pip install -e .
```

## Quick Start

### Unified Workflow (Recommended)

The complete workflow is available through a single function:

```python
from ssbc import BinaryClassifierSimulator, generate_rigorous_pac_report

# Generate or load calibration data
sim = BinaryClassifierSimulator(
    p_class1=0.2,
    beta_params_class0=(1, 7),
    beta_params_class1=(5, 2),
    seed=42
)
labels, probs = sim.generate(n_samples=100)

# Generate comprehensive PAC report with operational bounds
report = generate_rigorous_pac_report(
    labels=labels,
    probs=probs,
    alpha_target=0.10,     # Target 90% coverage
    delta=0.10,            # 90% PAC confidence
    test_size=1000,        # Expected deployment size
    use_union_bound=True,  # Simultaneous guarantees
)

# Access results
pac_bounds = report['pac_bounds_marginal']
print(f"Singleton rate: {pac_bounds['singleton_rate_bounds']}")
print(f"Expected: {pac_bounds['expected_singleton_rate']:.3f}")
```

**Output includes:**
- ✅ PAC coverage guarantees (SSBC-corrected thresholds)
- ✅ Rigorous operational bounds (singleton, doublet, abstention, error rates)
- ✅ Per-class and marginal statistics
- ✅ Class-conditional error metrics (P(error | singleton & class))

### Core SSBC Algorithm

For fine-grained control, use the core algorithm directly:

```python
from ssbc import ssbc_correct

result = ssbc_correct(
    alpha_target=0.10,  # Target 10% miscoverage
    n=50,               # Calibration set size
    delta=0.10,         # PAC parameter (90% confidence)
    mode="beta"         # Infinite test window
)

print(f"Corrected α: {result.alpha_corrected:.4f}")
print(f"u*: {result.u_star}")
```

### Validation and Diagnostics

Empirically validate your PAC bounds:

```python
from ssbc import validate_pac_bounds, print_validation_results

# Generate report
report = generate_rigorous_pac_report(labels, probs, delta=0.10)

# Validate empirically
validation = validate_pac_bounds(
    report=report,
    simulator=sim,
    test_size=1000,
    n_trials=10000
)

# Print results
print_validation_results(validation)
```

Cross-conformal validation for calibration diagnostics:

```python
from ssbc import cross_conformal_validation

results = cross_conformal_validation(
    labels=labels,
    probs=probs,
    n_folds=10,
    alpha_target=0.10,
    delta=0.10
)

print(f"Singleton rate: {results['marginal']['singleton']['mean']:.3f}")
print(f"Std dev: {results['marginal']['singleton']['std']:.3f}")
```

## Key Features

- ✅ **Small-Sample Correction**: PAC-valid conformal prediction for small calibration sets
- ✅ **Mondrian Conformal Prediction**: Per-class calibration for handling class imbalance
- ✅ **PAC Operational Bounds**: Rigorous bounds on deployment rates (LOO-CV + Clopper-Pearson)
- ✅ **LOO-CV Uncertainty Correction**: Small-sample uncertainty quantification
- ✅ **Method Comparison**: Analytical, exact, and Hoeffding bounds comparison
- ✅ **Empirical Validation**: Verify theoretical guarantees in practice
- ✅ **Comprehensive Statistics**: Detailed reporting with exact confidence intervals
- ✅ **Hyperparameter Tuning**: Interactive parallel coordinates visualization
- ✅ **Simulation Tools**: Built-in data generators for testing

## Examples

The `examples/` directory contains comprehensive demonstrations:

### Essential Examples

```bash
# Core algorithm
python examples/ssbc_core_example.py

# Mondrian conformal prediction
python examples/mondrian_conformal_example.py

# Complete workflow with all uncertainty analyses
python examples/complete_workflow_example.py

# SLA/deployment contracts
python examples/sla_example.py

# Alpha scanning across thresholds
python examples/alpha_scan_example.py

# Empirical validation
python examples/pac_validation_example.py
```

## Understanding the Output

### Per-Class Statistics (Conditioned on True Label)

For each class, the report shows:
- **Abstentions**: Empty prediction sets (no confident prediction)
- **Singletons**: Single-label predictions (automated decisions)
- **Doublets**: Both labels included (escalated to human review)
- **Singleton Error Rate**: P(error | singleton prediction)

### Marginal Statistics (Deployment View)

Overall performance metrics (deployment perspective):
- **Coverage**: Fraction of predictions containing the true label
- **Automation Rate**: Fraction of confident predictions (singletons)
- **Escalation Rate**: Fraction requiring human review (doublets + abstentions)
- **Error Rate**: Among automated decisions

### PAC Operational Bounds

Rigorous bounds on all operational metrics:
- Computed via Leave-One-Out Cross-Validation (LOO-CV)
- Clopper-Pearson confidence intervals account for estimation uncertainty
- Union bound ensures all metrics hold simultaneously
- Valid for any future test set from the same distribution

## Citation

If you use SSBC in your research, please cite:

```bibtex
@software{ssbc2024,
  author = {Zwart, Petrus H},
  title = {SSBC: Small-Sample Beta Correction},
  year = {2024},
  url = {https://github.com/phzwart/ssbc}
}
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Credits

This package was created with [Cookiecutter](https://github.com/audreyfeldroy/cookiecutter) and the [audreyfeldroy/cookiecutter-pypackage](https://github.com/audreyfeldroy/cookiecutter-pypackage) project template.
