# SSBC: Small-Sample Beta Correction

![PyPI version](https://img.shields.io/pypi/v/ssbc.svg)
[![Documentation Status](https://readthedocs.org/projects/ssbc/badge/?version=latest)](https://ssbc.readthedocs.io/en/latest/?version=latest)

**Small-Sample Beta Correction** provides PAC (Probably Approximately Correct) guarantees for conformal prediction with small calibration sets.

* PyPI package: https://pypi.org/project/ssbc/
* Free software: MIT License
* Documentation: https://ssbc.readthedocs.io.

## Overview

SSBC addresses the challenge of constructing valid prediction sets when you have limited calibration data. Traditional conformal prediction assumes large calibration sets, but in practice, data is often scarce. SSBC provides **finite-sample PAC guarantees** and **deployment-ready operational estimates**.

### What Makes SSBC Unique?

Unlike asymptotic methods, SSBC provides:

1. **Finite-Sample PAC Coverage** (via SSBC algorithm)
   - Rigorous guarantees that hold for ANY sample size
   - Automatically adapts to class imbalance
   - Example: "≥90% coverage with 95% probability" even with n=50

2. **Deployment Operational Estimates** (via LOO-CV)
   - Predict automation rates before deployment
   - Confidence intervals on singleton/doublet/abstention rates
   - Example: "85-97% automation, 5-9% error rate" with 95% confidence

3. **Contract-Ready Guarantees**
   - Transform theory into deployable systems
   - Resource planning (human oversight needs)
   - SLA compliance (performance bounds)

**📖 For detailed theory and deployment guide, see [docs/theory.md](docs/theory.md)**

### Key Features

- ✅ **Small-Sample Correction**: PAC-valid conformal prediction for small calibration sets
- ✅ **Mondrian Conformal Prediction**: Per-class calibration for handling class imbalance
- ✅ **LOO-CV Operational Bounds**: Predict deployment behavior (automation rates, error rates)
- ✅ **Comprehensive Statistics**: Detailed reporting with Clopper-Pearson confidence intervals
- ✅ **Hyperparameter Tuning**: Interactive parallel coordinates visualization for parameter optimization
- ✅ **Simulation Tools**: Built-in data generators for testing and validation

## Installation

```bash
pip install ssbc
```

Or from source:

```bash
git clone https://github.com/yourusername/ssbc.git
cd ssbc
pip install -e .
```

## Quick Start

### Complete Deployment Workflow

```python
import numpy as np
from ssbc import (
    BinaryClassifierSimulator,
    split_by_class,
    mondrian_conformal_calibrate,
    compute_mondrian_operational_bounds,
    compute_marginal_operational_bounds,
    report_prediction_stats,
)

# 1. Generate simulated data (or load your own)
sim = BinaryClassifierSimulator(
    p_class1=0.1,
    beta_params_class0=(2, 8),
    beta_params_class1=(8, 2),
    seed=42
)
labels, probs = sim.generate(n_samples=200)

# 2. Split by class for Mondrian CP
class_data = split_by_class(labels, probs)

# 3. Get PAC coverage guarantees with SSBC
cal_result, pred_stats = mondrian_conformal_calibrate(
    class_data=class_data,
    alpha_target=0.10,  # Target 90% coverage
    delta=0.05,         # 95% PAC confidence
)

# 4. Estimate operational rates with LOO-CV
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

# 5. Generate comprehensive deployment report
report_prediction_stats(
    prediction_stats=pred_stats,
    calibration_result=cal_result,
    operational_bounds_per_class=per_class_bounds,
    marginal_operational_bounds=marginal_bounds,
    verbose=True,
)
```

**Output includes:**
- ✅ PAC coverage guarantee: "≥90% coverage with 95% probability"
- ✅ Automation estimate: "85-97% singletons with 95% confidence"
- ✅ Error rate estimate: "5-9% errors in singletons with 95% confidence"
- ✅ Per-class breakdowns for minority/majority classes

## Core Algorithm: SSBC

The SSBC algorithm finds the optimal corrected miscoverage rate α' that satisfies:

**P(Coverage(α') ≥ 1 - α_target) ≥ 1 - δ**

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

### Parameters

- `alpha_target`: Target miscoverage rate (e.g., 0.10 for 90% coverage)
- `n`: Calibration set size
- `delta`: PAC risk tolerance (probability of violating guarantee)
- `mode`: "beta" (infinite test) or "beta-binomial" (finite test)

## Module Structure

The library is organized into focused modules:

### Core Modules

- **`ssbc.core`**: Core SSBC algorithm (`ssbc_correct`, `SSBCResult`)
- **`ssbc.conformal`**: Mondrian conformal prediction (`mondrian_conformal_calibrate`, `split_by_class`)
- **`ssbc.statistics`**: Statistical utilities (`clopper_pearson_intervals`, `cp_interval`)

### Analysis & Visualization

- **`ssbc.visualization`**: Reporting and plotting (`report_prediction_stats`, `plot_parallel_coordinates_plotly`)
- **`ssbc.hyperparameter`**: Parameter tuning (`sweep_hyperparams_and_collect`, `sweep_and_plot_parallel_plotly`)

### Testing & Simulation

- **`ssbc.simulation`**: Data generators (`BinaryClassifierSimulator`)

## Examples

The `examples/` directory contains comprehensive demonstrations:

### 1. Core SSBC Algorithm
```bash
python examples/ssbc_core_example.py
```
Demonstrates the SSBC algorithm for different calibration set sizes.

### 2. Mondrian Conformal Prediction
```bash
python examples/mondrian_conformal_example.py
```
Complete workflow: simulation → calibration → per-class reporting.

### 3. Complete SLA Workflow
```bash
python examples/sla_example.py
```
**Full deployment pipeline**: PAC coverage + LOO-CV operational bounds + comprehensive reporting.
This example demonstrates how to get deployment-ready guarantees combining SSBC and operational estimates.

## Hyperparameter Tuning

Sweep over α and δ values to find optimal configurations:

```python
from ssbc import sweep_and_plot_parallel_plotly
import numpy as np

# Define grid
alpha_grid = np.arange(0.05, 0.20, 0.05)
delta_grid = np.arange(0.05, 0.20, 0.05)

# Run sweep and visualize
df, fig = sweep_and_plot_parallel_plotly(
    class_data=class_data,
    alpha_0=alpha_grid, delta_0=delta_grid,
    alpha_1=alpha_grid, delta_1=delta_grid,
    color='err_all'  # Color by error rate
)

# Save interactive plot
fig.write_html("sweep_results.html")

# Analyze results
print(df[['a0', 'd0', 'cov', 'sing_rate', 'err_all']].head())
```

The interactive plot allows you to:
- Brush (select) ranges on any axis to filter configurations
- Explore trade-offs between coverage, automation, and error rates
- Identify Pareto-optimal hyperparameter settings

## Understanding the Output

### Per-Class Statistics (Conditioned on True Label)

For each class, the report shows:
- **Abstentions**: Empty prediction sets
- **Singletons**: Confident predictions (automated decisions)
  - Correct: True label in singleton set
  - Incorrect: True label not in singleton set
- **Doublets**: Both labels included (escalated to human review)

### Marginal Statistics (Deployment View)

Overall performance metrics ignoring true labels:
- **Coverage**: Fraction of predictions containing the true label
- **Singleton rate**: Fraction of confident predictions (automation level)
- **Escalation rate**: Fraction requiring human review
- **Error rates**: By predicted class and overall

### PAC Bounds

The report includes theoretical and observed singleton error rates:
- **α'_bound**: Theoretical upper bound from PAC analysis
- **α'_observed**: Observed error rate on calibration data
- ✓ if observed ≤ bound (PAC guarantee satisfied)

## Citation

If you use SSBC in your research, please cite:

```bibtex
@software{ssbc2024,
  author = {Zwart, Petrus H},
  title = {SSBC: Small-Sample Beta Correction},
  year = {2024},
  url = {https://github.com/yourusername/ssbc}
}
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Credits

This package was created with [Cookiecutter](https://github.com/audreyfeldroy/cookiecutter) and the [audreyfeldroy/cookiecutter-pypackage](https://github.com/audreyfeldroy/cookiecutter-pypackage) project template.
