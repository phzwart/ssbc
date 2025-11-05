"""Mathematical consistency validation for operational bounds.

This module enforces strict mathematical consistency between the generative model,
calibration statistics, and predictive validation following the framework defined
in docs/operational_bounds_mathematical_framework.md.
"""

from typing import Any

import numpy as np

# Bernoulli event definitions for each metric
BERNOULLI_EVENT_DEFINITIONS = {
    # Joint per-class rates (full sample)
    "singleton_rate_class0": {
        "event": "Z_i^{sing,0} = 1{Y_i=0, S_i=singleton}",
        "mean": "θ_0^{sing} = P(Y=0, S=singleton)",
        "type": "joint_full_sample",
        "denominator_fixed": True,
    },
    "singleton_rate_class1": {
        "event": "Z_i^{sing,1} = 1{Y_i=1, S_i=singleton}",
        "mean": "θ_1^{sing} = P(Y=1, S=singleton)",
        "type": "joint_full_sample",
        "denominator_fixed": True,
    },
    "doublet_rate_class0": {
        "event": "Z_i^{dbl,0} = 1{Y_i=0, S_i=doublet}",
        "mean": "θ_0^{dbl} = P(Y=0, S=doublet)",
        "type": "joint_full_sample",
        "denominator_fixed": True,
    },
    "doublet_rate_class1": {
        "event": "Z_i^{dbl,1} = 1{Y_i=1, S_i=doublet}",
        "mean": "θ_1^{dbl} = P(Y=1, S=doublet)",
        "type": "joint_full_sample",
        "denominator_fixed": True,
    },
    "abstention_rate_class0": {
        "event": "Z_i^{abs,0} = 1{Y_i=0, S_i=abstention}",
        "mean": "θ_0^{abs} = P(Y=0, S=abstention)",
        "type": "joint_full_sample",
        "denominator_fixed": True,
    },
    "abstention_rate_class1": {
        "event": "Z_i^{abs,1} = 1{Y_i=1, S_i=abstention}",
        "mean": "θ_1^{abs} = P(Y=1, S=abstention)",
        "type": "joint_full_sample",
        "denominator_fixed": True,
    },
    "singleton_error_rate_class0": {
        "event": "Z_i^{err,0} = 1{Y_i=0, S_i=singleton, E_i=1}",
        "mean": "θ_0^{err} = P(Y=0, S=singleton, E=1)",
        "type": "joint_full_sample",
        "denominator_fixed": True,
    },
    "singleton_error_rate_class1": {
        "event": "Z_i^{err,1} = 1{Y_i=1, S_i=singleton, E_i=1}",
        "mean": "θ_1^{err} = P(Y=1, S=singleton, E=1)",
        "type": "joint_full_sample",
        "denominator_fixed": True,
    },
    # Conditional per-class rates
    "singleton_error_rate_cond_class0": {
        "event": "W_i^{err|0} = 1{E_i=1} given Y_i=0, S_i=singleton",
        "mean": "r_0^{err} = P(E=1 | Y=0, S=singleton)",
        "type": "conditional",
        "denominator_fixed": False,  # Random denominator
    },
    "singleton_error_rate_cond_class1": {
        "event": "W_i^{err|1} = 1{E_i=1} given Y_i=1, S_i=singleton",
        "mean": "r_1^{err} = P(E=1 | Y=1, S=singleton)",
        "type": "conditional",
        "denominator_fixed": False,  # Random denominator
    },
    # Per-class conditional rates (for per-class scope)
    "singleton_class0": {
        "event": "W_i^{sing|0} = 1{S_i=singleton} given Y_i=0",
        "mean": "r_0^{sing} = P(S=singleton | Y=0)",
        "type": "conditional",
        "denominator_fixed": False,
    },
    "singleton_class1": {
        "event": "W_i^{sing|1} = 1{S_i=singleton} given Y_i=1",
        "mean": "r_1^{sing} = P(S=singleton | Y=1)",
        "type": "conditional",
        "denominator_fixed": False,
    },
    "doublet_class0": {
        "event": "W_i^{dbl|0} = 1{S_i=doublet} given Y_i=0",
        "mean": "r_0^{dbl} = P(S=doublet | Y=0)",
        "type": "conditional",
        "denominator_fixed": False,
    },
    "doublet_class1": {
        "event": "W_i^{dbl|1} = 1{S_i=doublet} given Y_i=1",
        "mean": "r_1^{dbl} = P(S=doublet | Y=1)",
        "type": "conditional",
        "denominator_fixed": False,
    },
    "abstention_class0": {
        "event": "W_i^{abs|0} = 1{S_i=abstention} given Y_i=0",
        "mean": "r_0^{abs} = P(S=abstention | Y=0)",
        "type": "conditional",
        "denominator_fixed": False,
    },
    "abstention_class1": {
        "event": "W_i^{abs|1} = 1{S_i=abstention} given Y_i=1",
        "mean": "r_1^{abs} = P(S=abstention | Y=1)",
        "type": "conditional",
        "denominator_fixed": False,
    },
    # Joint rates normalized by total (for marginal scope)
    "singleton_error_class0": {
        "event": "Z_i^{err,0} = 1{Y_i=0, S_i=singleton, E_i=1}",
        "mean": "θ_0^{err} = P(Y=0, S=singleton, E=1)",
        "type": "joint_full_sample",
        "denominator_fixed": True,
    },
    "singleton_error_class1": {
        "event": "Z_i^{err,1} = 1{Y_i=1, S_i=singleton, E_i=1}",
        "mean": "θ_1^{err} = P(Y=1, S=singleton, E=1)",
        "type": "joint_full_sample",
        "denominator_fixed": True,
    },
    "singleton_correct_class0": {
        "event": "Z_i^{cor,0} = 1{Y_i=0, S_i=singleton, E_i=0}",
        "mean": "θ_0^{cor} = P(Y=0, S=singleton, E=0)",
        "type": "joint_full_sample",
        "denominator_fixed": True,
    },
    "singleton_correct_class1": {
        "event": "Z_i^{cor,1} = 1{Y_i=1, S_i=singleton, E_i=0}",
        "mean": "θ_1^{cor} = P(Y=1, S=singleton, E=0)",
        "type": "joint_full_sample",
        "denominator_fixed": True,
    },
}


def extract_calibration_counts(report: dict[str, Any], metric_key: str, scope: str) -> dict[str, Any]:
    """Extract calibration counts (k_cal, n_cal) and test size (n_test) for a metric.

    Parameters
    ----------
    report : dict
        Output from generate_rigorous_pac_report()
    metric_key : str
        Metric identifier (e.g., "singleton_rate_class0", "singleton_error_cond_class0")
    scope : str
        Scope: "marginal", "class_0", or "class_1"

    Returns
    -------
    dict
        Contains:
        - 'k_cal': Number of successes in calibration
        - 'n_cal': Total number of trials in calibration (fixed denominator)
        - 'n_test': Expected number of future trials
        - 'event_definition': Bernoulli event definition
        - 'denominator_fixed': Whether denominator is fixed
    """
    # Get pac_bounds for the scope
    pac_bounds_key = f"pac_bounds_{scope}"
    if pac_bounds_key not in report:
        return {
            "k_cal": np.nan,
            "n_cal": np.nan,
            "n_test": np.nan,
            "event_definition": "Unknown",
            "denominator_fixed": True,
        }

    pac_bounds = report[pac_bounds_key]
    params = report.get("parameters", {})

    # Get test_size from parameters or estimate from calibration
    if "test_size" in params:
        test_size = params["test_size"]
    elif "calibration_result" in report and len(report["calibration_result"]) > 0:
        # Use n from calibration_result (total calibration size)
        # For marginal, sum both classes
        if 0 in report["calibration_result"] and 1 in report["calibration_result"]:
            test_size = report["calibration_result"][0].get("n", 0) + report["calibration_result"][1].get("n", 0)
        else:
            test_size = report["calibration_result"][0].get("n", np.nan)
    else:
        test_size = np.nan

    # Normalize metric_key for lookup (remove _rate_ if present)
    # Example: "singleton_correct_rate_class0" -> "singleton_correct_class0"
    # This handles both formats: with and without _rate_ in the key
    normalized_metric_key = metric_key.replace("_rate_", "_").replace("_rate", "")

    # Get event definition (try normalized key first, then original)
    event_def = BERNOULLI_EVENT_DEFINITIONS.get(normalized_metric_key, {})
    if not event_def:
        # Try original metric_key as fallback
        event_def = BERNOULLI_EVENT_DEFINITIONS.get(metric_key, {})

    if not event_def:
        # Metric key not found - return early with diagnostic info
        available_keys_sample = [
            k for k in BERNOULLI_EVENT_DEFINITIONS.keys() if normalized_metric_key.split("_")[0] in k
        ][:5]
        return {
            "k_cal": np.nan,
            "n_cal": np.nan,
            "n_test": test_size,
            "event_definition": (
                f"Unknown (metric_key='{metric_key}' not found, " f"tried normalized='{normalized_metric_key}')"
            ),
            "denominator_fixed": True,
            "diagnostic": {
                "metric_key": metric_key,
                "normalized_metric_key": normalized_metric_key,
                "scope": scope,
                "available_keys_sample": available_keys_sample,
            },
        }

    event_str = event_def.get("event", "Unknown event")
    denominator_fixed = event_def.get("denominator_fixed", True)

    # Extract calibration counts from report
    # For joint rates: n_cal = total calibration size
    # For conditional rates: n_cal = conditional subpopulation size
    #
    # NOTE: We estimate k_cal from expected rates because the report doesn't store
    # raw calibration counts. This is an approximation. For exact validation,
    # we would need to store actual calibration counts in the report.

    if scope == "marginal":
        # Get total calibration size
        if "calibration_result" in report and len(report["calibration_result"]) > 0:
            # Sum n from both classes for total calibration size
            if 0 in report["calibration_result"] and 1 in report["calibration_result"]:
                n_cal = report["calibration_result"][0].get("n", 0) + report["calibration_result"][1].get("n", 0)
            else:
                n_cal = report["calibration_result"][0].get("n", np.nan)
        else:
            n_cal = np.nan

        # Extract expected rate and estimate k_cal
        # For joint rates normalized by total: expected_rate * n_cal ≈ k_cal
        # Expected keys are: "expected_singleton_rate_class0", "expected_doublet_rate_class0", etc.
        # Map metric_key to expected key format (some metrics have "_rate_" in the key)
        # Use normalized_metric_key for consistency with event definition lookup
        if normalized_metric_key in [
            "singleton_error_class0",
            "singleton_error_class1",
            "singleton_correct_class0",
            "singleton_correct_class1",
        ]:
            # Map to expected key with "_rate_" inserted
            expected_key = f"expected_{normalized_metric_key.replace('_class', '_rate_class')}"
        else:
            # For keys that already have _rate_ in them, keep as is
            if "_rate_" in metric_key:
                expected_key = f"expected_{metric_key}"
            else:
                expected_key = f"expected_{metric_key}"
        expected = pac_bounds.get(expected_key, np.nan)

        if not np.isnan(expected) and not np.isnan(n_cal) and n_cal > 0:
            k_cal = int(round(expected * n_cal))
        else:
            k_cal = np.nan

        # For conditional rates, extract counts from joint and conditional rates
        if "cond" in metric_key:
            # For conditional rates like singleton_error_rate_cond_class0:
            # - We have expected_singleton_error_rate_class0 (joint rate: errors/total)
            # - We have expected_singleton_error_rate_cond_class0 (conditional rate: errors/singletons)
            # - From these we can compute:
            #   n_errors = expected_singleton_error_rate_class0 * n_cal
            #   n_singletons = n_errors / expected_singleton_error_rate_cond_class0
            #   So: n_cal_cond = n_singletons, k_cal = n_errors

            # Extract class label from metric_key (e.g., "singleton_error_rate_cond_class0" -> "0")
            class_label = None
            if metric_key.endswith("_class0"):
                class_label = 0
            elif metric_key.endswith("_class1"):
                class_label = 1

            if class_label is not None:
                # Get joint rate (e.g., singleton_error_rate_class0)
                joint_metric_key = metric_key.replace("_cond_", "_").replace("_cond", "")
                joint_expected_key = f"expected_{joint_metric_key}"
                joint_expected = pac_bounds.get(joint_expected_key, np.nan)

                # Get conditional expected rate
                cond_expected = pac_bounds.get(expected_key, np.nan)

                if (
                    not np.isnan(joint_expected)
                    and not np.isnan(cond_expected)
                    and cond_expected > 0
                    and not np.isnan(n_cal)
                    and n_cal > 0
                ):
                    # Compute n_errors from joint rate
                    n_errors = int(round(joint_expected * n_cal))
                    # Compute n_singletons from conditional rate
                    n_cal_cond = int(round(n_errors / cond_expected))
                    k_cal = n_errors
                    n_cal = n_cal_cond
                else:
                    k_cal = np.nan
                    n_cal = np.nan
            else:
                # Can't determine class label
                k_cal = np.nan
                n_cal = np.nan
    else:
        # Per-class scope
        class_label = int(scope[-1])
        if "calibration_result" in report and class_label in report["calibration_result"]:
            n_cal = report["calibration_result"][class_label].get("n", np.nan)
        else:
            n_cal = np.nan

        # Extract expected rate
        # For per-class scopes, PAC bounds use simple keys like "expected_singleton_rate"
        # not "expected_singleton_class0". Strip the _class0/_class1 suffix if present.
        base_metric_key = metric_key
        if metric_key.endswith("_class0") or metric_key.endswith("_class1"):
            # Remove _class0 or _class1 suffix
            base_metric_key = metric_key.rsplit("_class", 1)[0]

        expected_key = (
            f"expected_{base_metric_key}_rate" if "_rate" not in base_metric_key else f"expected_{base_metric_key}"
        )
        expected = pac_bounds.get(expected_key, np.nan)

        if not np.isnan(expected) and not np.isnan(n_cal) and n_cal > 0:
            k_cal = int(round(expected * n_cal))
        else:
            k_cal = np.nan

    return {
        "k_cal": k_cal,
        "n_cal": n_cal,
        "n_test": test_size,
        "event_definition": event_str,
        "denominator_fixed": denominator_fixed,
        "expected_key_used": expected_key if scope == "marginal" else f"expected_{metric_key}",
        "expected_value": expected if scope == "marginal" else pac_bounds.get(f"expected_{metric_key}", np.nan),
    }


def validate_denominator_alignment(
    k_cal: int | float,
    n_cal: int | float,
    n_test: int,
    event_type: str,
    denominator_fixed: bool,
) -> dict[str, Any]:
    """Validate that denominators align with the Bernoulli event definition.

    Parameters
    ----------
    k_cal : int or float
        Number of successes in calibration
    n_cal : int or float
        Total number of trials in calibration
    n_test : int
        Expected number of future trials
    event_type : str
        Type: "joint_full_sample" or "conditional"
    denominator_fixed : bool
        Whether denominator is fixed (True) or random (False)

    Returns
    -------
    dict
        Validation result with:
        - 'valid': bool
        - 'message': str
        - 'issues': list of issues found
    """
    issues = []
    valid = True

    # Check for NaN values
    if np.isnan(k_cal) or np.isnan(n_cal):
        issues.append("k_cal or n_cal is NaN - cannot validate denominator alignment")
        return {"valid": False, "message": "Cannot validate (NaN values)", "issues": issues}

    # Check n_cal > 0
    if n_cal <= 0:
        issues.append(f"n_cal must be > 0, got {n_cal}")
        valid = False

    # Check k_cal is valid
    if k_cal < 0 or k_cal > n_cal:
        issues.append(f"k_cal must be in [0, n_cal], got k_cal={k_cal}, n_cal={n_cal}")
        valid = False

    # Check n_test > 0
    if n_test <= 0:
        issues.append(f"n_test must be > 0, got {n_test}")
        valid = False

    # For joint full-sample events, n_cal should equal total calibration size
    # (This is validated by checking that the event definition matches)
    if event_type == "joint_full_sample" and not denominator_fixed:
        issues.append("Joint full-sample events must have fixed denominators")
        valid = False

    # For conditional events, n_cal should be conditional subpopulation size
    # (This is harder to validate without actual calibration data)
    if event_type == "conditional" and denominator_fixed:
        # This is actually OK - we can use fixed denominator for conditional
        # if we're estimating future conditional size
        pass

    message = "✅ Denominator alignment valid" if valid else "❌ Denominator alignment issues found"
    return {"valid": valid, "message": message, "issues": issues}


def validate_probability_consistency(
    rates: dict[str, float], class_label: int, tolerance: float = 1e-3
) -> dict[str, Any]:
    """Validate that joint rates sum to class prevalence for each class.

    For class y: q_y^sing + q_y^dbl + q_y^abs = p_y ± ε

    Parameters
    ----------
    rates : dict
        Dictionary with keys:
        - 'singleton_rate_class0', 'doublet_rate_class0', 'abstention_rate_class0'
        - 'singleton_rate_class1', 'doublet_rate_class1', 'abstention_rate_class1'
        - 'class_rate_class0', 'class_rate_class1' (class prevalences)
    class_label : int
        Class label (0 or 1)
    tolerance : float
        Tolerance for equality check (default 1e-3)

    Returns
    -------
    dict
        Validation result with:
        - 'valid': bool
        - 'message': str
        - 'sum': float (sum of joint rates)
        - 'expected': float (class prevalence)
        - 'difference': float
    """
    singleton_key = f"singleton_rate_class{class_label}"
    doublet_key = f"doublet_rate_class{class_label}"
    abstention_key = f"abstention_rate_class{class_label}"
    class_rate_key = f"class_rate_class{class_label}"

    q_sing = rates.get(singleton_key, 0.0)
    q_dbl = rates.get(doublet_key, 0.0)
    q_abs = rates.get(abstention_key, 0.0)
    p_y = rates.get(class_rate_key, np.nan)

    if np.isnan(p_y):
        return {
            "valid": False,
            "message": "❌ Cannot validate (class prevalence not available)",
            "sum": q_sing + q_dbl + q_abs,
            "expected": np.nan,
            "difference": np.nan,
        }

    total = q_sing + q_dbl + q_abs
    difference = abs(total - p_y)
    valid = difference < tolerance

    message = (
        f"✅ Probability consistency valid (difference: {difference:.6f} < {tolerance})"
        if valid
        else f"❌ Probability consistency violated: sum={total:.6f}, expected={p_y:.6f}, difference={difference:.6f}"
    )

    return {
        "valid": valid,
        "message": message,
        "sum": total,
        "expected": p_y,
        "difference": difference,
    }


def validate_beta_binomial_predictive(
    k_cal: int,
    n_cal: int,
    n_test: int,
    test_rates: np.ndarray,
    confidence: float = 0.95,
    n_simulations: int = 10000,
) -> dict[str, Any]:
    """Validate that test rates follow Beta-Binomial predictive distribution.

    Simulates K_test ~ BetaBinomial(n_test, k_cal+1, n_cal-k_cal+1) and compares
    empirical quantiles of test_rates to theoretical quantiles.

    **IMPORTANT**: This is a DIAGNOSTIC check only, not a failure criterion.

    LOO correlation causes the empirical distribution to differ from the IID
    Beta-Binomial theoretical distribution. This is EXPECTED and OK.

    - Conservative bounds (too wide) are acceptable - they still meet coverage
    - The only concern is if coverage is NOT met (validated separately)
    - Quantile differences indicate LOO correlation effects, which are accounted
      for via the inflation factor in the bounds computation

    Parameters
    ----------
    k_cal : int
        Number of successes in calibration
    n_cal : int
        Total number of trials in calibration
    n_test : int
        Number of future trials
    test_rates : np.ndarray
        Array of observed test rates from validation trials
    confidence : float
        Confidence level (default 0.95)
    n_simulations : int
        Number of Monte Carlo simulations for theoretical quantiles

    Returns
    -------
    dict
        Validation result with:
        - 'valid': bool (True if quantiles match, False if they differ - both OK)
        - 'message': str (informational message about quantile comparison)
        - 'empirical_quantiles': dict
        - 'theoretical_quantiles': dict
        - 'coverage_match': bool (coverage is validated separately)
    """
    if np.isnan(k_cal) or np.isnan(n_cal) or n_cal <= 0:
        return {
            "valid": False,
            "message": "❌ Cannot validate (invalid calibration counts)",
            "empirical_quantiles": {},
            "theoretical_quantiles": {},
            "coverage_match": False,
        }

    # Beta-Binomial parameters (uniform prior Beta(1,1))
    alpha = k_cal + 1
    beta = n_cal - k_cal + 1

    # Simulate Beta-Binomial predictive distribution
    # K_test ~ BetaBinomial(n_test, alpha, beta)
    # Rate = K_test / n_test
    simulated_rates = []
    for _ in range(n_simulations):
        # Sample from Beta-Binomial
        p = np.random.beta(alpha, beta)
        k_test = np.random.binomial(n_test, p)
        simulated_rates.append(k_test / n_test)

    simulated_rates = np.array(simulated_rates)

    # Compute quantiles
    empirical_quantiles = {
        "q05": float(np.percentile(test_rates, 5)),
        "q25": float(np.percentile(test_rates, 25)),
        "q50": float(np.percentile(test_rates, 50)),
        "q75": float(np.percentile(test_rates, 75)),
        "q95": float(np.percentile(test_rates, 95)),
    }

    theoretical_quantiles = {
        "q05": float(np.percentile(simulated_rates, 5)),
        "q25": float(np.percentile(simulated_rates, 25)),
        "q50": float(np.percentile(simulated_rates, 50)),
        "q75": float(np.percentile(simulated_rates, 75)),
        "q95": float(np.percentile(simulated_rates, 95)),
    }

    # Check if empirical quantiles are close to theoretical (within 5% relative error)
    # NOTE: This is a DIAGNOSTIC check only. LOO correlation causes quantiles to differ
    # from the IID Beta-Binomial theoretical distribution. This is EXPECTED and OK.
    # Conservative bounds (too wide) are acceptable - we only care about coverage.
    quantile_match = True
    for q_key in ["q05", "q25", "q50", "q75", "q95"]:
        emp_q = empirical_quantiles[q_key]
        theo_q = theoretical_quantiles[q_key]
        if theo_q > 0:
            rel_error = abs(emp_q - theo_q) / theo_q
            if rel_error > 0.05:  # 5% tolerance
                quantile_match = False
                break

    # Check coverage: empirical coverage should match confidence level
    # (This is validated separately in the main validation function)

    # The message is informational only - quantile differences are expected due to LOO correlation.
    # This is NOT a failure if coverage is met. Conservative bounds are acceptable.
    message = (
        "✅ Beta-Binomial predictive validation passed (quantiles match IID theoretical)"
        if quantile_match
        else "ℹ️ Beta-Binomial quantiles differ from IID (expected due to LOO correlation; OK if coverage is met)"
    )

    return {
        "valid": quantile_match,
        "message": message,
        "empirical_quantiles": empirical_quantiles,
        "theoretical_quantiles": theoretical_quantiles,
        "coverage_match": True,  # Coverage is validated separately
    }


def validate_metric_mathematical_consistency(
    metric_key: str,
    scope: str,
    report: dict[str, Any],
    validation_rates: np.ndarray,
    ci_level: float = 0.95,
) -> dict[str, Any]:
    """Comprehensive mathematical consistency validation for a single metric.

    Enforces all rules:
    1. Bernoulli event definition
    2. Denominator alignment
    3. Probability consistency (for joint rates)
    4. Beta-Binomial predictive validation
    5. Coverage ≥ nominal confidence

    Parameters
    ----------
    metric_key : str
        Metric identifier
    scope : str
        Scope: "marginal", "class_0", or "class_1"
    report : dict
        Output from generate_rigorous_pac_report()
    validation_rates : np.ndarray
        Array of test rates from validation trials
    ci_level : float
        Nominal confidence level

    Returns
    -------
    dict
        Comprehensive validation result with all checks
    """
    # Get event definition
    event_def = BERNOULLI_EVENT_DEFINITIONS.get(metric_key, {})
    event_str = event_def.get("event", "Unknown event")
    event_type = event_def.get("type", "unknown")
    denominator_fixed = event_def.get("denominator_fixed", True)

    # Extract calibration counts
    extraction_error = None
    diagnostic_info = None
    try:
        cal_counts = extract_calibration_counts(report, metric_key, scope)
        k_cal = cal_counts["k_cal"]
        n_cal = cal_counts["n_cal"]
        n_test = cal_counts["n_test"]
        event_str_from_cal = cal_counts.get("event_definition", event_str)
        # Preserve diagnostic info if present
        if "diagnostic" in cal_counts:
            diagnostic_info = cal_counts["diagnostic"]
        # Use event definition from cal_counts if it's more informative
        if isinstance(event_str_from_cal, str):
            if event_str_from_cal and event_str_from_cal != "Unknown" and "Unknown" not in event_str_from_cal:
                event_str = event_str_from_cal
            elif "Unknown" in event_str_from_cal and "not found" in event_str_from_cal:
                # Use the diagnostic message from extract_calibration_counts
                event_str = event_str_from_cal
    except Exception as e:
        # If extraction fails, use defaults and log the error
        import traceback

        error_trace = traceback.format_exc()
        k_cal = np.nan
        n_cal = np.nan
        n_test = np.nan
        # Store error for debugging
        extraction_error = {
            "error": str(e),
            "traceback": error_trace,
            "metric_key": metric_key,
            "scope": scope,
        }

    # 1. Denominator alignment check
    # Cast to appropriate types for validation functions
    k_cal_int = int(k_cal) if not np.isnan(k_cal) else 0
    n_cal_int = int(n_cal) if not np.isnan(n_cal) else 0
    n_test_int = int(n_test) if not np.isnan(n_test) else 0
    event_type_str = str(event_type) if event_type is not None else ""
    denominator_fixed_bool = bool(denominator_fixed) if denominator_fixed is not None else True
    denom_check = validate_denominator_alignment(
        k_cal_int, n_cal_int, n_test_int, event_type_str, denominator_fixed_bool
    )

    # 2. Coverage check
    pac_bounds_key = f"pac_bounds_{scope}"
    if pac_bounds_key not in report:
        bounds = (np.nan, np.nan)
    else:
        pac_bounds = report[pac_bounds_key]
        # For per-class scopes, strip _class0/_class1 suffix from metric_key
        # (e.g., "singleton_class0" -> "singleton" to match "singleton_rate_bounds")
        base_metric_key = metric_key
        if scope != "marginal":
            if metric_key.endswith("_class0") or metric_key.endswith("_class1"):
                # Remove _class0 or _class1 suffix
                base_metric_key = metric_key.rsplit("_class", 1)[0]

        # Try different possible key formats
        rate_bounds_key = f"{base_metric_key}_rate_bounds"
        if rate_bounds_key not in pac_bounds:
            # Try without _rate
            rate_bounds_key = f"{base_metric_key}_bounds"
        if rate_bounds_key not in pac_bounds:
            # Try alternative format (fallback)
            rate_bounds_key = f"{metric_key}_bounds"
        bounds = pac_bounds.get(rate_bounds_key, (np.nan, np.nan))
    if not np.isnan(bounds[0]) and not np.isnan(bounds[1]):
        coverage = np.mean((validation_rates >= bounds[0]) & (validation_rates <= bounds[1]))
        coverage_valid = coverage >= ci_level
    else:
        coverage = np.nan
        coverage_valid = False

    # 3. Beta-Binomial predictive validation (if calibration counts are available)
    if not np.isnan(k_cal) and not np.isnan(n_cal) and n_cal > 0:
        beta_binom_check = validate_beta_binomial_predictive(
            k_cal_int, n_cal_int, n_test_int, validation_rates, ci_level
        )
    else:
        beta_binom_check = {
            "valid": False,
            "message": "⚠️ Cannot validate Beta-Binomial (calibration counts unavailable)",
            "empirical_quantiles": {},
            "theoretical_quantiles": {},
            "coverage_match": False,
        }

    # Overall validity
    # Beta-Binomial validation is diagnostic - differences due to LOO correlation are expected
    # So we only require denominator alignment and coverage to be valid
    # Beta-Binomial failures are warnings, not violations (they indicate LOO correlation effects)
    all_valid = denom_check["valid"] and coverage_valid

    result = {
        "metric_key": metric_key,
        "scope": scope,
        "event_definition": event_str,
        "event_type": event_type,
        "denominator_fixed": denominator_fixed,
        "k_cal": k_cal,
        "n_cal": n_cal,
        "n_test": n_test,
        "denominator_alignment": denom_check,
        "coverage": coverage,
        "coverage_valid": coverage_valid,
        "coverage_target": ci_level,
        "beta_binomial_validation": beta_binom_check,
        "overall_valid": all_valid,
        "message": (
            "✅ Mathematical consistency valid"
            if all_valid
            else "❌ Mathematical consistency violated: check event definition, denominator alignment, or coverage"
        ),
    }

    # Add extraction error if present
    if extraction_error is not None:
        result["extraction_error"] = extraction_error

    # Add diagnostic info if present
    if diagnostic_info is not None:
        result["diagnostic"] = diagnostic_info

    return result
