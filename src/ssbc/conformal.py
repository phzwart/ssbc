"""Mondrian conformal prediction with SSBC correction."""

from typing import Any, Literal

import numpy as np

from .core import ssbc_correct
from .statistics import cp_interval


def split_by_class(labels: np.ndarray, probs: np.ndarray) -> dict[int, dict[str, Any]]:
    """Split calibration data by true class for Mondrian conformal prediction.

    Parameters
    ----------
    labels : np.ndarray, shape (n,)
        True binary labels (0 or 1)
    probs : np.ndarray, shape (n, 2)
        Classification probabilities [P(class=0), P(class=1)]

    Returns
    -------
    dict
        Dictionary with keys 0 and 1, each containing:
        - 'labels': labels for this class (all same value)
        - 'probs': probabilities for samples in this class
        - 'indices': original indices (for tracking)
        - 'n': number of samples in this class

    Examples
    --------
    >>> labels = np.array([0, 1, 0, 1])
    >>> probs = np.array([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1], [0.2, 0.8]])
    >>> class_data = split_by_class(labels, probs)
    >>> print(class_data[0]['n'])  # Number of class 0 samples
    2
    """
    class_data = {}

    for label in [0, 1]:
        mask = labels == label
        indices = np.where(mask)[0]

        class_data[label] = {"labels": labels[mask], "probs": probs[mask], "indices": indices, "n": np.sum(mask)}

    return class_data


def mondrian_conformal_calibrate(
    class_data: dict[int, dict[str, Any]],
    alpha_target: float | dict[int, float],
    delta: float | dict[int, float],
    mode: Literal["beta", "beta-binomial"] = "beta",
    m: int | None = None,
) -> tuple[dict[int, dict[str, Any]], dict[Any, Any]]:
    """Perform Mondrian (per-class) conformal calibration with SSBC correction.

    For each class, compute:
    1. Nonconformity scores: s(x, y) = 1 - P(y|x)
    2. SSBC-corrected alpha for PAC guarantee
    3. Conformal quantile threshold
    4. Singleton error rate bounds via PAC guarantee

    Then evaluate prediction set sizes on calibration data PER CLASS and MARGINALLY.

    Parameters
    ----------
    class_data : dict
        Output from split_by_class()
    alpha_target : float or dict
        Target miscoverage rate for each class
        If float: same for both classes
        If dict: {0: α0, 1: α1} for per-class control
    delta : float or dict
        PAC risk tolerance for each class
        If float: same for both classes
        If dict: {0: δ0, 1: δ1} for per-class control
    mode : str, default="beta"
        "beta" (infinite test) or "beta-binomial" (finite test)
    m : int, optional
        Test window size for beta-binomial mode

    Returns
    -------
    calibration_result : dict
        Dictionary with keys 0 and 1, each containing calibration info
    prediction_stats : dict
        Dictionary with keys:
        - 0, 1: per-class statistics (conditioned on true label)
        - 'marginal': overall statistics (ignoring true labels)

    Examples
    --------
    >>> labels = np.array([0, 1, 0, 1])
    >>> probs = np.array([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1], [0.2, 0.8]])
    >>> class_data = split_by_class(labels, probs)
    >>> cal_result, pred_stats = mondrian_conformal_calibrate(
    ...     class_data, alpha_target=0.1, delta=0.1
    ... )
    """
    # Handle scalar or dict inputs for alpha and delta
    alpha_dict: dict[int, float]
    if isinstance(alpha_target, int | float):
        alpha_dict = {0: float(alpha_target), 1: float(alpha_target)}
    else:
        # alpha_target is dict[int, float] in this branch
        assert isinstance(alpha_target, dict), "alpha_target must be dict if not scalar"
        alpha_dict = {k: float(v) for k, v in alpha_target.items()}

    delta_dict: dict[int, float]
    if isinstance(delta, int | float):
        delta_dict = {0: float(delta), 1: float(delta)}
    else:
        # delta is dict[int, float] in this branch
        assert isinstance(delta, dict), "delta must be dict if not scalar"
        delta_dict = {k: float(v) for k, v in delta.items()}

    calibration_result = {}

    # Step 1: Calibrate per class
    for label in [0, 1]:
        data = class_data[label]
        n = data["n"]
        alpha_class = alpha_dict[label]
        delta_class = delta_dict[label]

        if n == 0:
            calibration_result[label] = {
                "n": 0,
                "alpha_target": alpha_class,
                "alpha_corrected": None,
                "delta": delta_class,
                "threshold": None,
                "scores": np.array([]),
                "ssbc_result": None,
                "error": "No calibration samples for this class",
            }
            continue

        # Compute nonconformity scores: s(x, y) = 1 - P(y|x)
        true_class_probs = data["probs"][:, label]
        scores = 1.0 - true_class_probs

        # Apply SSBC to get corrected alpha
        ssbc_result = ssbc_correct(alpha_target=alpha_class, n=n, delta=delta_class, mode=mode, m=m)

        alpha_corrected = ssbc_result.alpha_corrected

        # Compute conformal quantile threshold
        k = int(np.ceil((n + 1) * (1 - alpha_corrected)))
        k = min(k, n)

        sorted_scores = np.sort(scores)
        threshold = sorted_scores[k - 1] if k > 0 else sorted_scores[0]

        calibration_result[label] = {
            "n": n,
            "alpha_target": alpha_class,
            "alpha_corrected": alpha_corrected,
            "delta": delta_class,
            "threshold": threshold,
            "scores": sorted_scores,
            "ssbc_result": ssbc_result,
            "k": k,
        }

    # Step 2: Evaluate prediction sets
    if calibration_result[0].get("threshold") is None or calibration_result[1].get("threshold") is None:
        return calibration_result, {
            "error": "Cannot compute prediction sets - missing threshold for at least one class"
        }

    threshold_0 = calibration_result[0]["threshold"]
    threshold_1 = calibration_result[1]["threshold"]

    prediction_stats = {}

    # Step 2a: Evaluate per true class
    for true_label in [0, 1]:
        data = class_data[true_label]
        n_class = data["n"]

        if n_class == 0:
            prediction_stats[true_label] = {"n_class": 0, "error": "No samples in this class"}
            continue

        probs = data["probs"]
        prediction_sets = []

        for i in range(n_class):
            score_0 = 1.0 - probs[i, 0]
            score_1 = 1.0 - probs[i, 1]

            pred_set = []
            if score_0 <= threshold_0:
                pred_set.append(0)
            if score_1 <= threshold_1:
                pred_set.append(1)

            prediction_sets.append(pred_set)

        # Count set sizes and correctness
        n_abstentions = sum(len(ps) == 0 for ps in prediction_sets)
        n_doublets = sum(len(ps) == 2 for ps in prediction_sets)

        n_singletons_correct = sum(ps == [true_label] for ps in prediction_sets)
        n_singletons_incorrect = sum(len(ps) == 1 and true_label not in ps for ps in prediction_sets)
        n_singletons_total = n_singletons_correct + n_singletons_incorrect

        # PAC bounds
        n_escalations = n_doublets + n_abstentions

        if n_escalations > 0 and n_singletons_total > 0:
            rho = n_singletons_total / n_escalations
            kappa = n_abstentions / n_escalations
            alpha_singlet_bound = alpha_dict[true_label] * (1 + 1 / rho) - kappa / rho
            alpha_singlet_observed = n_singletons_incorrect / n_singletons_total if n_singletons_total > 0 else 0.0
        else:
            rho = None
            kappa = None
            alpha_singlet_bound = None
            alpha_singlet_observed = None

        prediction_stats[true_label] = {
            "n_class": n_class,
            "alpha_target": alpha_dict[true_label],
            "delta": delta_dict[true_label],
            "abstentions": cp_interval(n_abstentions, n_class),
            "singletons": cp_interval(n_singletons_total, n_class),
            "singletons_correct": cp_interval(n_singletons_correct, n_class),
            "singletons_incorrect": cp_interval(n_singletons_incorrect, n_class),
            "doublets": cp_interval(n_doublets, n_class),
            "prediction_sets": prediction_sets,
            "pac_bounds": {
                "rho": rho,
                "kappa": kappa,
                "alpha_singlet_bound": alpha_singlet_bound,
                "alpha_singlet_observed": alpha_singlet_observed,
                "n_singletons": n_singletons_total,
                "n_escalations": n_escalations,
            },
        }

    # Step 2b: MARGINAL ANALYSIS (ignoring true labels)
    # Reconstruct full dataset
    all_labels = np.concatenate([class_data[0]["labels"], class_data[1]["labels"]])
    all_probs = np.concatenate([class_data[0]["probs"], class_data[1]["probs"]], axis=0)
    all_indices = np.concatenate([class_data[0]["indices"], class_data[1]["indices"]])

    # Sort back to original order
    sort_idx = np.argsort(all_indices)
    all_labels = all_labels[sort_idx]
    all_probs = all_probs[sort_idx]

    n_total = len(all_labels)

    # Compute prediction sets for all samples
    all_prediction_sets = []
    for i in range(n_total):
        score_0 = 1.0 - all_probs[i, 0]
        score_1 = 1.0 - all_probs[i, 1]

        pred_set = []
        if score_0 <= threshold_0:
            pred_set.append(0)
        if score_1 <= threshold_1:
            pred_set.append(1)

        all_prediction_sets.append(pred_set)

    # Count overall set sizes
    n_abstentions_total = sum(len(ps) == 0 for ps in all_prediction_sets)
    n_singletons_total = sum(len(ps) == 1 for ps in all_prediction_sets)
    n_doublets_total = sum(len(ps) == 2 for ps in all_prediction_sets)

    # Break down singletons by predicted class
    n_singletons_pred_0 = sum(ps == [0] for ps in all_prediction_sets)
    n_singletons_pred_1 = sum(ps == [1] for ps in all_prediction_sets)

    # Compute overall coverage
    n_covered = sum(all_labels[i] in all_prediction_sets[i] for i in range(n_total))
    coverage = n_covered / n_total

    # Compute errors on singletons
    singleton_mask = [len(ps) == 1 for ps in all_prediction_sets]
    n_singletons_covered = sum(all_labels[i] in all_prediction_sets[i] for i in range(n_total) if singleton_mask[i])
    n_singletons_errors = n_singletons_total - n_singletons_covered

    # Overall PAC bounds (using weighted average of alphas for interpretation)
    n_escalations_total = n_doublets_total + n_abstentions_total

    if n_escalations_total > 0 and n_singletons_total > 0:
        rho_marginal = n_singletons_total / n_escalations_total
        kappa_marginal = n_abstentions_total / n_escalations_total

        # Weighted average alpha (by class size)
        n_0 = class_data[0]["n"]
        n_1 = class_data[1]["n"]
        alpha_weighted = (n_0 * alpha_dict[0] + n_1 * alpha_dict[1]) / (n_0 + n_1)

        alpha_singlet_bound_marginal = alpha_weighted * (1 + 1 / rho_marginal) - kappa_marginal / rho_marginal
        alpha_singlet_observed_marginal = n_singletons_errors / n_singletons_total
    else:
        rho_marginal = None
        kappa_marginal = None
        alpha_weighted = None
        alpha_singlet_bound_marginal = None
        alpha_singlet_observed_marginal = None

    prediction_stats["marginal"] = {
        "n_total": n_total,
        "coverage": {"count": n_covered, "rate": coverage, "ci_95": cp_interval(n_covered, n_total)},
        "abstentions": cp_interval(n_abstentions_total, n_total),
        "singletons": {
            **cp_interval(n_singletons_total, n_total),
            "pred_0": n_singletons_pred_0,
            "pred_1": n_singletons_pred_1,
            "errors": n_singletons_errors,
        },
        "doublets": cp_interval(n_doublets_total, n_total),
        "prediction_sets": all_prediction_sets,
        "pac_bounds": {
            "rho": rho_marginal,
            "kappa": kappa_marginal,
            "alpha_weighted": alpha_weighted,
            "alpha_singlet_bound": alpha_singlet_bound_marginal,
            "alpha_singlet_observed": alpha_singlet_observed_marginal,
            "n_singletons": n_singletons_total,
            "n_escalations": n_escalations_total,
        },
    }

    return calibration_result, prediction_stats
