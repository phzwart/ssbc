"""MCP Server for SSBC - Small-Sample Beta Correction.

Exposes SSBC functionality as MCP tools for AI assistants.
"""

from typing import Literal

from mcp.server.fastmcp import FastMCP  # type: ignore[import-untyped]
from ssbc.core import ssbc_correct

# Initialize FastMCP server
mcp = FastMCP("SSBC Server")


@mcp.tool()
def compute_ssbc_correction(
    alpha_target: float,
    n: int,
    delta: float,
    mode: Literal["beta", "beta-binomial"] = "beta",
) -> dict[str, float | int | str]:
    """Compute Small-Sample Beta Correction for conformal prediction.

    Corrects the miscoverage rate α to provide finite-sample PAC guarantees.
    Unlike asymptotic methods or concentration inequalities (Hoeffding, DKWM),
    SSBC uses the exact induced beta distribution for tighter bounds.

    Parameters
    ----------
    alpha_target : float
        Target miscoverage rate (e.g., 0.10 for 90% coverage target)
    n : int
        Calibration set size (number of calibration points)
    delta : float
        PAC risk parameter (e.g., 0.05 for 95% probability guarantee)
    mode : str, default="beta"
        "beta" for infinite test window, "beta-binomial" for finite test window

    Returns
    -------
    result : dict
        Dictionary containing:
        - alpha_corrected: Corrected miscoverage rate (α')
        - u_star: Optimal threshold index (1-based)
        - pac_mass: Beta distribution mass satisfying guarantee
        - guarantee: Human-readable guarantee statement

    Examples
    --------
    For a calibration set of 100 points targeting 90% coverage with 95% confidence:

    >>> compute_ssbc_correction(alpha_target=0.10, n=100, delta=0.05)
    {
        "alpha_corrected": 0.0571,
        "u_star": 95,
        "pac_mass": 0.9549,
        "guarantee": "With 95.0% probability, coverage ≥ 90.0%"
    }

    Notes
    -----
    Statistical Properties:
    - Distribution-free: No assumptions about P(X,Y)
    - Frequentist: Valid frequentist guarantee (no priors)
    - Finite-sample: Exact for ANY n (not asymptotic)
    - Model-agnostic: Works with any probabilistic classifier

    The corrected α' < α_target provides more conservative thresholds,
    leading to larger prediction sets and higher coverage guarantees.
    """
    # Call the core SSBC function
    result = ssbc_correct(alpha_target=alpha_target, n=n, delta=delta, mode=mode)

    # Format response
    return {
        "alpha_corrected": float(result.alpha_corrected),
        "u_star": int(result.u_star),
        "guarantee": f"With {100 * (1 - delta):.1f}% probability, coverage ≥ {100 * (1 - alpha_target):.1f}%",
        "explanation": (
            f"Use α'={result.alpha_corrected:.4f} instead of α={alpha_target:.4f}. "
            f"This ensures coverage ≥ {100 * (1 - alpha_target):.1f}% with "
            f"{100 * (1 - delta):.1f}% probability over calibration sets of size {n}."
        ),
        "calibration_size": n,
        "target_coverage": f"{100 * (1 - alpha_target):.1f}%",
        "pac_confidence": f"{100 * (1 - delta):.1f}%",
    }


if __name__ == "__main__":
    # Run the MCP server
    mcp.run()
