"""Example: Core SSBC Algorithm.

This example demonstrates the Small-Sample Beta Correction algorithm
for different calibration set sizes.
"""

import numpy as np
from ssbc import ssbc_correct


def main():
    """Run SSBC core algorithm examples."""
    
    print("=" * 80)
    print("SMALL-SAMPLE BETA CORRECTION (SSBC) - CORE ALGORITHM")
    print("=" * 80)
    
    # ========== Example 1: Small Calibration Set ==========
    print("\n" + "=" * 60)
    print("Example 1: Small calibration set (n=50)")
    print("=" * 60)
    
    result_small = ssbc_correct(
        alpha_target=0.10,
        n=50,
        delta=0.10,
        mode="beta"
    )
    
    print(f"\nInput Parameters:")
    print(f"  α_target: {result_small.alpha_target}")
    print(f"  n:        {result_small.n}")
    print(f"  δ:        {result_small.details['delta']}")
    print(f"  mode:     {result_small.mode}")
    
    print(f"\nResults:")
    print(f"  α_corrected:   {result_small.alpha_corrected:.6f}")
    print(f"  u*:            {result_small.u_star}")
    print(f"  u_guess:       {result_small.details['u_star_guess']}")
    print(f"  search_range:  {result_small.details['search_range']}")
    print(f"  satisfied_mass: {result_small.satisfied_mass:.6f}")
    
    print(f"\nSearch Log (u values tested):")
    for entry in result_small.details['search_log']:
        status = "✓ PASS" if entry['passes'] else "✗ FAIL"
        print(f"  u={entry['u']:2d}: α'={entry['alpha_prime']:.4f}, "
              f"Beta({entry['a']:2d},{entry['b']:2d}), "
              f"P(C≥0.9)={entry['ptail']:.4f} {status}")
    
    # ========== Example 2: Large Calibration Set ==========
    print("\n" + "=" * 60)
    print("Example 2: Large calibration set (n=5000)")
    print("=" * 60)
    
    result_large = ssbc_correct(
        alpha_target=0.10,
        n=5000,
        delta=0.10,
        mode="beta",
        bracket_width=15  # Narrow search for efficiency
    )
    
    print(f"\nInput Parameters:")
    print(f"  α_target: {result_large.alpha_target}")
    print(f"  n:        {result_large.n}")
    print(f"  δ:        {result_large.details['delta']}")
    print(f"  bracket_width: {result_large.details['bracket_width']}")
    
    print(f"\nResults:")
    print(f"  α_corrected:   {result_large.alpha_corrected:.6f}")
    print(f"  u*:            {result_large.u_star}")
    print(f"  u_guess:       {result_large.details['u_star_guess']}")
    print(f"  search_range:  {result_large.details['search_range']}")
    print(f"  satisfied_mass: {result_large.satisfied_mass:.6f}")
    print(f"  searched {len(result_large.details['search_log'])} values "
          f"(instead of {result_large.n})")
    
    print(f"\nSearch Log (first 5 and last 5):")
    log = result_large.details['search_log']
    for entry in log[:5]:
        status = "✓ PASS" if entry['passes'] else "✗ FAIL"
        print(f"  u={entry['u']:4d}: α'={entry['alpha_prime']:.6f}, "
              f"Beta({entry['a']:4d},{entry['b']:4d}), "
              f"P(C≥0.9)={entry['ptail']:.4f} {status}")
    if len(log) > 10:
        print("  ...")
    for entry in log[-5:]:
        status = "✓ PASS" if entry['passes'] else "✗ FAIL"
        print(f"  u={entry['u']:4d}: α'={entry['alpha_prime']:.6f}, "
              f"Beta({entry['a']:4d},{entry['b']:4d}), "
              f"P(C≥0.9)={entry['ptail']:.4f} {status}")
    
    # ========== Example 3: Comparison Across Different n ==========
    print("\n" + "=" * 60)
    print("Example 3: Effect of calibration set size")
    print("=" * 60)
    
    n_values = [10, 50, 100, 500, 1000, 5000]
    alpha_target = 0.10
    delta = 0.10
    
    print(f"\nFixed: α_target={alpha_target}, δ={delta}")
    print(f"\n{'n':>6} {'α_corrected':>12} {'u*':>6} {'Correction':>12}")
    print("-" * 40)
    
    for n in n_values:
        result = ssbc_correct(alpha_target=alpha_target, n=n, delta=delta, mode="beta")
        correction_pct = (alpha_target - result.alpha_corrected) / alpha_target * 100
        print(f"{n:6d} {result.alpha_corrected:12.6f} {result.u_star:6d} "
              f"{correction_pct:11.2f}%")
    
    print("\nObservation: As n increases, α_corrected approaches α_target")
    print("             (less correction needed for larger calibration sets)")
    
    # ========== Example 4: Beta-Binomial Mode ==========
    print("\n" + "=" * 60)
    print("Example 4: Beta-Binomial mode (finite test window)")
    print("=" * 60)
    
    result_bb = ssbc_correct(
        alpha_target=0.10,
        n=50,
        delta=0.10,
        mode="beta-binomial",
        m=100  # Test window size
    )
    
    print(f"\nInput Parameters:")
    print(f"  α_target: {result_bb.alpha_target}")
    print(f"  n:        {result_bb.n} (calibration set)")
    print(f"  m:        {result_bb.details['m']} (test window)")
    print(f"  δ:        {result_bb.details['delta']}")
    print(f"  mode:     {result_bb.mode}")
    
    print(f"\nResults:")
    print(f"  α_corrected:   {result_bb.alpha_corrected:.6f}")
    print(f"  u*:            {result_bb.u_star}")
    
    print("\nComparison:")
    print(f"  Beta mode:           α_corrected = {result_small.alpha_corrected:.6f}")
    print(f"  Beta-Binomial mode:  α_corrected = {result_bb.alpha_corrected:.6f}")
    print("\n  Note: Beta-binomial accounts for finite test window,")
    print("        typically requires slightly more conservative correction.")
    
    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()

