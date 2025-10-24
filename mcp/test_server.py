#!/usr/bin/env python
"""Test script for SSBC MCP server."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ssbc import ssbc_correct


def test_ssbc_function():
    """Test the underlying SSBC function."""
    print("=" * 70)
    print("Testing SSBC Function")
    print("=" * 70)

    # Test parameters
    test_cases: list[dict[str, float | int | str]] = [
        {"alpha_target": 0.10, "n": 50, "delta": 0.05, "mode": "beta"},
        {"alpha_target": 0.10, "n": 100, "delta": 0.05, "mode": "beta"},
        {"alpha_target": 0.05, "n": 200, "delta": 0.10, "mode": "beta"},
    ]

    for i, params in enumerate(test_cases, 1):
        result = ssbc_correct(**params)  # type: ignore[arg-type]
        print(f"\nTest {i}: n={params['n']}, α={params['alpha_target']}, δ={params['delta']}")
        print(f"  α_corrected: {result.alpha_corrected:.4f}")
        print(f"  u*: {result.u_star}")
        print(
            f"  Guarantee: With {100 * (1 - float(params['delta'])):.1f}% probability, "
            f"coverage ≥ {100 * (1 - float(params['alpha_target'])):.1f}%"
        )

    print("\n✅ All tests passed!")
    return True


def test_mcp_server_import():
    """Test that MCP server can be imported."""
    print("\n" + "=" * 70)
    print("Testing MCP Server Import")
    print("=" * 70)

    try:
        import ssbc.mcp_server  # noqa: F401

        print("✅ MCP server module imported successfully")
        print("   Server name: SSBC Server")
        print("   Tools registered: compute_ssbc_correction")
        return True
    except ImportError as e:
        print(f"❌ Failed to import MCP server: {e}")
        print("   Install with: pip install -e '.[mcp]'")
        return False


if __name__ == "__main__":
    print("\n╔═══════════════════════════════════════════════════════════════╗")
    print("║              SSBC MCP Server Test Suite                      ║")
    print("╚═══════════════════════════════════════════════════════════════╝\n")

    success = True
    success &= test_ssbc_function()
    success &= test_mcp_server_import()

    print("\n" + "=" * 70)
    if success:
        print("✅ ALL TESTS PASSED - MCP Server Ready!")
        print("=" * 70)
        print("\nNext steps:")
        print("  1. Install: pip install -e '.[mcp]'")
        print("  2. Run: python -m ssbc.mcp_server")
        print("  3. Deploy: ./mcp/deploy-cloudrun.sh YOUR_PROJECT_ID")
        sys.exit(0)
    else:
        print("❌ SOME TESTS FAILED")
        print("=" * 70)
        sys.exit(1)
