"""Console script for ssbc."""

import typer
from rich.console import Console
from rich.table import Table

from ssbc.core import ssbc_correct

app = typer.Typer(
    name="ssbc",
    help="Small-Sample Beta Correction for conformal prediction",
    add_completion=False,
)
console = Console()


@app.command()
def correct(
    alpha_target: float = typer.Option(0.10, "--alpha", "-a", help="Target miscoverage rate"),
    n: int = typer.Option(50, "--n", help="Calibration set size"),
    delta: float = typer.Option(0.10, "--delta", "-d", help="PAC risk tolerance"),
    mode: str = typer.Option("beta", "--mode", "-m", help="Mode: 'beta' or 'beta-binomial'"),
):
    """Compute SSBC correction for conformal prediction."""
    try:
        # Validate mode parameter
        if mode not in ("beta", "beta-binomial"):
            raise ValueError("mode must be 'beta' or 'beta-binomial'")

        result = ssbc_correct(
            alpha_target=alpha_target,
            n=n,
            delta=delta,
            mode=mode,  # type: ignore[arg-type]
        )

        # Create a nice table for the results
        table = Table(title="SSBC Correction Results")
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Target miscoverage (α)", f"{alpha_target:.4f}")
        table.add_row("Corrected miscoverage (α')", f"{result.alpha_corrected:.4f}")
        table.add_row("Optimal threshold (u*)", str(result.u_star))
        table.add_row("Calibration size (n)", str(result.n))
        table.add_row("PAC confidence", f"{100 * (1 - delta):.1f}%")
        table.add_row("Satisfied mass", f"{result.satisfied_mass:.4f}")
        table.add_row("Mode", result.mode)

        console.print(table)

        # Print the guarantee
        console.print(
            f"\n[bold green]Guarantee:[/bold green] With {100 * (1 - delta):.1f}% probability, "
            f"coverage ≥ {100 * (1 - alpha_target):.1f}%"
        )

    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from e


@app.command()
def version():
    """Show version information."""
    from . import __version__

    console.print(f"SSBC version: {__version__}")


if __name__ == "__main__":
    app()
