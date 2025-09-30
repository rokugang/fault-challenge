from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .data.loader import DataLoader
from .data.profiling import CoverageSummary, NumericSummary, persist_summary, summarize_coverage, summarize_numeric
from .data.schema import MANDATORY_CONTEXT_COLUMNS
from .detection.evaluation import evaluate_file

app = typer.Typer(add_completion=False, help="Diagnostics ETL utilities")
console = Console()


def _load_file(path: Path) -> tuple[CoverageSummary, NumericSummary]:
    loader = DataLoader()
    result = loader.load_reference_file(path)
    coverage = summarize_coverage(result.numeric, MANDATORY_CONTEXT_COLUMNS)
    numeric_summary = summarize_numeric(result.numeric.select_dtypes(include="number"))
    payload = {
        "source_path": str(path),
        "metadata": result.metadata,
        "coverage": coverage.to_dict(),
        "numeric_summary": numeric_summary.to_dict(),
    }
    output_path = persist_summary(payload, prefix=path.stem)
    console.print(f"Saved profiling summary to [bold]{output_path}[/bold]")
    return coverage, numeric_summary


@app.command()
def profile_reference(file: Path = typer.Argument(..., exists=True, dir_okay=False)) -> None:
    """Profile a reference CSV file and output coverage + summary statistics."""
    coverage, numeric_summary = _load_file(file)
    table = Table(title="Context Coverage")
    table.add_column("Column")
    table.add_column("Coverage %", justify="right")
    table.add_column("Missing", justify="right")
    for col, pct in coverage.coverage.items():
        missing = coverage.missing_counts.get(col, 0)
        table.add_row(col, f"{pct:.2f}", str(missing))
    console.print(table)

    stats_table = Table(title="Numeric Summary (subset)")
    stats_table.add_column("Column")
    stats_table.add_column("Mean", justify="right")
    stats_table.add_column("Std", justify="right")
    stats_table.add_column("p05", justify="right")
    stats_table.add_column("p50", justify="right")
    stats_table.add_column("p95", justify="right")
    for col, stat in list(numeric_summary.stats.items())[:10]:
        stats_table.add_row(
            col,
            f"{stat['mean']:.3f}",
            f"{stat['std']:.3f}",
            f"{stat['p05']:.3f}",
            f"{stat['p50']:.3f}",
            f"{stat['p95']:.3f}",
        )
    console.print(stats_table)


@app.command()
def profile_fault_example() -> None:
    """Profile the provided induced-fault dataset."""
    loader = DataLoader()
    result = loader.load_fault_example()
    coverage = summarize_coverage(result.numeric, MANDATORY_CONTEXT_COLUMNS)
    numeric_summary = summarize_numeric(result.numeric.select_dtypes(include="number"))
    payload = {
        "source_path": str(Path("fault_example.csv")),
        "metadata": result.metadata,
        "coverage": coverage.to_dict(),
        "numeric_summary": numeric_summary.to_dict(),
    }
    output_path = persist_summary(payload, prefix="fault_example")
    console.print(f"Saved profiling summary to [bold]{output_path}[/bold]")
    console.print(json.dumps(payload, indent=2, ensure_ascii=False))


@app.command()
def detect(file: Path = typer.Argument(..., exists=True, dir_okay=False)) -> None:
    """Run full pipeline and output fault detection decision."""
    result = evaluate_file(file)
    console.print_json(data=result)


@app.command()
def serve(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Launch FastAPI service for external integrations."""
    import uvicorn

    uvicorn.run("src.api:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    app()
