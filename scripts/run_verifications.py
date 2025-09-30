from __future__ import annotations

import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config  # noqa: E402
from src.data.loader import DataLoader  # noqa: E402
from src.data.profiling import summarize_coverage  # noqa: E402
from src.data.schema import MANDATORY_CONTEXT_COLUMNS  # noqa: E402
from src.detection.evaluation import evaluate_file  # noqa: E402
from src.pipeline import run_pipeline  # noqa: E402

console = Console()


def verify_reference_profiles(loader: DataLoader) -> None:
    table = Table(title="Reference Coverage Summary")
    table.add_column("File")
    table.add_column("Rows", justify="right")
    table.add_column("Min Coverage %", justify="right")
    table.add_column(
        "Coverage OK",
        justify="right",
    )
    table.add_column("Trouble Code %", justify="right")

    for path in sorted(config.REFERENCES_DIR.glob("*.csv")):
        try:
            load_result = loader.load_reference_file(path)
            coverage_summary = summarize_coverage(load_result.numeric, MANDATORY_CONTEXT_COLUMNS)
            min_cov = min(coverage_summary.coverage.values()) if coverage_summary.coverage else 0.0
            coverage_ok = min_cov >= 99.0
            trouble_fraction = load_result.metadata.get("trouble_code_fraction", 0.0) * 100.0
            table.add_row(
                path.name,
                str(coverage_summary.row_count),
                f"{min_cov:.2f}",
                "PASS" if coverage_ok else "FAIL",
                f"{trouble_fraction:.2f}",
            )
        except ValueError as exc:
            table.add_row(path.name, "-", "0.00", "ERROR", str(exc))

    console.print(table)


def verify_detection(loader: DataLoader) -> None:
    detection_table = Table(title="Detection Outcomes")
    detection_table.add_column("File")
    detection_table.add_column("Fault Detected")
    detection_table.add_column("Rich Ratio", justify="right")
    detection_table.add_column("Low Voltage Ratio", justify="right")
    detection_table.add_column("Min Voltage", justify="right")

    # Reference logs
    for path in sorted(config.REFERENCES_DIR.glob("*.csv")):
        try:
            result = evaluate_file(path, save_features=False)
            detection_table.add_row(
                path.name,
                "YES" if result["fault_detected"] else "NO",
                f"{result['metrics']['rich_idle_ratio']:.3f}",
                f"{result['metrics']['low_voltage_ratio']:.3f}",
                f"{result['metrics']['low_voltage_min']:.2f}",
            )
        except ValueError as exc:
            detection_table.add_row(path.name, "ERROR", "-", "-", str(exc))

    # Fault example
    fault_result = evaluate_file(config.FAULT_EXAMPLE_FILE, save_features=False)
    detection_table.add_row(
        config.FAULT_EXAMPLE_FILE.name,
        "YES" if fault_result["fault_detected"] else "NO",
        f"{fault_result['metrics']['rich_idle_ratio']:.3f}",
        f"{fault_result['metrics']['low_voltage_ratio']:.3f}",
        f"{fault_result['metrics']['low_voltage_min']:.2f}",
    )

    console.print(detection_table)

    df = run_pipeline(config.FAULT_EXAMPLE_FILE, save_features=False)
    rich_count = int((df["rich_idle_score"] >= 2).sum())
    low_voltage_count = int((df["low_battery_score"] >= 1).sum())
    overlap_count = int(((df["rich_idle_score"] >= 2) & (df["low_battery_score"] >= 1)).sum())

    console.print(
        f"Rich-idle frames: {rich_count} | Low-voltage frames: {low_voltage_count} | Overlap: {overlap_count}"
    )


def main() -> None:
    console.rule("Verification Suite")
    loader = DataLoader()
    console.print("[bold]Step 1:[/bold] Profiling reference datasets")
    verify_reference_profiles(loader)

    console.print("[bold]Step 2:[/bold] Detection checks on reference and fault datasets")
    verify_detection(loader)
    console.rule("Completed Verification")


if __name__ == "__main__":
    main()
