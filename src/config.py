from __future__ import annotations

from pathlib import Path

# Root directory for project assets
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

DATA_ROOT: Path = PROJECT_ROOT / "datasets"
REFERENCES_DIR: Path = DATA_ROOT / "references"
FAULT_EXAMPLE_FILE: Path = DATA_ROOT / "fault_example.csv"

ARTIFACTS_DIR: Path = PROJECT_ROOT / "artifacts"
LOGS_DIR: Path = PROJECT_ROOT / "logs"
DOCS_DIR: Path = PROJECT_ROOT / "docs"

# Create directories lazily when modules import config
for _dir in (ARTIFACTS_DIR, LOGS_DIR, DOCS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)
