from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

from .. import config


@dataclass
class CoverageSummary:
    coverage: Dict[str, float]
    missing_counts: Dict[str, int]
    row_count: int

    def to_dict(self) -> Dict[str, object]:
        return {
            "row_count": self.row_count,
            "coverage_pct": self.coverage,
            "missing_counts": self.missing_counts,
        }


@dataclass
class NumericSummary:
    stats: Dict[str, Dict[str, float]]

    def to_dict(self) -> Dict[str, Dict[str, float]]:
        return self.stats


def summarize_coverage(df: pd.DataFrame, columns: Iterable[str]) -> CoverageSummary:
    cols = list(columns)
    available = [col for col in cols if col in df.columns]
    coverage = df[available].notnull().mean()
    missing_counts = df[available].isnull().sum()
    return CoverageSummary(
        coverage={col: float(coverage[col] * 100.0) for col in available},
        missing_counts={col: int(missing_counts[col]) for col in available},
        row_count=len(df),
    )


def summarize_numeric(df: pd.DataFrame, columns: Optional[Iterable[str]] = None) -> NumericSummary:
    if columns is None:
        numeric_df = df.select_dtypes(include=[np.number])
    else:
        numeric_df = df[list(columns)]
    stats: Dict[str, Dict[str, float]] = {}
    for col in numeric_df.columns:
        series = numeric_df[col].dropna()
        if series.empty:
            continue
        stats[col] = {
            "mean": float(series.mean()),
            "std": float(series.std(ddof=0)),
            "min": float(series.min()),
            "p05": float(series.quantile(0.05)),
            "p50": float(series.quantile(0.50)),
            "p95": float(series.quantile(0.95)),
            "max": float(series.max()),
        }
    return NumericSummary(stats)


def persist_summary(payload: Dict[str, object], prefix: str) -> Path:
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_dir = config.ARTIFACTS_DIR / "profiling"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{prefix}_{timestamp}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
