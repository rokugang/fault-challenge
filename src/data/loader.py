from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .. import config
from ..logging_utils import get_logger
from .schema import MANDATORY_CONTEXT_COLUMNS, NUMERIC_COLUMNS, REFERENCE_BASE_SCHEMA

logger = get_logger(__name__)

LOCALE_DECIMAL = {",": "."}
UNIT_SUFFIXES = {
    " °C": "",
    " %": "",
    " V": "",
    " mbar": "",
    " rpm": "",
    " m": "",
}


@dataclass
class LoadResult:
    raw: pd.DataFrame
    numeric: pd.DataFrame
    metadata: Dict[str, float]


class DataLoader:
    """Load and validate scanner CSV logs."""

    def __init__(self, dropna_threshold: float = 0.99) -> None:
        self.dropna_threshold = dropna_threshold

    def load_reference_file(self, path: Path) -> LoadResult:
        df = self._read_csv(path)
        REFERENCE_BASE_SCHEMA.validate(df, lazy=True)
        df = self._normalize_strings(df)
        numeric = self._convert_to_numeric(df)
        self._validate_context_columns(numeric)
        metadata = self._compute_metadata(numeric, source=str(path))
        return LoadResult(raw=df, numeric=numeric, metadata=metadata)

    def load_fault_example(self) -> LoadResult:
        return self.load_reference_file(config.FAULT_EXAMPLE_FILE)

    def _read_csv(self, path: Path) -> pd.DataFrame:
        logger.info("Loading CSV %s", path)
        try:
            df = pd.read_csv(path, encoding="utf-8", sep=",")
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="latin-1", sep=",")
        return df

    def _normalize_strings(self, df: pd.DataFrame) -> pd.DataFrame:
        cleaned = df.copy()
        for col in cleaned.columns:
            cleaned[col] = (
                cleaned[col]
                .astype(str)
                .str.replace("null", "", case=False)
                .str.translate(str.maketrans(LOCALE_DECIMAL))
                .str.strip()
            )
            for suffix, replacement in UNIT_SUFFIXES.items():
                cleaned[col] = cleaned[col].str.replace(suffix, replacement, regex=False)
        return cleaned.replace({"": np.nan, "N/A": np.nan})

    def _convert_to_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric = df.copy()
        for col in NUMERIC_COLUMNS:
            if col in numeric.columns:
                numeric[col] = pd.to_numeric(numeric[col], errors="coerce")
        return numeric

    def _validate_context_columns(self, df: pd.DataFrame) -> None:
        missing_cols = [col for col in MANDATORY_CONTEXT_COLUMNS if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Mandatory columns missing: {missing_cols}")
        coverage = df[MANDATORY_CONTEXT_COLUMNS].notnull().mean()
        low_coverage = coverage[coverage < self.dropna_threshold]
        if not low_coverage.empty:
            raise ValueError(
                "Context coverage below threshold: "
                + ", ".join(f"{col}={pct:.1%}" for col, pct in low_coverage.items())
            )

    def _compute_metadata(self, df: pd.DataFrame, source: Optional[str] = None) -> Dict[str, float]:
        metadata: Dict[str, float] = {}
        if "Nº de falhas na memória" in df:
            trouble_codes = df["Nº de falhas na memória"].fillna(0)
            metadata["trouble_code_fraction"] = float((trouble_codes != 0).mean())
        if source:
            metadata["source_path"] = source
        return metadata
