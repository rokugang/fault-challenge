from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

RICH_IDLE_THRESHOLD_RATIO = 0.05  # ≥5% of frames with rich-idle pattern
LOW_VOLTAGE_MIN_THRESHOLD = 12.0  # volts
LOW_VOLTAGE_RATIO_THRESHOLD = 0.05


@dataclass
class DetectionResult:
    fault_detected: bool
    reasons: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)


class FaultDetector:
    """Rule-based detector for rich idle mixture plus low battery voltage."""

    def evaluate(self, df: pd.DataFrame) -> DetectionResult:
        self._check_inputs(df)
        metrics: Dict[str, float] = {}
        reasons: List[str] = []

        rich_condition = self._rich_idle_condition(df)
        rich_ratio = float(rich_condition.mean()) if len(df) else 0.0
        rich_count = int(rich_condition.sum())
        metrics["rich_idle_ratio"] = rich_ratio
        metrics["rich_idle_count"] = rich_count

        # Check voltage if column exists
        if "Tensão do módulo" in df.columns:
            low_voltage_condition = self._low_voltage_condition(df)
            low_voltage_ratio = float(low_voltage_condition.mean()) if len(df) else 0.0
            low_voltage_min = float(df["Tensão do módulo"].min())
            voltage_ok = (
                low_voltage_min <= LOW_VOLTAGE_MIN_THRESHOLD
                or low_voltage_ratio >= LOW_VOLTAGE_RATIO_THRESHOLD
            )
        else:
            low_voltage_ratio = 0.0
            low_voltage_min = np.nan
            voltage_ok = False
        
        metrics["low_voltage_ratio"] = low_voltage_ratio
        metrics["low_voltage_min"] = low_voltage_min

        rich_ok = rich_ratio >= RICH_IDLE_THRESHOLD_RATIO

        if rich_ok:
            reasons.append(
                "Rich mixture at idle detected: >=2 signals present in %.1f%% of frames" % (rich_ratio * 100.0)
            )
        if voltage_ok:
            reasons.append(
                "Low battery voltage observed: minimum %.2f V, %.1f%% of frames below threshold"
                % (low_voltage_min, low_voltage_ratio * 100.0)
            )

        fault_detected = rich_ok and voltage_ok
        if fault_detected:
            reasons.insert(0, "Fault: rich air-fuel mixture at idle plus low battery voltage")
        else:
            reasons.insert(0, "No combined rich-idle + low-voltage fault detected")

        return DetectionResult(fault_detected=fault_detected, reasons=reasons, metrics=metrics)

    def _check_inputs(self, df: pd.DataFrame) -> None:
        if "rich_idle_score" not in df.columns:
            raise ValueError("Feature column 'rich_idle_score' missing; run feature engineering first")

    def _rich_idle_condition(self, df: pd.DataFrame) -> pd.Series:
        return df["rich_idle_score"].fillna(0) >= 2

    def _low_voltage_condition(self, df: pd.DataFrame) -> pd.Series:
        voltage = df["Tensão do módulo"].astype(float)
        return voltage <= LOW_VOLTAGE_MIN_THRESHOLD
    
    def run_detection(self, df: pd.DataFrame) -> DetectionResult:
        """Alias for evaluate() for backward compatibility with tests."""
        return self.evaluate(df)
