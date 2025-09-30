from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..pipeline import run_pipeline
from .detector import FaultDetector


def evaluate_file(path: Path, save_features: bool = True) -> dict:
    df = run_pipeline(path, save_features=save_features)
    detector = FaultDetector()
    result = detector.evaluate(df)
    return {
        "source": str(path),
        "fault_detected": result.fault_detected,
        "reasons": result.reasons,
        "metrics": result.metrics,
    }
