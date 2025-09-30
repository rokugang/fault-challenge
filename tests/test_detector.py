from __future__ import annotations

import pandas as pd

from src.detection.detector import FaultDetector


def test_fault_detector_flags_combined_conditions() -> None:
    df = pd.DataFrame(
        {
            "rich_idle_score": [3, 0, 2, 1],
            "Tensão do módulo": [11.5, 12.4, 11.8, 12.0],
        }
    )
    detector = FaultDetector()
    result = detector.evaluate(df)
    assert result.fault_detected
    assert any("Fault:" in reason for reason in result.reasons)


def test_fault_detector_requires_features() -> None:
    df = pd.DataFrame({"Tensão do módulo": [12.5, 12.4, 12.2]})
    detector = FaultDetector()
    result = detector.evaluate(df.assign(rich_idle_score=[0, 0, 0]))
    assert not result.fault_detected
    assert "Fault" not in result.reasons[0]
