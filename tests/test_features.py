from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.features.engineering import FeatureEngineer


def test_feature_engineering_generates_scores(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "Ajuste de combustível de curto prazo - Banco 1": [-12, -5, 0, -9],
            "Sonda lambda - Banco 1, sensor 1": [0.85, 0.82, 0.75, 0.83],
            "Rotação do motor - RPM": [900, 1200, 800, 950],
            "Tensão do módulo": [11.8, 12.5, 11.2, 11.9],
            "Pressão no coletor de admissão - MAP": [320, 300, 310, 305],
            "Carga calculada do motor": [25, 30, 20, 27],
        }
    )

    engineer = FeatureEngineer()
    enriched = engineer.transform(df)

    assert "rich_idle_score" in enriched.columns
    assert "low_battery_score" in enriched.columns
    assert enriched.loc[0, "rich_idle_score"] >= 2
    assert enriched["low_battery_score"].max() >= 1
