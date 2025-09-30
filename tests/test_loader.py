from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data.loader import DataLoader
from src.data.schema import MANDATORY_CONTEXT_COLUMNS


def test_load_reference_file(tmp_path: Path) -> None:
    csv_content = """Temperatura do líquido de arrefecimento do motor - CTS,Carga calculada do motor,Rotação do motor - RPM,Altitude,Nº de falhas na memória,Sonda lambda - Banco 1, sensor 1,Tensão do módulo
90,30,800,10,0,0.45,13.5
"""
    csv_path = tmp_path / "reference.csv"
    csv_path.write_text(csv_content, encoding="utf-8")

    loader = DataLoader()
    result = loader.load_reference_file(csv_path)

    assert not result.numeric.empty
    assert set(MANDATORY_CONTEXT_COLUMNS).issubset(result.numeric.columns)
    assert result.metadata["trouble_code_fraction"] == 0.0


def test_load_reference_file_with_missing_context(tmp_path: Path) -> None:
    csv_content = """Temperatura do líquido de arrefecimento do motor - CTS,Nº de falhas na memória
90,0
"""
    csv_path = tmp_path / "bad_reference.csv"
    csv_path.write_text(csv_content, encoding="utf-8")

    loader = DataLoader()
    with pytest.raises(ValueError):
        loader.load_reference_file(csv_path)
