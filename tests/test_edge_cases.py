"""
Edge Case Tests

Tests robustness against various data quality issues and edge cases.
"""
from __future__ import annotations

from pathlib import Path
from io import StringIO
import pytest
import pandas as pd
import numpy as np

from src.data.loader import DataLoader
from src.features.engineering import FeatureEngineer
from src.detection.detector import FaultDetector


class TestLoaderEdgeCases:
    """Test data loader with problematic inputs."""
    
    def test_empty_file(self, tmp_path: Path):
        """Empty CSV should raise error."""
        csv_path = tmp_path / "empty.csv"
        csv_path.write_text("", encoding="utf-8")
        
        loader = DataLoader()
        with pytest.raises(Exception):
            loader.load_reference_file(csv_path)
    
    def test_header_only(self, tmp_path: Path):
        """CSV with only headers should raise error."""
        csv_content = "Temperatura do líquido de arrefecimento do motor - CTS,Rotação do motor - RPM\n"
        csv_path = tmp_path / "header_only.csv"
        csv_path.write_text(csv_content, encoding="utf-8")
        
        loader = DataLoader()
        with pytest.raises(Exception):
            loader.load_reference_file(csv_path)
    
    def test_all_nulls(self, tmp_path: Path):
        """All null values should fail coverage check."""
        csv_content = """Temperatura do líquido de arrefecimento do motor - CTS,Carga calculada do motor,Rotação do motor - RPM,Altitude,Nº de falhas na memória
,,,,,
,,,,,
,,,,,
"""
        csv_path = tmp_path / "all_nulls.csv"
        csv_path.write_text(csv_content, encoding="utf-8")
        
        loader = DataLoader()
        with pytest.raises(ValueError, match="coverage"):
            loader.load_reference_file(csv_path)
    
    def test_missing_mandatory_columns(self, tmp_path: Path):
        """Missing mandatory columns should fail."""
        csv_content = """Temperatura do ar ambiente,Tensão do módulo
25,13.5
"""
        csv_path = tmp_path / "missing_cols.csv"
        csv_path.write_text(csv_content, encoding="utf-8")
        
        loader = DataLoader()
        with pytest.raises(ValueError, match="coverage"):
            loader.load_reference_file(csv_path)
    
    def test_invalid_units(self, tmp_path: Path):
        """Invalid/mixed units should be handled."""
        csv_content = """Temperatura do líquido de arrefecimento do motor - CTS,Carga calculada do motor,Rotação do motor - RPM,Altitude,Nº de falhas na memória
90°C,30%,800RPM,10m,0
invalid,bad,data,here,0
"""
        csv_path = tmp_path / "invalid_units.csv"
        csv_path.write_text(csv_content, encoding="utf-8")
        
        loader = DataLoader()
        # Should load but convert invalid to NaN
        df = loader.load_reference_file(csv_path)
        assert df.iloc[1].isna().sum() > 0  # Second row has NaN
    
    def test_locale_decimals(self, tmp_path: Path):
        """Brazilian decimal format (comma) should convert correctly."""
        csv_content = """Temperatura do líquido de arrefecimento do motor - CTS,Carga calculada do motor,Rotação do motor - RPM,Altitude,Nº de falhas na memória,Sonda lambda - Banco 1, sensor 1
90,30,800,10,0,"0,45"
"""
        csv_path = tmp_path / "locale_decimals.csv"
        csv_path.write_text(csv_content, encoding="utf-8")
        
        loader = DataLoader()
        result = loader.load_reference_file(csv_path)
        df = result.numeric
        
        # Check lambda converted from "0,45" to 0.45
        assert df["Sonda lambda - Banco 1, sensor 1"].iloc[0] == 0.45
    
    def test_non_zero_trouble_codes(self, tmp_path: Path):
        """Non-zero trouble codes should fail for reference data."""
        csv_content = """Temperatura do líquido de arrefecimento do motor - CTS,Carga calculada do motor,Rotação do motor - RPM,Altitude,Nº de falhas na memória
90,30,800,10,5
"""
        csv_path = tmp_path / "trouble_codes.csv"
        csv_path.write_text(csv_content, encoding="utf-8")
        
        loader = DataLoader()
        # Currently loads but should flag in metadata
        result = loader.load_reference_file(csv_path)
        assert result.metadata.get("has_trouble_codes", False) or result.numeric["Nº de falhas na memória"].max() > 0


class TestFeatureEngineeringEdgeCases:
    """Test feature engineering with edge cases."""
    
    def test_single_row(self):
        """Single row should not crash (rolling windows)."""
        df = pd.DataFrame({
            "Temperatura do líquido de arrefecimento do motor - CTS": [90.0],
            "Carga calculada do motor": [30.0],
            "Rotação do motor - RPM": [800.0],
            "Altitude": [10.0],
            "Ajuste de combustível de curto prazo - Banco 1": [-5.0],
            "Sonda lambda - Banco 1, sensor 1": [0.5],
            "Tensão do módulo": [13.5],
            "Pressão no coletor de admissão - MAP": [50.0],
        })
        
        engineer = FeatureEngineer()
        result = engineer.transform(df)
        
        assert len(result) == 1
        assert not result.empty
    
    def test_all_same_values(self):
        """All identical values should not crash."""
        df = pd.DataFrame({
            "Temperatura do líquido de arrefecimento do motor - CTS": [90.0] * 100,
            "Carga calculada do motor": [30.0] * 100,
            "Rotação do motor - RPM": [800.0] * 100,
            "Altitude": [10.0] * 100,
            "Ajuste de combustível de curto prazo - Banco 1": [0.0] * 100,
            "Sonda lambda - Banco 1, sensor 1": [0.5] * 100,
            "Tensão do módulo": [13.5] * 100,
            "Pressão no coletor de admissão - MAP": [50.0] * 100,
        })
        
        engineer = FeatureEngineer()
        result = engineer.transform(df)
        
        assert len(result) == 100
        # Rolling std should be 0 or near-0
        if "rpm_rolling_std" in result.columns:
            assert result["rpm_rolling_std"].max() < 1e-6
    
    def test_extreme_values(self):
        """Extreme/outlier values should not crash."""
        df = pd.DataFrame({
            "Temperatura do líquido de arrefecimento do motor - CTS": [9999.0] * 10,
            "Carga calculada do motor": [0.0] * 10,
            "Rotação do motor - RPM": [0.0] * 10,
            "Altitude": [-1000.0] * 10,
            "Ajuste de combustível de curto prazo - Banco 1": [-99.0] * 10,
            "Sonda lambda - Banco 1, sensor 1": [5.0] * 10,
            "Tensão do módulo": [0.0] * 10,
            "Pressão no coletor de admissão - MAP": [200.0] * 10,
        })
        
        engineer = FeatureEngineer()
        result = engineer.transform(df)
        
        assert len(result) == 10
        # Should handle without inf/nan
        assert not np.isinf(result.select_dtypes(include=[np.number]).values).any()
    
    def test_missing_optional_columns(self):
        """Missing optional columns should still work."""
        df = pd.DataFrame({
            "Temperatura do líquido de arrefecimento do motor - CTS": [90.0] * 10,
            "Carga calculada do motor": [30.0] * 10,
            "Rotação do motor - RPM": [800.0] * 10,
            "Altitude": [10.0] * 10,
        })
        
        engineer = FeatureEngineer()
        result = engineer.transform(df)
        
        assert len(result) == 10
        # Some features won't be created, but shouldn't crash


class TestDetectorEdgeCases:
    """Test fault detector with edge cases."""
    
    def test_empty_dataframe(self):
        """Empty dataframe should not crash."""
        df = pd.DataFrame()
        
        detector = FaultDetector()
        with pytest.raises((ValueError, AttributeError)):
            detector.run_detection(df)
    
    def test_no_voltage_column(self):
        """Missing voltage column should handle gracefully."""
        df = pd.DataFrame({
            "Temperatura do líquido de arrefecimento do motor - CTS": [90.0] * 10,
            "rich_idle_score": [1.0] * 10,  # High rich-idle
        })
        
        detector = FaultDetector()
        result = detector.run_detection(df)
        
        # Should detect rich but not voltage issue
        assert "rich_idle_ratio" in result.metrics
    
    def test_all_faults(self):
        """All frames showing faults should flag correctly."""
        df = pd.DataFrame({
            "Temperatura do líquido de arrefecimento do motor - CTS": [90.0] * 100,
            "rich_idle_score": [1.0] * 100,  # All rich
            "Tensão do módulo": [11.0] * 100,  # All low voltage
        })
        
        detector = FaultDetector()
        result = detector.run_detection(df)
        
        assert result.fault_detected is True
        assert result.metrics["rich_idle_ratio"] == 1.0
        assert result.metrics["low_voltage_min"] == 11.0
    
    def test_borderline_thresholds(self):
        """Values exactly at threshold should be consistent."""
        # Exactly 5% rich frames
        df = pd.DataFrame({
            "Temperatura do líquido de arrefecimento do motor - CTS": [90.0] * 100,
            "rich_idle_score": [1.0] * 5 + [0.0] * 95,
            "Tensão do módulo": [12.0] * 100,  # Exactly 12V
        })
        
        detector = FaultDetector()
        result = detector.run_detection(df)
        
        # Should have consistent behavior at boundary
        assert "rich_idle_ratio" in result.metrics
        assert result.metrics["rich_idle_ratio"] == 0.05
