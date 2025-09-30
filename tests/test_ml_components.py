"""
Tests for ML components (ensemble, production utils, temporal).

Author: Rohit Gangupantulu
"""
from __future__ import annotations

from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from src.ml.ensemble_detector import EnsembleDetector
from src.ml.production_utils import ProductionWrapper, HealthCheck
from src.ml.temporal_detector import TemporalFaultDetector
from src.detection.detector import FaultDetector


class TestEnsembleDetector:
    """Test ensemble voting detector."""
    
    def test_initialization_no_models(self):
        """Ensemble should initialize without models."""
        detector = EnsembleDetector()
        assert detector.models == {}
        assert detector.weights == {}
    
    def test_initialization_with_models(self, tmp_path):
        """Ensemble should load models if directory exists."""
        models_dir = tmp_path / "ml_models"
        models_dir.mkdir()
        
        # Would need actual model files for full test
        detector = EnsembleDetector(models_dir=models_dir)
        assert detector.models is not None
    
    def test_feature_preparation(self):
        """Should extract and prepare features correctly."""
        df = pd.DataFrame({
            "feature1": [1.0, 2.0, 3.0],
            "feature2": [4.0, 5.0, 6.0],
            "timestamp": ["2025-01-01", "2025-01-02", "2025-01-03"]
        })
        
        detector = EnsembleDetector()
        X, feature_names = detector._prepare_features(df)
        
        assert X.shape == (3, 2)
        assert "timestamp" not in feature_names
        assert "feature1" in feature_names
        assert "feature2" in feature_names
    
    def test_ensemble_predict_no_models(self):
        """Should handle missing models gracefully."""
        detector = EnsembleDetector()
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        
        ratio, scores = detector._ensemble_predict(X)
        assert ratio == 0.0
        assert scores == {}
    
    def test_explain_anomalies(self):
        """Should identify most anomalous features."""
        df = pd.DataFrame({
            "normal_feature": [1.0, 1.1, 1.2],
            "anomalous_feature": [1.0, 1.0, 100.0],  # Spike in last row
        })
        
        detector = EnsembleDetector()
        X, _ = detector._prepare_features(df)
        detector.feature_names = ["normal_feature", "anomalous_feature"]
        
        top_features = detector._explain_anomalies(df, X)
        
        assert len(top_features) <= 5
        # Anomalous feature should rank higher
        assert top_features[0][0] == "anomalous_feature"


class TestProductionWrapper:
    """Test production utilities."""
    
    def test_initialization(self):
        """Should initialize with monitoring metrics."""
        detector = FaultDetector()
        wrapper = ProductionWrapper(detector)
        
        assert wrapper.predictions_count == 0
        assert wrapper.error_count == 0
        assert wrapper.circuit_open is False
    
    def test_input_validation_empty(self):
        """Should reject empty dataframe."""
        detector = FaultDetector()
        wrapper = ProductionWrapper(detector)
        
        df = pd.DataFrame()
        valid, msg = wrapper._validate_input(df)
        
        assert valid is False
        assert "empty" in msg.lower()
    
    def test_input_validation_insufficient_data(self):
        """Should reject too few rows."""
        detector = FaultDetector()
        wrapper = ProductionWrapper(detector)
        
        df = pd.DataFrame({"col1": [1, 2]})
        valid, msg = wrapper._validate_input(df)
        
        assert valid is False
        assert "insufficient" in msg.lower()
    
    def test_input_validation_all_nan(self):
        """Should reject all-NaN dataframe."""
        detector = FaultDetector()
        wrapper = ProductionWrapper(detector)
        
        df = pd.DataFrame({"col1": [np.nan] * 10})
        valid, msg = wrapper._validate_input(df)
        
        assert valid is False
        assert "nan" in msg.lower()
    
    def test_input_validation_success(self):
        """Should accept valid dataframe."""
        detector = FaultDetector()
        wrapper = ProductionWrapper(detector)
        
        df = pd.DataFrame({"col1": [1.0, 2.0, 3.0, 4.0]})
        valid, msg = wrapper._validate_input(df)
        
        assert valid is True
        assert msg is None
    
    def test_circuit_breaker_logic(self):
        """Should open circuit breaker at high error rate."""
        detector = FaultDetector()
        wrapper = ProductionWrapper(detector)
        
        # Simulate 10 predictions with 6 errors
        wrapper.predictions_count = 10
        wrapper.error_count = 6
        
        assert wrapper._should_use_fallback() is True
    
    def test_health_check(self):
        """Should return health metrics."""
        detector = FaultDetector()
        wrapper = ProductionWrapper(detector)
        
        wrapper.predictions_count = 5
        wrapper.error_count = 1
        
        health = wrapper.health_check()
        
        assert isinstance(health, HealthCheck)
        assert health.predictions_count == 5
        assert health.error_count == 1
        assert health.model_loaded is True
    
    def test_circuit_breaker_reset(self):
        """Should reset circuit breaker."""
        detector = FaultDetector()
        wrapper = ProductionWrapper(detector)
        
        wrapper.predictions_count = 10
        wrapper.error_count = 6
        wrapper.circuit_open = True
        
        wrapper.reset_circuit_breaker()
        
        assert wrapper.predictions_count == 0
        assert wrapper.error_count == 0
        assert wrapper.circuit_open is False


class TestTemporalDetector:
    """Test temporal windowing detector."""
    
    def test_initialization(self):
        """Should initialize with window parameters."""
        detector = TemporalFaultDetector(
            window_size_sec=30.0,
            overlap_ratio=0.5,
            min_windows_fault=2
        )
        
        assert detector.window_size_sec == 30.0
        assert detector.overlap_ratio == 0.5
        assert detector.min_windows_fault == 2
    
    def test_sampling_rate_estimation(self):
        """Should estimate sampling rate from timestamps."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01", periods=10, freq="1s")
        })
        
        detector = TemporalFaultDetector()
        rate = detector._estimate_sampling_rate(df)
        
        assert rate == pytest.approx(1.0, rel=0.1)  # 1 Hz
    
    def test_window_creation(self):
        """Should create overlapping windows."""
        df = pd.DataFrame({"col1": range(100)})
        
        detector = TemporalFaultDetector(
            window_size_sec=10.0,
            overlap_ratio=0.5
        )
        
        windows = detector._create_windows(df, sampling_rate=1.0)
        
        # Should have windows
        assert len(windows) > 0
        
        # First window should start at 0
        assert windows[0][0] == 0
        
        # Windows should overlap
        if len(windows) > 1:
            window1_end = windows[0][1]
            window2_start = windows[1][0]
            assert window2_start < window1_end  # Overlap
    
    def test_window_evaluation(self):
        """Should evaluate individual window."""
        df = pd.DataFrame({
            "Temperatura do líquido de arrefecimento do motor - CTS": [90.0] * 20,
            "rich_idle_score": [3.0] * 20,  # High rich-idle
            "Tensão do módulo": [11.0] * 20,  # Low voltage
        })
        
        detector = TemporalFaultDetector()
        window = detector._evaluate_window(df, 0, 20)
        
        assert window.start_idx == 0
        assert window.end_idx == 20
        assert window.fault_detected is True
        assert window.rich_ratio > 0
        assert window.voltage_min < 12.0
    
    def test_temporal_evaluation_sustained_fault(self):
        """Should detect sustained fault across windows."""
        # Create data with sustained fault
        df = pd.DataFrame({
            "Temperatura do líquido de arrefecimento do motor - CTS": [90.0] * 100,
            "rich_idle_score": [3.0] * 100,  # Sustained rich-idle
            "Tensão do módulo": [11.0] * 100,  # Sustained low voltage
        })
        
        detector = TemporalFaultDetector(
            window_size_sec=10.0,
            min_windows_fault=2
        )
        
        result = detector.evaluate_temporal(df)
        
        assert result["fault_detected"] is True
        assert result["fault_window_count"] >= 2
        assert len(result["windows"]) > 0
        assert "Temporal fault detected" in result["reasons"][0]
    
    def test_temporal_evaluation_transient_fault(self):
        """Should reject transient single-window faults."""
        # Only first 5 frames have fault (very transient)
        rich_scores = [3.0] * 5 + [0.0] * 95
        voltages = [11.0] * 5 + [13.5] * 95
        
        df = pd.DataFrame({
            "Temperatura do líquido de arrefecimento do motor - CTS": [90.0] * 100,
            "rich_idle_score": rich_scores,
            "Tensão do módulo": voltages,
        })
        
        detector = TemporalFaultDetector(
            window_size_sec=10.0,
            min_windows_fault=3,  # Require 3 windows to be more strict
            overlap_ratio=0.5
        )
        
        result = detector.evaluate_temporal(df)
        
        # Should not trigger (transient fault in <3 windows)
        assert result["fault_detected"] is False
        assert result["fault_window_count"] < 3
