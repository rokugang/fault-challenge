"""
ML-Enhanced Fault Detector

Combines rule-based detection with ML anomaly scoring for improved accuracy.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

from src.detection.detector import FaultDetector, DetectionResult


class MLFaultDetector(FaultDetector):
    """
    Enhanced detector using ML anomaly scores + rule-based logic.
    
    Approach:
    1. Train ML models (IsolationForest, etc.) on clean reference features
    2. Score new logs using both ML anomaly detection and domain rules
    3. Combine scores for final decision
    """
    
    def __init__(self, model_path: Optional[Path] = None, scaler_path: Optional[Path] = None):
        super().__init__()
        self.model = None
        self.scaler = None
        self.model_type = None
        
        if model_path and model_path.exists():
            self.load_model(model_path, scaler_path)
    
    def load_model(self, model_path: Path, scaler_path: Optional[Path] = None):
        """Load trained ML model."""
        self.model = joblib.load(model_path)
        self.model_type = type(self.model).__name__
        
        if scaler_path and scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
        
        print(f"Loaded {self.model_type} from {model_path.name}")
    
    def _prepare_ml_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract features for ML model."""
        feature_cols = [
            col for col in df.columns 
            if col not in ["fault_detected", "timestamp"]
            and df[col].dtype in [np.float64, np.int64, np.float32, np.int32]
        ]
        
        X = df[feature_cols].fillna(df[feature_cols].mean())
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.mean())
        
        if self.scaler:
            X = self.scaler.transform(X.values)
        else:
            X = X.values
        
        return X
    
    def _ml_anomaly_score(self, df: pd.DataFrame) -> Tuple[float, float]:
        """
        Compute ML anomaly score.
        
        Returns:
            (anomaly_ratio, anomaly_score_mean)
        """
        if self.model is None:
            return 0.0, 0.0
        
        try:
            X = self._prepare_ml_features(df)
            
            # Get predictions (-1 for anomalies, 1 for normal)
            predictions = self.model.predict(X)
            anomaly_ratio = float((predictions == -1).mean())
            
            # Get anomaly scores (lower = more anomalous)
            if hasattr(self.model, 'decision_function'):
                scores = self.model.decision_function(X)
                anomaly_score_mean = float(scores.mean())
            elif hasattr(self.model, 'score_samples'):
                scores = self.model.score_samples(X)
                anomaly_score_mean = float(scores.mean())
            else:
                anomaly_score_mean = -anomaly_ratio  # fallback
            
            return anomaly_ratio, anomaly_score_mean
            
        except Exception as e:
            print(f"ML scoring failed: {e}")
            return 0.0, 0.0
    
    def detect(self, df: pd.DataFrame) -> DetectionResult:
        """
        Hybrid detection: ML + rules.
        
        Strategy:
        - If ML flags high anomaly ratio (>10%) → likely fault
        - If rules flag fault → check ML for confirmation
        - Combined confidence score
        """
        # Get rule-based result
        rule_result = super().detect(df)
        
        # Get ML anomaly score
        ml_anomaly_ratio, ml_score = self._ml_anomaly_score(df)
        
        # Combine detections
        ml_flags_fault = ml_anomaly_ratio > 0.10  # >10% frames anomalous
        
        # Decision logic
        if rule_result.fault_detected and ml_flags_fault:
            # Both agree = high confidence
            confidence = "high"
            reasons = rule_result.reasons + [
                f"ML confirms anomaly ({ml_anomaly_ratio*100:.1f}% anomalous frames)"
            ]
        elif rule_result.fault_detected:
            # Rules only
            confidence = "medium"
            reasons = rule_result.reasons + [
                f"ML neutral ({ml_anomaly_ratio*100:.1f}% anomalous)"
            ]
        elif ml_flags_fault:
            # ML only
            confidence = "low"
            reasons = [f"ML detects anomaly ({ml_anomaly_ratio*100:.1f}% anomalous frames)"]
            rule_result.fault_detected = True
        else:
            # Neither flag fault
            confidence = "none"
            reasons = rule_result.reasons
        
        # Enhanced metrics
        enhanced_metrics = {
            **rule_result.metrics,
            "ml_anomaly_ratio": ml_anomaly_ratio,
            "ml_anomaly_score": ml_score,
            "confidence": confidence,
            "model_type": self.model_type or "none"
        }
        
        return DetectionResult(
            fault_detected=rule_result.fault_detected,
            reasons=reasons,
            metrics=enhanced_metrics
        )
