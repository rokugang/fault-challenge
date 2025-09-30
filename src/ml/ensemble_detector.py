"""
Ensemble ML Detector with Explainability

Combines multiple anomaly detection models with weighted voting.
Uses SHAP TreeExplainer for feature attribution.

Author: Rohit Gangupantulu
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import joblib
from dataclasses import dataclass

from src.detection.detector import FaultDetector, DetectionResult


@dataclass
class ExplainableResult:
    """Detection result with feature importance."""
    fault_detected: bool
    confidence: str
    reasons: List[str]
    metrics: Dict[str, float]
    top_anomaly_features: List[Tuple[str, float]]  # Feature name, importance


class EnsembleDetector(FaultDetector):
    """
    Ensemble anomaly detector with explainability.
    
    Uses weighted voting across three models:
    - IsolationForest (50%)
    - LOF (30%)
    - Mahalanobis (20%)
    
    Outputs confidence scores and feature attribution.
    """
    
    def __init__(self, models_dir: Optional[Path] = None, use_shap: bool = True):
        super().__init__()
        self.models = {}
        self.weights = {}
        self.feature_names = []
        self.shap_explainer = None
        self.use_shap = use_shap
        
        if models_dir and models_dir.exists():
            self._load_ensemble(models_dir)
            
            # Auto-initialize SHAP if IsolationForest loaded
            if use_shap and "isolation_forest" in self.models:
                self._init_shap()
    
    def _load_ensemble(self, models_dir: Path):
        """Load multiple models for voting."""
        # Try loading different models
        model_configs = [
            ("isolation_forest", "isolation_forest_0.050.pkl", 0.5),
            ("lof", "lof_0.050.pkl", 0.3),
            ("mahalanobis", "mahalanobis_95.pkl", 0.2),
        ]
        
        for name, filename, weight in model_configs:
            model_path = models_dir / filename
            if model_path.exists():
                try:
                    if name == "mahalanobis":
                        # Mahalanobis stored as tuple (cov, threshold)
                        self.models[name] = joblib.load(model_path)
                    else:
                        self.models[name] = joblib.load(model_path)
                    self.weights[name] = weight
                    print(f"Loaded {name} for ensemble")
                except Exception as e:
                    print(f"Failed to load {name}: {e}")
    
    def _prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Extract features and remember names for explainability."""
        feature_cols = [
            col for col in df.columns 
            if col not in ["fault_detected", "timestamp"]
            and df[col].dtype in [np.float64, np.int64, np.float32, np.int32]
        ]
        
        X = df[feature_cols].fillna(df[feature_cols].mean())
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.mean())
        
        self.feature_names = feature_cols
        return X.values, feature_cols
    
    def _ensemble_predict(self, X: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """
        Weighted voting across models.
        Returns: (anomaly_ratio, individual_scores)
        """
        if not self.models:
            return 0.0, {}
        
        individual_scores = {}
        weighted_predictions = []
        
        for name, model in self.models.items():
            weight = self.weights[name]
            
            try:
                if name == "mahalanobis":
                    cov, threshold = model
                    distances = cov.mahalanobis(X)
                    predictions = (distances > threshold).astype(int) * 2 - 1  # Convert to -1/1
                    score = float((predictions == -1).mean())
                else:
                    predictions = model.predict(X)
                    score = float((predictions == -1).mean())
                
                individual_scores[name] = score
                weighted_predictions.append(score * weight)
                
            except Exception as e:
                print(f"Prediction failed for {name}: {e}")
                continue
        
        if weighted_predictions:
            ensemble_score = sum(weighted_predictions) / sum(self.weights.values())
        else:
            ensemble_score = 0.0
        
        return ensemble_score, individual_scores
    
    def _init_shap(self):
        """Initialize SHAP explainer automatically."""
        try:
            from src.ml.explainability import SHAPExplainer, SHAP_AVAILABLE
            
            if SHAP_AVAILABLE and "isolation_forest" in self.models:
                self.shap_explainer = SHAPExplainer(
                    self.models["isolation_forest"],
                    self.feature_names
                )
                print("SHAP explainer initialized for IsolationForest")
        except Exception as e:
            print(f"SHAP initialization failed: {e}. Using z-score fallback.")
            self.shap_explainer = None
    
    def _explain_anomalies(self, df: pd.DataFrame, X: np.ndarray) -> List[Tuple[str, float]]:
        """
        Identify which features contribute most to anomaly.
        
        Uses SHAP if available, falls back to z-scores.
        """
        if not self.feature_names:
            return []
        
        # Try SHAP first
        if self.shap_explainer is not None:
            try:
                top_features, _ = self.shap_explainer.explain_prediction(X)
                return top_features[:5]
            except Exception as e:
                print(f"SHAP failed: {e}, using z-score")
        
        # Fallback: z-score attribution
        feature_deviations = []
        
        for i, feature_name in enumerate(self.feature_names):
            feature_values = X[:, i]
            mean_val = np.mean(feature_values)
            std_val = np.std(feature_values)
            
            if std_val > 0:
                max_zscore = np.abs((feature_values - mean_val) / std_val).max()
            else:
                max_zscore = 0.0
            
            feature_deviations.append((feature_name, float(max_zscore)))
        
        feature_deviations.sort(key=lambda x: x[1], reverse=True)
        return feature_deviations[:5]
    
    def detect_with_explanation(self, df: pd.DataFrame) -> ExplainableResult:
        """
        Enhanced detection with feature-level explanations.
        """
        # Get rule-based result
        rule_result = self.run_detection(df)
        
        # Prepare features for ML
        X, feature_names = self._prepare_features(df)
        
        # Get ensemble predictions
        ensemble_anomaly_ratio, individual_scores = self._ensemble_predict(X)
        
        # Get feature importance
        top_features = self._explain_anomalies(df, X)
        
        # Determine confidence with ensemble voting
        ml_flags_fault = ensemble_anomaly_ratio > 0.10
        
        # Enhanced decision logic
        if rule_result.fault_detected and ml_flags_fault:
            confidence = "high"
            reasons = rule_result.reasons + [
                f"Ensemble confirms ({ensemble_anomaly_ratio*100:.1f}% anomalous)",
                f"Models agree: " + ", ".join(f"{k}={v*100:.0f}%" for k, v in individual_scores.items())
            ]
        elif rule_result.fault_detected:
            confidence = "medium"
            reasons = rule_result.reasons + [f"Ensemble uncertain ({ensemble_anomaly_ratio*100:.1f}%)"]
        elif ml_flags_fault:
            confidence = "low"
            reasons = [f"Ensemble detects anomaly ({ensemble_anomaly_ratio*100:.1f}%)"]
            # Add top anomalous features to reasons
            if top_features:
                reasons.append(f"Unusual: {', '.join(f[0][:30] for f in top_features[:3])}")
            rule_result.fault_detected = True
        else:
            confidence = "none"
            reasons = ["No fault detected"]
        
        # Enhanced metrics
        enhanced_metrics = {
            **rule_result.metrics,
            "ensemble_anomaly_ratio": ensemble_anomaly_ratio,
            **{f"model_{k}": v for k, v in individual_scores.items()},
            "confidence": confidence,
        }
        
        return ExplainableResult(
            fault_detected=rule_result.fault_detected,
            confidence=confidence,
            reasons=reasons,
            metrics=enhanced_metrics,
            top_anomaly_features=top_features
        )
    
    def run_detection(self, df: pd.DataFrame) -> DetectionResult:
        """Backward compatible detection method."""
        result = self.detect_with_explanation(df)
        return DetectionResult(
            fault_detected=result.fault_detected,
            reasons=result.reasons,
            metrics=result.metrics
        )
