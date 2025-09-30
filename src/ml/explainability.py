"""
SHAP-based Explainability for ML Models

Uses SHAP (SHapley Additive exPlanations) for accurate feature attribution.
Author: Rohit Gangupantulu
"""
from __future__ import annotations

from typing import List, Tuple, Optional, Dict
import numpy as np
import pandas as pd

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not installed. Install with: pip install shap")


class SHAPExplainer:
    """
    SHAP-based feature attribution for anomaly detection models.
    
    Uses TreeExplainer for IsolationForest to compute exact Shapley values.
    Falls back to z-score approximation if SHAP unavailable.
    """
    
    def __init__(self, model, feature_names: List[str]):
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP library required. Install: pip install shap")
        
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        
        # Initialize SHAP explainer based on model type
        model_type = type(model).__name__
        
        if model_type == "IsolationForest":
            # TreeExplainer for tree-based models (exact, fast)
            self.explainer = shap.TreeExplainer(model)
        else:
            # KernelExplainer for other models (slower, approximate)
            # Would need background data sample
            print(f"Warning: {model_type} not optimized for SHAP. Using z-score fallback.")
    
    def explain_prediction(
        self, 
        X: np.ndarray, 
        sample_idx: Optional[int] = None
    ) -> Tuple[List[Tuple[str, float]], Optional[np.ndarray]]:
        """
        Get SHAP values for a prediction.
        
        Args:
            X: Feature matrix
            sample_idx: Specific sample to explain (None = explain all)
        
        Returns:
            (top_features, shap_values)
            top_features: List of (feature_name, shap_value) sorted by importance
            shap_values: Full SHAP value matrix
        """
        if self.explainer is None:
            # Fallback to z-score if SHAP not available
            return self._zscore_fallback(X, sample_idx), None
        
        try:
            # Compute SHAP values
            shap_values = self.explainer.shap_values(X)
            
            # For IsolationForest, shap_values is anomaly score contribution
            # More negative = more anomalous
            
            if sample_idx is not None:
                sample_shap = shap_values[sample_idx]
            else:
                # Aggregate across all samples (mean absolute)
                sample_shap = np.abs(shap_values).mean(axis=0)
            
            # Sort features by absolute SHAP value
            # Ensure we don't exceed bounds
            n_features = min(len(sample_shap), len(self.feature_names))
            feature_importance = [
                (self.feature_names[i], float(sample_shap[i]))
                for i in range(n_features)
            ]
            feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
            
            return feature_importance[:10], shap_values
            
        except Exception as e:
            print(f"SHAP computation failed: {e}")
            return self._zscore_fallback(X, sample_idx), None
    
    def _zscore_fallback(
        self, 
        X: np.ndarray, 
        sample_idx: Optional[int]
    ) -> List[Tuple[str, float]]:
        """Fallback to z-score attribution if SHAP fails."""
        if sample_idx is not None:
            sample_data = X[sample_idx:sample_idx+1]
        else:
            sample_data = X
        
        # Compute z-scores
        mean_vals = np.mean(sample_data, axis=0)
        std_vals = np.std(sample_data, axis=0)
        
        z_scores = []
        for i in range(len(self.feature_names)):
            if std_vals[i] > 0:
                z = abs((mean_vals[i] - X[:, i].mean()) / std_vals[i])
            else:
                z = 0.0
            z_scores.append((self.feature_names[i], float(z)))
        
        z_scores.sort(key=lambda x: x[1], reverse=True)
        return z_scores[:10]
    
    def plot_waterfall(
        self, 
        X: np.ndarray, 
        sample_idx: int, 
        max_display: int = 10
    ):
        """
        Create waterfall plot showing feature contributions.
        
        Args:
            X: Feature matrix
            sample_idx: Sample to explain
            max_display: Number of features to show
        """
        if not SHAP_AVAILABLE or self.explainer is None:
            print("SHAP not available for plotting")
            return
        
        try:
            shap_values = self.explainer.shap_values(X)
            shap.plots.waterfall(
                shap.Explanation(
                    values=shap_values[sample_idx],
                    base_values=self.explainer.expected_value,
                    data=X[sample_idx],
                    feature_names=self.feature_names
                ),
                max_display=max_display
            )
        except Exception as e:
            print(f"Plot failed: {e}")
    
    def get_feature_importance_summary(
        self, 
        X: np.ndarray
    ) -> Dict[str, float]:
        """
        Get global feature importance across all samples.
        
        Returns:
            Dictionary of feature_name: mean_abs_shap_value
        """
        top_features, shap_values = self.explain_prediction(X)
        
        if shap_values is not None:
            # Compute mean absolute SHAP value per feature
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            return {
                self.feature_names[i]: float(mean_abs_shap[i])
                for i in range(len(self.feature_names))
            }
        else:
            # Fallback: return z-score based importance
            return {name: score for name, score in top_features}
