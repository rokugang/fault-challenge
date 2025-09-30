"""
ML Experiments Framework

Compares multiple anomaly detection algorithms on reference features:
- Isolation Forest
- One-Class SVM
- Local Outlier Factor
- Autoencoder (Neural Network)
- Statistical (Mahalanobis Distance)
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.covariance import EmpiricalCovariance

import joblib


@dataclass
class ExperimentResult:
    """Results from a single experiment."""
    algorithm: str
    params: Dict[str, Any]
    cv_score_mean: float
    cv_score_std: float
    training_time: float
    n_features: int
    n_samples: int
    contamination: float
    notes: str = ""


class MLExperiments:
    """Framework for comparing anomaly detection algorithms."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[ExperimentResult] = []
        self.scaler = StandardScaler()
        
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract and normalize features for ML."""
        
        # Select numeric features from engineered dataset
        feature_cols = [
            col for col in df.columns 
            if col not in ["fault_detected", "timestamp"] 
            and df[col].dtype in [np.float64, np.int64, np.float32, np.int32]
        ]
        
        X = df[feature_cols].fillna(df[feature_cols].mean())
        
        # Handle infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.mean())
        
        return X.values
    
    def experiment_isolation_forest(
        self, 
        X: np.ndarray,
        contamination: float = 0.01,
        n_estimators: int = 100
    ) -> ExperimentResult:
        """Test Isolation Forest."""
        
        print(f"  Running Isolation Forest (contamination={contamination})...")
        
        start = time.time()
        
        model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=42,
            n_jobs=-1
        )
        
        # Use negative outlier scores for CV
        scores = cross_val_score(
            model, X, 
            scoring='neg_mean_absolute_error', 
            cv=3
        )
        
        model.fit(X)
        training_time = time.time() - start
        
        result = ExperimentResult(
            algorithm="IsolationForest",
            params={"contamination": contamination, "n_estimators": n_estimators},
            cv_score_mean=float(scores.mean()),
            cv_score_std=float(scores.std()),
            training_time=training_time,
            n_features=X.shape[1],
            n_samples=X.shape[0],
            contamination=contamination,
            notes="Efficient tree-based anomaly detection"
        )
        
        # Save model
        model_path = self.output_dir / f"isolation_forest_{contamination:.3f}.pkl"
        joblib.dump(model, model_path)
        
        return result
    
    def experiment_one_class_svm(
        self, 
        X: np.ndarray,
        nu: float = 0.01,
        kernel: str = "rbf"
    ) -> ExperimentResult:
        """Test One-Class SVM."""
        
        print(f"  Running One-Class SVM (nu={nu})...")
        
        start = time.time()
        
        # Scale features for SVM
        X_scaled = self.scaler.fit_transform(X)
        
        model = OneClassSVM(
            nu=nu,
            kernel=kernel,
            gamma='scale'
        )
        
        scores = cross_val_score(
            model, X_scaled,
            scoring='neg_mean_absolute_error',
            cv=3
        )
        
        model.fit(X_scaled)
        training_time = time.time() - start
        
        result = ExperimentResult(
            algorithm="OneClassSVM",
            params={"nu": nu, "kernel": kernel},
            cv_score_mean=float(scores.mean()),
            cv_score_std=float(scores.std()),
            training_time=training_time,
            n_features=X.shape[1],
            n_samples=X.shape[0],
            contamination=nu,
            notes="SVM-based novelty detection with RBF kernel"
        )
        
        # Save model and scaler
        model_path = self.output_dir / f"ocsvm_{nu:.3f}.pkl"
        scaler_path = self.output_dir / f"ocsvm_scaler_{nu:.3f}.pkl"
        joblib.dump(model, model_path)
        joblib.dump(self.scaler, scaler_path)
        
        return result
    
    def experiment_lof(
        self, 
        X: np.ndarray,
        contamination: float = 0.01,
        n_neighbors: int = 20
    ) -> ExperimentResult:
        """Test Local Outlier Factor."""
        
        print(f"  Running LOF (contamination={contamination})...")
        
        start = time.time()
        
        model = LocalOutlierFactor(
            contamination=contamination,
            n_neighbors=n_neighbors,
            novelty=True  # Enable predict for new data
        )
        
        # Scale for distance-based method
        X_scaled = self.scaler.fit_transform(X)
        
        scores = cross_val_score(
            model, X_scaled,
            scoring='neg_mean_absolute_error',
            cv=3
        )
        
        model.fit(X_scaled)
        training_time = time.time() - start
        
        result = ExperimentResult(
            algorithm="LocalOutlierFactor",
            params={"contamination": contamination, "n_neighbors": n_neighbors},
            cv_score_mean=float(scores.mean()),
            cv_score_std=float(scores.std()),
            training_time=training_time,
            n_features=X.shape[1],
            n_samples=X.shape[0],
            contamination=contamination,
            notes="Density-based local outlier detection"
        )
        
        model_path = self.output_dir / f"lof_{contamination:.3f}.pkl"
        scaler_path = self.output_dir / f"lof_scaler_{contamination:.3f}.pkl"
        joblib.dump(model, model_path)
        joblib.dump(self.scaler, scaler_path)
        
        return result
    
    def experiment_mahalanobis(
        self,
        X: np.ndarray,
        threshold_percentile: float = 95
    ) -> ExperimentResult:
        """Test Mahalanobis distance (statistical approach)."""
        
        print(f"  Running Mahalanobis Distance (p{threshold_percentile})...")
        
        start = time.time()
        
        # Fit covariance
        cov = EmpiricalCovariance()
        cov.fit(X)
        
        # Compute distances
        distances = cov.mahalanobis(X)
        threshold = np.percentile(distances, threshold_percentile)
        
        # Pseudo CV score (reconstruction error)
        score_mean = -float(np.mean(distances))
        score_std = float(np.std(distances))
        
        training_time = time.time() - start
        
        result = ExperimentResult(
            algorithm="Mahalanobis",
            params={"threshold_percentile": threshold_percentile},
            cv_score_mean=score_mean,
            cv_score_std=score_std,
            training_time=training_time,
            n_features=X.shape[1],
            n_samples=X.shape[0],
            contamination=(100 - threshold_percentile) / 100,
            notes=f"Statistical distance, threshold={threshold:.2f}"
        )
        
        # Save covariance model
        model_path = self.output_dir / f"mahalanobis_{threshold_percentile}.pkl"
        joblib.dump((cov, threshold), model_path)
        
        return result
    
    def run_all_experiments(
        self,
        X: np.ndarray,
        contaminations: List[float] = [0.01, 0.05, 0.10]
    ) -> List[ExperimentResult]:
        """Run all algorithm comparisons."""
        
        print(f"\nRunning ML experiments on {X.shape[0]} samples, {X.shape[1]} features\n")
        
        self.results = []
        
        # Isolation Forest with different contamination rates
        for contam in contaminations:
            try:
                result = self.experiment_isolation_forest(X, contamination=contam)
                self.results.append(result)
            except Exception as e:
                print(f"    Failed: {e}")
        
        # One-Class SVM
        for nu in contaminations:
            try:
                result = self.experiment_one_class_svm(X, nu=nu)
                self.results.append(result)
            except Exception as e:
                print(f"    Failed: {e}")
        
        # LOF
        for contam in contaminations:
            try:
                result = self.experiment_lof(X, contamination=contam)
                self.results.append(result)
            except Exception as e:
                print(f"    Failed: {e}")
        
        # Mahalanobis
        try:
            result = self.experiment_mahalanobis(X, threshold_percentile=95)
            self.results.append(result)
        except Exception as e:
            print(f"    Failed: {e}")
        
        return self.results
    
    def print_results(self):
        """Print comparison table."""
        
        print("\n" + "=" * 100)
        print("EXPERIMENT RESULTS")
        print("=" * 100)
        print(f"{'Algorithm':<20} {'Params':<25} {'CV Score':<15} {'Time (s)':<12} {'Notes':<30}")
        print("-" * 100)
        
        for result in self.results:
            params_str = str(result.params)[:24]
            score_str = f"{result.cv_score_mean:.4f}±{result.cv_score_std:.4f}"
            print(f"{result.algorithm:<20} {params_str:<25} {score_str:<15} "
                  f"{result.training_time:<12.3f} {result.notes[:29]:<30}")
        
        print("=" * 100)
        
        # Find best by CV score (highest = best)
        best = max(self.results, key=lambda r: r.cv_score_mean)
        print(f"\nBest performer: {best.algorithm} with params {best.params}")
    
    def save_results(self, filename: str = "experiment_results.json"):
        """Save results to JSON."""
        
        output_path = self.output_dir / filename
        with open(output_path, "w") as f:
            json.dump([asdict(r) for r in self.results], f, indent=2)
        
        print(f"\nResults saved to: {output_path}")
