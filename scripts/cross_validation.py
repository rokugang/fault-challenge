"""
Cross-Validation Analysis for Model Performance

Performs k-fold cross-validation on anomaly detection models to provide
robust performance estimates despite limited data.

Author: Rohit Gangupantulu
"""
from __future__ import annotations

import sys
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.data.loader import DataLoader
from src.features.engineering import FeatureEngineer


def create_frame_labels(df: pd.DataFrame) -> np.ndarray:
    """Label frames as fault (1) if rich-idle OR low-voltage present."""
    rich_condition = df["rich_idle_score"].fillna(0) >= 2
    voltage_condition = df["Tensão do módulo"].fillna(14) <= 12.0
    fault_frames = rich_condition | voltage_condition
    return fault_frames.astype(int).values


def prepare_cv_dataset():
    """Prepare dataset for cross-validation."""
    loader = DataLoader()
    engineer = FeatureEngineer()
    
    reference_dir = config.DATA_ROOT / "references"
    reference_files = sorted(reference_dir.glob("*.csv"))
    
    all_samples = []
    all_labels = []
    
    for ref_file in reference_files[:5]:
        try:
            result = loader.load_reference_file(ref_file)
            df = result.numeric
            if len(df) < 10:
                continue
            
            df_features = engineer.transform(df)
            labels = create_frame_labels(df_features)
            
            all_samples.append(df_features)
            all_labels.append(labels)
        except:
            continue
    
    try:
        fault_result = loader.load_reference_file(config.DATA_ROOT / "fault_example.csv")
        df_fault = fault_result.numeric
        df_fault_features = engineer.transform(df_fault)
        labels_fault = create_frame_labels(df_fault_features)
        
        all_samples.append(df_fault_features)
        all_labels.append(labels_fault)
    except:
        pass
    
    if all_samples:
        df_combined = pd.concat(all_samples, ignore_index=True)
        y_combined = np.concatenate(all_labels)
        
        feature_cols = [
            col for col in df_combined.columns
            if col not in ["fault_detected", "timestamp"]
            and df_combined[col].dtype in [np.float64, np.int64, np.float32, np.int32]
        ]
        
        X = df_combined[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values
        return X, y_combined, feature_cols
    
    return None, None, []


def cross_validate_model(model, scaler, X, y, n_splits=5):
    """
    Perform stratified k-fold cross-validation.
    
    Returns dict with mean and std of metrics across folds.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    roc_aucs = []
    precisions = []
    recalls = []
    f1_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        if scaler:
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
        
        model.fit(X_train)
        
        predictions = model.predict(X_val)
        y_pred = (predictions == -1).astype(int)
        
        if hasattr(model, 'decision_function'):
            scores = -model.decision_function(X_val)
        elif hasattr(model, 'score_samples'):
            scores = -model.score_samples(X_val)
        else:
            scores = -predictions.astype(float)
        
        if len(np.unique(y_val)) > 1:
            roc_auc = roc_auc_score(y_val, scores)
            roc_aucs.append(roc_auc)
        
        precision = precision_score(y_val, y_pred, zero_division=0)
        recall = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        
        roc_str = f"{roc_auc:.3f}" if roc_aucs else "N/A"
        print(f"  Fold {fold+1}: ROC-AUC={roc_str}, "
              f"Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
    
    return {
        "roc_auc_mean": np.mean(roc_aucs) if roc_aucs else None,
        "roc_auc_std": np.std(roc_aucs) if roc_aucs else None,
        "precision_mean": np.mean(precisions),
        "precision_std": np.std(precisions),
        "recall_mean": np.mean(recalls),
        "recall_std": np.std(recalls),
        "f1_mean": np.mean(f1_scores),
        "f1_std": np.std(f1_scores)
    }


def main():
    print("=" * 80)
    print("CROSS-VALIDATION ANALYSIS")
    print("=" * 80)
    
    print("\nPreparing dataset...")
    X, y, feature_names = prepare_cv_dataset()
    
    if X is None:
        print("Failed to prepare dataset")
        return
    
    print(f"Dataset: {len(X)} samples, {X.shape[1]} features")
    print(f"Class distribution: {(y==0).sum()} normal, {(y==1).sum()} fault")
    
    models_dir = config.ARTIFACTS_DIR / "ml_models"
    
    print("\n" + "=" * 80)
    print("IsolationForest (1% contamination)")
    print("=" * 80)
    
    model_path = models_dir / "isolation_forest_0.010.pkl"
    if model_path.exists():
        from sklearn.ensemble import IsolationForest
        model_template = IsolationForest(contamination=0.01, random_state=42)
        
        cv_results = cross_validate_model(model_template, None, X, y, n_splits=5)
        
        print("\nCross-Validation Results:")
        if cv_results["roc_auc_mean"]:
            print(f"  ROC-AUC:   {cv_results['roc_auc_mean']:.3f} (+/- {cv_results['roc_auc_std']:.3f})")
        print(f"  Precision: {cv_results['precision_mean']:.3f} (+/- {cv_results['precision_std']:.3f})")
        print(f"  Recall:    {cv_results['recall_mean']:.3f} (+/- {cv_results['recall_std']:.3f})")
        print(f"  F1 Score:  {cv_results['f1_mean']:.3f} (+/- {cv_results['f1_std']:.3f})")
        
        output_dir = config.ARTIFACTS_DIR / "performance_evaluation"
        output_dir.mkdir(exist_ok=True)
        
        output_file = output_dir / "cross_validation_results.json"
        with open(output_file, 'w') as f:
            json.dump(cv_results, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")
    else:
        print("Model file not found. Train models first.")
    
    print("\n" + "=" * 80)
    print("Analysis complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
