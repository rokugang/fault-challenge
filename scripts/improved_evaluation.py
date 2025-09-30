"""
Improved Performance Evaluation with Frame-Level Labels

Fixes evaluation by labeling individual frames as fault/normal based on 
actual rich-idle and low-voltage indicators, not entire files.

Author: Rohit Gangupantulu
"""
from __future__ import annotations

import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix, classification_report
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.data.loader import DataLoader
from src.features.engineering import FeatureEngineer

sns.set_theme(style="whitegrid", palette="muted")


def create_frame_level_labels(df: pd.DataFrame) -> np.ndarray:
    """
    Create frame-level labels based on actual fault indicators.
    
    A frame is labeled as FAULT (1) if EITHER:
    - rich_idle_score >= 2 (actual rich-idle condition)
    OR
    - Voltage <= 12V (low voltage condition)
    
    This matches the actual fault patterns where rich-idle and low-voltage
    may occur in different frames (no temporal overlap).
    """
    rich_condition = df["rich_idle_score"].fillna(0) >= 2
    voltage_condition = df["Tensão do módulo"].fillna(14) <= 12.0
    
    # Either condition indicates anomalous frame
    fault_frames = rich_condition | voltage_condition
    
    return fault_frames.astype(int).values


def prepare_dataset_with_frame_labels() -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Prepare dataset with proper frame-level labels.
    
    Returns:
        X: Feature matrix
        y: Frame-level labels (0=normal, 1=fault)
        feature_names: List of feature names
    """
    loader = DataLoader()
    engineer = FeatureEngineer()
    
    # Load reference files (normal frames)
    reference_dir = config.DATA_ROOT / "references"
    reference_files = sorted(reference_dir.glob("*.csv"))
    
    all_samples = []
    all_labels = []
    
    # Process reference files (mostly normal)
    for ref_file in reference_files[:5]:  # Use 5 clean files
        try:
            result = loader.load_reference_file(ref_file)
            df = result.numeric
            if len(df) < 10:
                continue
            
            df_features = engineer.transform(df)
            
            # Label frames based on actual conditions
            labels = create_frame_level_labels(df_features)
            
            all_samples.append(df_features)
            all_labels.append(labels)
            
            print(f"  {ref_file.name}: {len(df)} frames, {labels.sum()} fault frames")
        except Exception as e:
            print(f"  Skipped {ref_file.name}: {e}")
            continue
    
    # Load fault example
    try:
        fault_result = loader.load_reference_file(config.DATA_ROOT / "fault_example.csv")
        df_fault = fault_result.numeric
        df_fault_features = engineer.transform(df_fault)
        
        labels_fault = create_frame_level_labels(df_fault_features)
        
        all_samples.append(df_fault_features)
        all_labels.append(labels_fault)
        
        print(f"  fault_example.csv: {len(df_fault)} frames, {labels_fault.sum()} fault frames")
    except Exception as e:
        print(f"  Failed to load fault_example.csv: {e}")
        return None, None, []
    
    # Combine
    if all_samples:
        df_combined = pd.concat(all_samples, ignore_index=True)
        y_combined = np.concatenate(all_labels)
        
        # Extract features
        feature_cols = [
            col for col in df_combined.columns
            if col not in ["fault_detected", "timestamp"]
            and df_combined[col].dtype in [np.float64, np.int64, np.float32, np.int32]
        ]
        
        X = df_combined[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values
        
        return X, y_combined, feature_cols
    
    return None, None, []


def evaluate_with_frame_labels():
    """Improved evaluation with frame-level labels."""
    
    print("=" * 80)
    print("IMPROVED PERFORMANCE EVALUATION (Frame-Level Labels)")
    print("=" * 80)
    
    # Load models
    print("\nLoading models...")
    models_dir = config.ARTIFACTS_DIR / "ml_models"
    
    models = {}
    model_configs = [
        ("IsolationForest", "isolation_forest_0.050.pkl", None),
        ("IsolationForest_1%", "isolation_forest_0.010.pkl", None),
        ("LOF", "lof_0.050.pkl", "lof_scaler_0.050.pkl"),
    ]
    
    for name, model_file, scaler_file in model_configs:
        model_path = models_dir / model_file
        if model_path.exists():
            model = joblib.load(model_path)
            scaler = None
            if scaler_file:
                scaler_path = models_dir / scaler_file
                if scaler_path.exists():
                    scaler = joblib.load(scaler_path)
            models[name] = {"model": model, "scaler": scaler}
            print(f"  Loaded {name}")
    
    # Prepare dataset with frame-level labels
    print("\nPreparing dataset with frame-level labels...")
    X, y, feature_names = prepare_dataset_with_frame_labels()
    
    if X is None:
        print("Failed to prepare dataset")
        return
    
    print(f"\nDataset statistics:")
    print(f"  Total frames: {len(X)}")
    print(f"  Normal frames: {(y == 0).sum()} ({(y == 0).mean()*100:.1f}%)")
    print(f"  Fault frames: {(y == 1).sum()} ({(y == 1).mean()*100:.1f}%)")
    
    # Evaluate each model
    print("\n" + "=" * 80)
    print("RESULTS WITH FRAME-LEVEL LABELS")
    print("=" * 80)
    
    results = []
    
    for model_name, model_data in models.items():
        print(f"\n{model_name}:")
        
        # Scale if needed
        if model_data["scaler"]:
            X_scaled = model_data["scaler"].transform(X)
        else:
            X_scaled = X
        
        # Predictions
        predictions = model_data["model"].predict(X_scaled)
        y_pred = (predictions == -1).astype(int)
        
        # Scores
        if hasattr(model_data["model"], 'decision_function'):
            scores = model_data["model"].decision_function(X_scaled)
            scores = -scores
        elif hasattr(model_data["model"], 'score_samples'):
            scores = model_data["model"].score_samples(X_scaled)
            scores = -scores
        else:
            scores = -predictions.astype(float)
        
        # Metrics
        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / len(y)
        
        # ROC
        fpr, tpr, _ = roc_curve(y, scores)
        roc_auc = auc(fpr, tpr)
        
        print(f"  Accuracy:  {accuracy:.3f}")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall:    {recall:.3f}")
        print(f"  F1 Score:  {f1:.3f}")
        print(f"  ROC-AUC:   {roc_auc:.3f}")
        print(f"  Confusion: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
        
        results.append({
            "model": model_name,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn)
        })
    
    # Save results
    output_dir = config.ARTIFACTS_DIR / "performance_evaluation"
    output_dir.mkdir(exist_ok=True)
    
    results_file = output_dir / "improved_evaluation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n" + "=" * 80)
    print(f"Results saved to: {results_file}")
    print("=" * 80)
    
    # Summary table
    print("\nSUMMARY:")
    print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'ROC-AUC':<10}")
    print("-" * 80)
    for r in results:
        print(f"{r['model']:<20} {r['accuracy']:<10.3f} {r['precision']:<10.3f} "
              f"{r['recall']:<10.3f} {r['f1']:<10.3f} {r['roc_auc']:<10.3f}")


if __name__ == "__main__":
    evaluate_with_frame_labels()
