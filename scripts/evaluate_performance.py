"""
Performance Evaluation Suite

Comprehensive ML model evaluation with metrics, ROC curves, and comparison charts.
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
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.data.loader import DataLoader
from src.features.engineering import FeatureEngineer

# Styling
sns.set_theme(style="whitegrid", palette="muted")


def load_trained_models() -> Dict:
    """Load all trained models from artifacts."""
    models_dir = config.ARTIFACTS_DIR / "ml_models"
    
    models = {}
    model_configs = [
        ("isolation_forest", "isolation_forest_0.050.pkl", None),
        ("lof", "lof_0.050.pkl", "lof_scaler_0.050.pkl"),
        ("ocsvm", "ocsvm_0.050.pkl", "ocsvm_scaler_0.050.pkl"),
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
            print(f"Loaded {name}")
    
    return models


def prepare_dataset() -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Prepare dataset with labels.
    
    Returns:
        X: Feature matrix
        y: Labels (0=normal, 1=fault)
        feature_names: List of feature names
    """
    loader = DataLoader()
    engineer = FeatureEngineer()
    
    # Load reference files (normal, label=0)
    reference_dir = config.DATA_ROOT / "references"
    reference_files = sorted(reference_dir.glob("*.csv"))
    
    normal_samples = []
    for ref_file in reference_files[:3]:  # Use first 3 for speed
        try:
            result = loader.load_reference_file(ref_file)
            df = result.numeric
            if len(df) < 10:
                continue
            
            df_features = engineer.transform(df)
            normal_samples.append(df_features)
        except:
            continue
    
    # Load fault example (label=1)
    try:
        fault_result = loader.load_reference_file(config.DATA_ROOT / "fault_example.csv")
        df_fault = fault_result.numeric
        df_fault_features = engineer.transform(df_fault)
    except:
        df_fault_features = None
    
    # Combine
    if normal_samples and df_fault_features is not None:
        df_normal = pd.concat(normal_samples, ignore_index=True)
        
        # Sample to balance if needed
        n_fault = len(df_fault_features)
        df_normal_sampled = df_normal.sample(n=min(len(df_normal), n_fault * 10), random_state=42)
        
        # Extract features
        feature_cols = [
            col for col in df_normal_sampled.columns
            if col not in ["fault_detected", "timestamp"]
            and df_normal_sampled[col].dtype in [np.float64, np.int64, np.float32, np.int32]
        ]
        
        X_normal = df_normal_sampled[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values
        X_fault = df_fault_features[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values
        
        X = np.vstack([X_normal, X_fault])
        y = np.hstack([np.zeros(len(X_normal)), np.ones(len(X_fault))])
        
        return X, y, feature_cols
    
    return None, None, []


def evaluate_model(model, scaler, X, y, model_name: str) -> Dict:
    """Evaluate single model and return metrics."""
    
    # Scale if needed
    if scaler:
        X_scaled = scaler.transform(X)
    else:
        X_scaled = X
    
    # Predictions (-1 for anomaly, 1 for normal)
    predictions = model.predict(X_scaled)
    y_pred = (predictions == -1).astype(int)  # Convert to 0/1
    
    # Get anomaly scores if available
    if hasattr(model, 'decision_function'):
        scores = model.decision_function(X_scaled)
        scores = -scores  # More negative = more anomalous, flip for ROC
    elif hasattr(model, 'score_samples'):
        scores = model.score_samples(X_scaled)
        scores = -scores
    else:
        scores = -predictions.astype(float)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    
    # Metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    # ROC
    fpr, tpr, _ = roc_curve(y, scores)
    roc_auc = auc(fpr, tpr)
    
    # Precision-Recall
    prec, rec, _ = precision_recall_curve(y, scores)
    pr_auc = auc(rec, prec)
    
    return {
        "model_name": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": {"TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn)},
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "precision_curve": prec.tolist(),
        "recall_curve": rec.tolist()
    }


def plot_model_comparison(results: List[Dict], output_dir: Path):
    """Plot model comparison charts."""
    
    # Metrics comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    models = [r["model_name"] for r in results]
    metrics = ["accuracy", "precision", "recall", "f1_score"]
    titles = ["Accuracy", "Precision", "Recall", "F1 Score"]
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 2, idx % 2]
        values = [r[metric] for r in results]
        bars = ax.bar(models, values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax.set_ylim(0, 1.0)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(f'{title} Comparison', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'model_comparison.png'}")
    plt.close()


def plot_roc_curves(results: List[Dict], output_dir: Path):
    """Plot ROC curves for all models."""
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for result, color in zip(results, colors):
        ax.plot(result["fpr"], result["tpr"], 
               label=f'{result["model_name"]} (AUC = {result["roc_auc"]:.3f})',
               color=color, linewidth=2)
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title('ROC Curves - Model Comparison', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curves.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'roc_curves.png'}")
    plt.close()


def plot_confusion_matrices(results: List[Dict], output_dir: Path):
    """Plot confusion matrices for all models."""
    
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
    
    if n_models == 1:
        axes = [axes]
    
    for ax, result in zip(axes, results):
        cm = result["confusion_matrix"]
        cm_matrix = np.array([[cm["TN"], cm["FP"]], 
                              [cm["FN"], cm["TP"]]])
        
        sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='Blues', 
                   cbar=False, ax=ax,
                   xticklabels=['Normal', 'Fault'],
                   yticklabels=['Normal', 'Fault'])
        ax.set_title(f'{result["model_name"]}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted', fontsize=10)
        ax.set_ylabel('Actual', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrices.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'confusion_matrices.png'}")
    plt.close()


def main():
    print("Performance Evaluation Suite")
    print("=" * 70)
    
    # Load models
    print("\nLoading trained models...")
    models = load_trained_models()
    
    if not models:
        print("No models found! Run scripts/run_ml_experiments.py first.")
        return
    
    # Prepare dataset
    print("\nPreparing dataset...")
    X, y, feature_names = prepare_dataset()
    
    if X is None:
        print("Failed to prepare dataset.")
        return
    
    print(f"Dataset: {len(X)} samples, {X.shape[1]} features")
    print(f"Normal samples: {int((y == 0).sum())}, Fault samples: {int((y == 1).sum())}")
    
    # Evaluate each model
    print("\nEvaluating models...")
    results = []
    
    for model_name, model_data in models.items():
        print(f"  Evaluating {model_name}...")
        metrics = evaluate_model(
            model_data["model"],
            model_data["scaler"],
            X, y,
            model_name
        )
        results.append(metrics)
    
    # Create output directory
    output_dir = config.ARTIFACTS_DIR / "performance_evaluation"
    output_dir.mkdir(exist_ok=True)
    
    # Save results
    results_file = output_dir / "evaluation_results.json"
    with open(results_file, 'w') as f:
        # Remove numpy arrays for JSON serialization
        json_results = []
        for r in results:
            r_copy = r.copy()
            r_copy.pop('fpr', None)
            r_copy.pop('tpr', None)
            r_copy.pop('precision_curve', None)
            r_copy.pop('recall_curve', None)
            json_results.append(r_copy)
        json.dump(json_results, f, indent=2)
    print(f"\nSaved: {results_file}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("PERFORMANCE SUMMARY")
    print("=" * 70)
    print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'ROC-AUC':<10}")
    print("-" * 70)
    
    for r in results:
        print(f"{r['model_name']:<20} {r['accuracy']:<10.3f} {r['precision']:<10.3f} "
              f"{r['recall']:<10.3f} {r['f1_score']:<10.3f} {r['roc_auc']:<10.3f}")
    
    print("=" * 70)
    
    # Generate plots
    print("\nGenerating visualizations...")
    plot_model_comparison(results, output_dir)
    plot_roc_curves(results, output_dir)
    plot_confusion_matrices(results, output_dir)
    
    print("\nEvaluation complete!")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
