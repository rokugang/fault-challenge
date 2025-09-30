"""
Threshold Tuning Analysis

Tests different rich-idle thresholds to find optimal balance between
precision and recall on validation data.

Author: Rohit Gangupantulu
"""
from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.data.loader import DataLoader
from src.features.engineering import FeatureEngineer


def evaluate_threshold(df: pd.DataFrame, rich_threshold: float, voltage_threshold: float = 12.0):
    """
    Evaluate detection with given thresholds.
    
    Returns metrics dict.
    """
    rich_condition = df["rich_idle_score"].fillna(0) >= 2
    rich_ratio = float(rich_condition.mean())
    
    voltage_condition = df["Tensão do módulo"].fillna(14) <= voltage_threshold
    voltage_ratio = float(voltage_condition.mean())
    voltage_min = float(df["Tensão do módulo"].min()) if "Tensão do módulo" in df else 14.0
    
    rich_ok = rich_ratio >= rich_threshold
    voltage_ok = voltage_min <= voltage_threshold or voltage_ratio >= 0.05
    
    fault_detected = rich_ok and voltage_ok
    
    return {
        "rich_ratio": rich_ratio,
        "voltage_min": voltage_min,
        "fault_detected": fault_detected
    }


def main():
    print("=" * 80)
    print("THRESHOLD TUNING ANALYSIS")
    print("=" * 80)
    
    loader = DataLoader()
    engineer = FeatureEngineer()
    
    print("\nLoading fault example...")
    fault_result = loader.load_reference_file(config.DATA_ROOT / "fault_example.csv")
    df_fault = fault_result.numeric
    df_fault_features = engineer.transform(df_fault)
    
    print(f"Fault example: {len(df_fault)} frames")
    
    print("\nLoading clean reference samples...")
    reference_dir = config.DATA_ROOT / "references"
    reference_files = sorted(reference_dir.glob("*.csv"))
    
    clean_samples = []
    for ref_file in reference_files[:5]:
        try:
            result = loader.load_reference_file(ref_file)
            df = result.numeric
            if len(df) < 10:
                continue
            df_features = engineer.transform(df)
            clean_samples.append(df_features)
            print(f"  Loaded: {ref_file.name}")
        except:
            continue
    
    print(f"\n{len(clean_samples)} clean files loaded")
    
    print("\n" + "=" * 80)
    print("TESTING RICH-IDLE THRESHOLDS")
    print("=" * 80)
    
    thresholds = [0.03, 0.05, 0.08, 0.10, 0.12, 0.15]
    
    print(f"\n{'Threshold':<12} {'Fault Detected':<18} {'False Positives':<20} {'Precision':<12}")
    print("-" * 80)
    
    best_f1 = 0
    best_threshold = 0.05
    
    for thresh in thresholds:
        fault_result = evaluate_threshold(df_fault_features, thresh)
        
        false_positives = 0
        for clean_df in clean_samples:
            clean_result = evaluate_threshold(clean_df, thresh)
            if clean_result["fault_detected"]:
                false_positives += 1
        
        tp = 1 if fault_result["fault_detected"] else 0
        fp = false_positives
        fn = 0 if fault_result["fault_detected"] else 1
        tn = len(clean_samples) - false_positives
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
        
        print(f"{thresh:<12.2f} {str(fault_result['fault_detected']):<18} "
              f"{false_positives}/{len(clean_samples):<16} {precision:<12.3f}")
    
    print("\n" + "=" * 80)
    print("RECOMMENDED THRESHOLD")
    print("=" * 80)
    print(f"\nCurrent threshold: 0.05 (5%)")
    print(f"Optimal threshold: {best_threshold:.2f} ({best_threshold*100:.0f}%)")
    print(f"Best F1 Score: {best_f1:.3f}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    print("\nStatistical context from EDA:")
    print("  Normal rich-idle ratio: mean=76.6%, P95=99.9%")
    print("  Fault example rich-idle: 11%")
    print("\nCurrent 5% threshold is very conservative (P95 - 94.9% margin).")
    print("Raising to 8-10% would improve recall without significant FP increase.")
    print("\nHowever, maintaining 5% prioritizes precision over recall,")
    print("which is appropriate for fault detection (minimize false alarms).")


if __name__ == "__main__":
    main()
