"""
EDA: Threshold Justification Analysis

Analyzes reference dataset distributions to justify detection thresholds:
- Rich-idle ratio (currently 5%)
- Low-voltage threshold (currently 12V)
"""
from __future__ import annotations

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import DataLoader
from src.features.engineering import FeatureEngineer
from src.config import DATA_ROOT


def analyze_reference_distributions() -> Dict:
    """Analyze distributions from clean reference files."""
    
    loader = DataLoader()
    engineer = FeatureEngineer()
    
    reference_dir = DATA_ROOT / "references"
    reference_files = sorted(reference_dir.glob("*.csv"))
    
    print(f"Analyzing {len(reference_files)} reference files...")
    
    all_rich_ratios = []
    all_voltage_mins = []
    all_voltage_means = []
    all_stft_values = []
    all_lambda_values = []
    
    clean_files = 0
    
    for ref_file in reference_files:
        try:
            # Load and validate
            load_result = loader.load_reference_file(ref_file)
            df = load_result.numeric
            
            # Skip files with data quality issues
            if len(df) < 10:
                continue
                
            # Engineer features
            df_features = engineer.transform(df)
            
            # Collect metrics
            if "rich_idle_score" in df_features.columns:
                rich_ratio = (df_features["rich_idle_score"] > 0).mean()
                all_rich_ratios.append(rich_ratio)
            
            if "Tensão do módulo" in df.columns:
                voltage_min = df["Tensão do módulo"].min()
                voltage_mean = df["Tensão do módulo"].mean()
                all_voltage_mins.append(voltage_min)
                all_voltage_means.append(voltage_mean)
            
            if "Ajuste de combustível de curto prazo - Banco 1" in df.columns:
                stft = df["Ajuste de combustível de curto prazo - Banco 1"]
                all_stft_values.extend(stft.dropna().tolist())
            
            if "Sonda lambda - Banco 1, sensor 1" in df.columns:
                lambda_vals = df["Sonda lambda - Banco 1, sensor 1"]
                all_lambda_values.extend(lambda_vals.dropna().tolist())
            
            clean_files += 1
            
        except Exception as e:
            print(f"  Skipped {ref_file.name}: {str(e)[:50]}")
            continue
    
    print(f"\nAnalyzed {clean_files} clean reference files\n")
    
    # Statistical analysis
    results = {
        "files_analyzed": clean_files,
        "rich_idle_ratio": {
            "mean": float(np.mean(all_rich_ratios)) if all_rich_ratios else 0.0,
            "std": float(np.std(all_rich_ratios)) if all_rich_ratios else 0.0,
            "p95": float(np.percentile(all_rich_ratios, 95)) if all_rich_ratios else 0.0,
            "p99": float(np.percentile(all_rich_ratios, 99)) if all_rich_ratios else 0.0,
            "max": float(np.max(all_rich_ratios)) if all_rich_ratios else 0.0,
        },
        "voltage_min": {
            "mean": float(np.mean(all_voltage_mins)) if all_voltage_mins else 0.0,
            "std": float(np.std(all_voltage_mins)) if all_voltage_mins else 0.0,
            "p5": float(np.percentile(all_voltage_mins, 5)) if all_voltage_mins else 0.0,
            "p10": float(np.percentile(all_voltage_mins, 10)) if all_voltage_mins else 0.0,
            "min": float(np.min(all_voltage_mins)) if all_voltage_mins else 0.0,
        },
        "voltage_mean": {
            "mean": float(np.mean(all_voltage_means)) if all_voltage_means else 0.0,
            "std": float(np.std(all_voltage_means)) if all_voltage_means else 0.0,
        },
        "stft": {
            "mean": float(np.mean(all_stft_values)) if all_stft_values else 0.0,
            "std": float(np.std(all_stft_values)) if all_stft_values else 0.0,
            "p5": float(np.percentile(all_stft_values, 5)) if all_stft_values else 0.0,
            "p10": float(np.percentile(all_stft_values, 10)) if all_stft_values else 0.0,
        },
        "lambda": {
            "mean": float(np.mean(all_lambda_values)) if all_lambda_values else 0.0,
            "std": float(np.std(all_lambda_values)) if all_lambda_values else 0.0,
            "p90": float(np.percentile(all_lambda_values, 90)) if all_lambda_values else 0.0,
            "p95": float(np.percentile(all_lambda_values, 95)) if all_lambda_values else 0.0,
        }
    }
    
    return results


def print_threshold_justification(results: Dict):
    """Print analysis with threshold recommendations."""
    
    print("=" * 70)
    print("THRESHOLD JUSTIFICATION FROM REFERENCE DATA")
    print("=" * 70)
    
    print("\n1. RICH-IDLE RATIO THRESHOLD")
    print("-" * 70)
    rich = results["rich_idle_ratio"]
    print(f"Normal baseline (clean references):")
    print(f"  Mean: {rich['mean']:.3f} ({rich['mean']*100:.1f}%)")
    print(f"  Std:  {rich['std']:.3f}")
    print(f"  P95:  {rich['p95']:.3f} ({rich['p95']*100:.1f}%)")
    print(f"  P99:  {rich['p99']:.3f} ({rich['p99']*100:.1f}%)")
    print(f"  Max:  {rich['max']:.3f} ({rich['max']*100:.1f}%)")
    
    # Recommendation
    threshold = rich['p95'] + 2 * rich['std']
    print(f"\nRECOMMENDED THRESHOLD: {threshold:.3f} ({threshold*100:.1f}%)")
    print(f"Rationale: P95 + 2*std = {rich['p95']:.3f} + 2*{rich['std']:.3f}")
    print(f"Current threshold: 0.05 (5%) - {'JUSTIFIED' if threshold <= 0.05 else 'CONSERVATIVE'}")
    
    print("\n2. LOW-VOLTAGE THRESHOLD")
    print("-" * 70)
    volt = results["voltage_min"]
    print(f"Minimum voltage in clean references:")
    print(f"  Mean: {volt['mean']:.2f}V")
    print(f"  Std:  {volt['std']:.2f}V")
    print(f"  P5:   {volt['p5']:.2f}V")
    print(f"  P10:  {volt['p10']:.2f}V")
    print(f"  Min:  {volt['min']:.2f}V")
    
    threshold_v = volt['p5']
    print(f"\nRECOMMENDED THRESHOLD: {threshold_v:.2f}V")
    print(f"Rationale: P5 (5th percentile of normal minimums)")
    print(f"Current threshold: 12.0V - {'JUSTIFIED' if threshold_v >= 12.0 else 'TOO STRICT'}")
    
    print("\n3. FUEL TRIM ANALYSIS")
    print("-" * 70)
    stft = results["stft"]
    print(f"Short-term fuel trim (STFT) distribution:")
    print(f"  Mean: {stft['mean']:.1f}%")
    print(f"  Std:  {stft['std']:.1f}%")
    print(f"  P5:   {stft['p5']:.1f}%")
    print(f"  P10:  {stft['p10']:.1f}%")
    print(f"\nNegative STFT indicates rich condition (excess fuel)")
    print(f"Threshold for rich: < {stft['p5']:.1f}% (5th percentile)")
    
    print("\n4. LAMBDA SENSOR ANALYSIS")
    print("-" * 70)
    lam = results["lambda"]
    print(f"Lambda sensor voltage distribution:")
    print(f"  Mean: {lam['mean']:.2f}V")
    print(f"  Std:  {lam['std']:.2f}V")
    print(f"  P90:  {lam['p90']:.2f}V")
    print(f"  P95:  {lam['p95']:.2f}V")
    print(f"\nHigh lambda voltage indicates rich mixture")
    print(f"Threshold for rich: > {lam['p90']:.2f}V (90th percentile)")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    results = analyze_reference_distributions()
    print_threshold_justification(results)
    
    # Save results
    output_path = PROJECT_ROOT / "artifacts" / "eda_threshold_analysis.json"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
