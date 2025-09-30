"""
Run ML Experiments

Trains and compares multiple anomaly detection algorithms on clean reference data.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.data.loader import DataLoader
from src.features.engineering import FeatureEngineer
from src.ml.experiments import MLExperiments


def main():
    print("Loading clean reference datasets...")
    
    loader = DataLoader()
    engineer = FeatureEngineer()
    
    # Load all clean reference files
    reference_dir = config.DATA_ROOT / "references"
    reference_files = sorted(reference_dir.glob("*.csv"))
    
    all_features = []
    clean_count = 0
    
    for ref_file in reference_files:
        try:
            load_result = loader.load_reference_file(ref_file)
            df = load_result.numeric
            if len(df) < 10:
                continue
            
            # Engineer features
            df_features = engineer.transform(df)
            all_features.append(df_features)
            clean_count += 1
            print(f"  OK {ref_file.name}: {len(df)} rows")
            
        except Exception as e:
            print(f"  SKIP {ref_file.name}: {str(e)[:50]}")
            continue
    
    if not all_features:
        print("No clean reference files found!")
        return
    
    print(f"\nCombined {clean_count} reference files for training")
    
    # Combine all reference features
    import pandas as pd
    combined_features = pd.concat(all_features, ignore_index=True)
    
    print(f"Total samples: {len(combined_features)}")
    print(f"Total features: {len(combined_features.columns)}")
    
    # Run experiments
    experiments = MLExperiments(output_dir=config.ARTIFACTS_DIR / "ml_models")
    
    X = experiments.prepare_features(combined_features)
    print(f"\nPrepared feature matrix: {X.shape}")
    
    # Try different contamination rates
    contaminations = [0.01, 0.05, 0.10]  # 1%, 5%, 10%
    
    results = experiments.run_all_experiments(X, contaminations=contaminations)
    
    # Print comparison
    experiments.print_results()
    
    # Save results
    experiments.save_results()
    
    print("\nDone! Models saved to artifacts/ml_models/")


if __name__ == "__main__":
    main()
