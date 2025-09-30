"""Quick test to debug ensemble feature attribution."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.data.loader import DataLoader
from src.features.engineering import FeatureEngineer
from src.ml.ensemble_detector import EnsembleDetector
from src import config

# Load data
loader = DataLoader()
result = loader.load_reference_file(config.DATA_ROOT / "fault_example.csv")
df = result.numeric

# Engineer features
engineer = FeatureEngineer()
df_features = engineer.transform(df)

print(f"Features shape: {df_features.shape}")
print(f"Columns: {len(df_features.columns)}")

# Run ensemble detector
models_dir = config.ARTIFACTS_DIR / "ml_models"
detector = EnsembleDetector(models_dir=models_dir, use_shap=True)

print("\n" + "="*80)
print("Running ensemble detection...")
print("="*80)

result = detector.detect_with_explanation(df_features)

print("\n" + "="*80)
print("RESULT")
print("="*80)
print(f"Fault detected: {result.fault_detected}")
print(f"Confidence: {result.confidence}")
print(f"Top anomaly features: {len(result.top_anomaly_features) if result.top_anomaly_features else 0}")

if result.top_anomaly_features:
    print("\nTop 5 anomalous features:")
    for feat, score in result.top_anomaly_features[:5]:
        print(f"  {feat}: {score:.3f}")
else:
    print("\nNO FEATURES RETURNED!")
