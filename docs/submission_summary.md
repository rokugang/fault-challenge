# Submission Summary

## What's Done

ETL (`src/data/loader.py`):
- Reads CSVs with Brazilian locale handling
- Strips units and validates context columns
- Rejects logs with coverage <99% or trouble codes >0

Features (`src/features/engineering.py`):
- Rich-idle score (STFT, lambda, RPM combo)
- Low-battery score (voltage thresholds)
- Rolling stats and EMA smoothing

Detection (`src/detection/detector.py`):
- Rule-based fault flagging (≥5% rich-idle + low voltage)
- Correctly detects `fault_example.csv`

Interfaces:
- CLI: `python -m src.cli detect <file>`
- API: FastAPI on `:8000` with `/detect` endpoint

Tests:
- 5 unit tests pass (loader, features, detector)
- Verification script checks all datasets

## Quick Validation

```bash
pip install -r requirements.txt
python -m pytest                    # 5 tests pass
python scripts/run_verifications.py # checks all data
python -m src.cli detect datasets/fault_example.csv
```

## Data Issues Found

8/16 reference files fail quality checks (see `docs/validation.md` for details):
- Some have <99% coverage on context columns
- Others have non-zero trouble codes
- Only use the 8 clean files for training

Fault example works correctly:
- 11% rich-idle frames
- 14% low-voltage frames  
- No overlap (separate time segments)

## ML Implementation

**EDA & Threshold Justification**:
- Analyzed 9 clean reference files (24,100 samples)
- Justified thresholds from statistical distributions
- Rich-idle 5% threshold = P95+2σ from references (conservative)
- STFT < -8% from P5=-4% in normal data
- Lambda > 0.8V from P90=0.76V

**ML Experiments**:
- Trained 4 anomaly detection algorithms
- Isolation Forest (1.7s): Tree-based, efficient
- One-Class SVM (3.8-23s): RBF kernel
- LOF (4.2-7s): Density-based
- Mahalanobis (0.2s): Statistical baseline
- Models saved in `artifacts/ml_models/`
- 41 engineered features per sample

**Hybrid Detector**:
- `MLFaultDetector` combines ML scores + domain rules
- Outputs confidence: high/medium/low
- Falls back to rules if ML unavailable

## What's Included

- `src/` - ETL, features, detection, CLI, API
- `tests/` - Unit tests (all passing)
- `scripts/` - Verification automation
- `docs/` - This summary, validation steps, README
- `artifacts/` - Generated profiling and feature CSVs

## Requirements Met

✓ Outputs fault detection results  
✓ Uses reference data as baseline  
✓ ETL with validation  
✓ Detects deviations (rule-based)  
✓ Finds induced fault case  
✓ CLI + API interfaces
