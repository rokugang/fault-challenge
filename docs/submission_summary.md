# Submission Notes
**Rohit Gangupantulu**

## What I Built

Spent 8 hours building a fault detector for car engines. The task: find rich air-fuel mixture at idle + low battery voltage in OBD-II scanner logs.

**ETL was 60% of the work**:
- Real scanner data is filthy. Units embedded in values, Brazilian commas, missing columns
- Built robust loader that strips units, converts locales, validates coverage
- Half the "reference" files failed quality checks—documented which ones to exclude

**Features**:
- Engineered ~40 signals from raw OBD-II parameters
- Key insight: combine STFT + lambda sensor + RPM to isolate rich-idle condition
- Rolling windows and EMA to smooth out sensor noise

**Detection evolved** (rules → ML → ensemble):
1. Started with thresholds (5% rich-idle, <12V battery)
2. Justified them with EDA on 9 clean reference files
3. Trained 4 ML models (IsolationForest won)
4. Built ensemble voting + confidence scoring
5. Added production wrapper (fallback, monitoring, circuit breaker)

**Interfaces**:
- CLI for quick testing
- FastAPI for deployments  
- Both work, tested end-to-end

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

## The ML Journey

**Step 1: Justify thresholds with data** (not gut feel):
- Analyzed 24k samples from 9 clean files
- Found normal cars show 76.6% idle frames on average (P95=99.9%!)
- My 5% threshold is ultra-conservative
- STFT normal range: -4% to +2%, so -8% catches real deviations
- Lambda P90=0.76V, threshold 0.8V captures 95th percentile

**Step 2: Train models, compare honestly**:
- IsolationForest: 1.7s training, works great
- OneClassSVM: 23s training, too slow to scale
- LOF: 4-7s, memory-heavy for large fleets
- Mahalanobis: 0.2s, but assumes Gaussian (doesn't hold)

Winner: IsolationForest at 5% contamination.

**Step 3: Build ensemble for robustness**:
- Vote across 3 models (IF, LOF, Mahalanobis)
- Weight them (50%, 30%, 20%)
- Add feature attribution (which sensors flagged anomaly)
- Output confidence level so humans can triage

**Step 4: Production-ize it**:
- Circuit breaker (if error rate >50%, use fallback)
- Input validation (reject empty/garbage)
- Monitoring (track prediction times, error rates)
- Graceful degradation if ML fails

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
