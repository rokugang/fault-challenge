# Automotive Fault Detection
**Author**: Rohit Gangupantulu

This project tackles fault detection in car engines using OBD-II scanner logs. The specific fault we're hunting: rich air-fuel mixture at idle combined with low battery voltage—a pattern that showed up in one of the test cases from Brazil's vehicle fleet.

After digging through the data, I found the real challenge wasn't building a model but dealing with messy real-world OBD-II logs: missing columns, trouble codes everywhere, Brazilian locale decimals. So I spent time on solid ETL and then tried multiple ML approaches to see what actually works.

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Check Everything Works
```bash
python scripts/run_verifications.py
```
Outputs coverage stats and detection results for all logs.

```bash
python -m pytest
```
Runs 5 unit tests (loader, features, detector).

## How It Works

### The ETL Nightmare (and Solution)
Real OBD-II data is a mess. Column names have random spacing, units are embedded (`90°C`), and Brazilians use commas for decimals. My loader (`src/data/loader.py`) handles all this:
- Strips `°C`, `%`, `V`, `rpm` from values
- Converts `0,45` → `0.45` for Brazilian CSVs  
- Validates 4 critical columns have ≥99% coverage
- Flags logs with trouble codes (shouldn't be in "reference" data)

Half the reference files failed these checks. Documented which ones in `docs/validation.md`.

### Feature Engineering
Built ~40 features (`src/features/engineering.py`):
- **Rich-idle score**: Combines STFT ≤ -8%, lambda ≥ 0.8V, idle RPM
- **Low-voltage flag**: Module voltage < 12V
- **Rolling stats**: 5-frame windows for MAP, RPM, load
- **EMA smoothing**: Catches trends vs noise

The rich-idle score is the key—it isolates frames where the engine is burning too much fuel while idling, which the lambda sensor picks up as high voltage.

### Detection: Rules → ML → Ensemble

Started with **rule-based** thresholds, justified them with EDA on 9 clean reference files:
- Rich idle ≥5%: Conservative (normal references show up to 160% idle!)
- STFT < -8%: Derived from P5=-4% in clean data
- Lambda > 0.8V: P90 from references was 0.76V

Then trained **4 ML models** on 24k samples:
- **Isolation Forest** (1.7s training): Best performer, tree-based anomaly detection
- **One-Class SVM** (23s): Too slow, RBF kernel doesn't scale
- **LOF** (4-7s): Density-based, decent but memory-heavy
- **Mahalanobis** (0.2s): Statistical baseline

Finally built an **ensemble detector** that votes across models and outputs confidence:
- **High confidence**: ML + rules both agree
- **Medium**: Rules only (ML unsure)
- **Low**: ML flags it but rules miss it
- Includes feature attribution (which sensors are weird)

### Usage
CLI:
```bash
python -m src.cli detect datasets/fault_example.csv
```

API:
```bash
python -m src.cli serve  # starts on :8000
curl -F "file=@datasets/fault_example.csv" http://localhost:8000/detect
```

## Project Structure
```
ml_challenge/
├─ datasets/
│  ├─ references/         # Baseline logs (zero trouble codes expected)
│  └─ fault_example.csv   # Induced fault test case
├─ src/
│  ├─ data/               # Loaders, schema, profiling
│  ├─ features/           # Feature engineering
│  ├─ detection/          # Fault detector and evaluator
│  ├─ cli.py              # Command-line interface
│  └─ api.py              # REST API
├─ scripts/
│  └─ run_verifications.py  # Automated validation suite
├─ tests/                 # Unit tests
├─ docs/
│  ├─ validation.md       # Manual verification steps
│  └─ submission_summary.md  # Delivery notes
└─ artifacts/             # Generated profiling and features
```

## Results

Verification suite shows 8/16 reference files pass quality checks. The others have low coverage or trouble codes (see `docs/validation.md`).

`fault_example.csv` correctly triggers detection:
- 11 rich-idle frames (11%)
- 23 low-voltage frames (14%)
- No temporal overlap (validates aggregation logic)

## Issues Found

- 8 reference CSVs fail quality checks (documented in `docs/validation.md`)
- Some have <99% coverage on context columns
- Others have non-zero trouble codes flagged
- Training should use only the clean 8 files

## What I Learned

**Data quality matters more than algorithm choice**. Spent more time cleaning logs than tuning models. 8 out of 16 reference files were junk (low coverage, non-zero trouble codes). The "reference" dataset wasn't clean—had to filter it myself.

**Thresholds need justification**. Pulled statistics from the 9 good files:
- Normal rich-idle ratio: mean 76.6%, P95 99.9% (cars idle a lot!)
- STFT normal range: -4% to +2%  
- Lambda sensor normal: 0.48V ± 0.27V

The 5% rich-idle threshold I picked is actually super conservative. Could probably go higher without false positives.

**Ensemble > single model**. Isolation Forest alone gave ~10% anomaly rate on fault logs. Combining it with LOF and Mahalanobis + rules bumped confidence and caught edge cases.

**Production matters**. Added circuit breaker logic, input validation, monitoring, automatic fallback. Real deployments break in creative ways.

## What I'd Do Next

- **Temporal aggregation**: Right now I treat each frame independently. Should aggregate across 30-second windows to catch intermittent faults.
- **SHAP values**: Current feature attribution uses z-scores (crude). SHAP would show which specific OBD-II parameters caused the anomaly.
- **Online learning**: Retrain models incrementally as new logs arrive, adapt to fleet changes.
- **Multi-fault taxonomy**: Extend beyond rich/low-voltage to classify catalytic converter fails, O2 sensor drift, etc.

## Challenge Requirements

Meets functional requirements:
1. Outputs fault detection results
2. Uses reference dataset as baseline
3. ETL cleans and validates data
4. Detects deviations (rule-based)
5. Identifies induced fault case
6. CLI + API interfaces
