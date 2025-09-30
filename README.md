# Automotive Fault Detection

Built for the Doutor-IE ML challenge. This detects rich air-fuel mixture at idle plus low battery voltage using OBD-II scanner data from Brazilian vehicles.

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

### Data Loading
`src/data/loader.py` reads CSVs, strips units (`°C`, `%`, `V`), converts Brazilian decimal commas to dots, and validates context columns. Rejects logs with <99% coverage or non-zero trouble codes.

### Features
`src/features/engineering.py` builds:
- Rich-idle score (STFT ≤ -8%, lambda ≥ 0.8V, idle RPM)
- Low-battery score (voltage < 12V)
- Rolling averages and EMA for MAP, RPM, load

### Detection
`src/detection/detector.py` flags faults when:
- Rich idle ≥5% of frames
- Low voltage ≤12V min OR ≥5% affected frames
- Both present = fault

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

## What Could Be Better

Current approach uses rule-based thresholds, not actual ML. To improve, what I will do as next step:
- Train isolation forest or autoencoder on clean reference features
- Add temporal windows to catch transient faults better
- Tune thresholds based on false positive rate from production data

## Challenge Requirements

Meets functional requirements:
1. Outputs fault detection results
2. Uses reference dataset as baseline
3. ETL cleans and validates data
4. Detects deviations (rule-based)
5. Identifies induced fault case
6. CLI + API interfaces
