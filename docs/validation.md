# Data Validation Checklist

## Reference Quality Check

Run `python scripts/run_verifications.py` to check coverage and trouble codes.

Context columns need ≥99% coverage: `CTS`, `Carga calculada do motor`, `RPM`, `Altitude`. Also reject any logs with `Nº de falhas na memória` > 0.

Files that fail:
  - `20250611T153319273-LOG-REFERENCIA.csv` – Carga calculada coverage 14.5%
  - `20250613T083808785-LOG-REFERENCIA.csv` – Non-zero trouble codes (100%)
  - `20250630T122010818-LOG-REFERENCIA.csv` – CTS coverage 98.2%
  - `20250704T171544935-LOG-REFERENCIA.csv` – Carga calculada coverage 92.6%
  - `20250705T115626369-LOG-REFERENCIA.csv` – Non-zero trouble codes (100%)
  - `20250707T112156935-LOG-REFERENCIA.csv` – Carga calculada coverage 0%
  - `20250707T154228985-LOG-REFERENCIA.csv` – Carga calculada coverage 58.0%
  - `20250805T180208949-LOG-REFERENCIA.csv` – Carga calculada coverage 0%
Only use the 8 clean files (PASS + zero trouble codes) for training.

## Fault Example Check

`datasets/fault_example.csv` has two separate segments:
- 11 rich-idle frames (STFT ≤ -10%, lambda ≥ 0.8V, idle RPM)
- 23 low-voltage frames (< 12V)
- No overlap—they occur at different times

Test: `python -m src.cli detect datasets/fault_example.csv`
Expected: `fault_detected: true`

## Manual Checks

Profile a reference:
```bash
python -m src.cli profile-reference datasets/references/20250610T162905774-LOG-REFERENCIA.csv
```

Check detection on fault log:
```bash
python -m src.cli detect datasets/fault_example.csv
```

Run full verification:
```bash
python scripts/run_verifications.py
```

Test API (optional):
```bash
python -m src.cli serve
curl -F "file=@datasets/fault_example.csv" http://localhost:8000/detect
```

## Artifacts

- Feature CSVs: `artifacts/features/`
- Profiling JSON: `artifacts/profiling/`
