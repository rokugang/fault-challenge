# Interactive Dashboard

Streamlit web interface for real-time fault detection and analysis.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Launch dashboard
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`

## Features

### 1. File Upload
- Drag-and-drop CSV upload interface
- Real-time validation and processing
- Supports standard OBD-II scanner format

### 2. Detection Analysis
- Ensemble ML detection with confidence scoring
- Real-time fault status display
- Detailed metrics (rich-idle ratio, voltage min, anomaly rate)
- Detection reasoning explanation

### 3. Visualizations
- **Timeline Plot**: Rich-idle score and voltage over time
- **Feature Attribution**: SHAP/z-score importance for top anomalous sensors
- **Distribution Analysis**: STFT and lambda sensor histograms

### 4. Configuration Options
- Toggle ensemble vs single-model detection
- Enable/disable temporal windowing
- SHAP attribution on/off

### 5. Data Inspection
- Raw data preview (first 50 frames)
- Engineered features preview
- Interactive dataframes

## Example Usage

1. Launch dashboard: `streamlit run app.py`
2. Upload `datasets/fault_example.csv`
3. View detection result (should show FAULT DETECTED)
4. Examine timeline to see rich-idle spike and low voltage
5. Check feature attribution to identify problematic sensors

## Architecture

```python
Upload CSV 
  → Data Validation (src/data/loader.py)
  → Feature Engineering (src/features/engineering.py)  
  → Ensemble Detection (src/ml/ensemble_detector.py)
  → Visualization (matplotlib/seaborn)
  → Interactive Display (streamlit)
```

## Performance

- Loads 1000-frame log in ~2 seconds
- Feature engineering: ~1 second
- ML detection: ~0.5 seconds (ensemble)
- Total latency: <4 seconds for complete analysis

## Technical Details

**Models Used**:
- Isolation Forest (50% weight)
- Local Outlier Factor (30% weight)  
- Mahalanobis Distance (20% weight)

**Explainability**:
- SHAP TreeExplainer for exact Shapley values
- Z-score fallback if SHAP unavailable
- Top 10 most anomalous features displayed

**Visualization**:
- Matplotlib for time series and distributions
- Seaborn for statistical plots
- Custom styling with professional color palette

## Deployment

### Local (Recommended)
```bash
streamlit run app.py
```

### Docker
```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

## Limitations

- Requires trained models in `artifacts/ml_models/`
- SHAP visualization requires SHAP library installed
- Large files (>10k frames) may take longer to process
- Browser-based (not suitable for embedded systems)
