"""
Streamlit Dashboard for Automotive Fault Detection System

Interactive web interface for OBD-II log analysis and fault detection.
Author: Rohit Gangupantulu
"""
from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import DataLoader
from src.features.engineering import FeatureEngineer
from src.ml.ensemble_detector import EnsembleDetector
from src.config import ARTIFACTS_DIR

# Configure page
st.set_page_config(
    page_title="Automotive Fault Detection",
    page_icon="car",
    layout="wide"
)

# Styling
sns.set_theme(style="whitegrid")

# Title
st.title("Automotive Fault Detection System")
st.markdown("**Upload OBD-II scan log for fault analysis**")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    
    use_ensemble = st.checkbox("Use Ensemble Detector", value=True)
    use_temporal = st.checkbox("Temporal Windowing", value=False)
    show_shap = st.checkbox("Show SHAP Attribution", value=True)
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("Detects rich air-fuel mixture at idle + low battery voltage")
    st.markdown("**Models**: IsolationForest, LOF, Mahalanobis")
    st.markdown("**Author**: Rohit Gangupantulu")

# Main content
uploaded_file = st.file_uploader("Choose OBD-II CSV file", type=['csv'])

if uploaded_file is not None:
    try:
        # Load data
        with st.spinner("Loading and validating data..."):
            loader = DataLoader()
            
            # Save uploaded file temporarily
            temp_path = PROJECT_ROOT / "temp_upload.csv"
            with open(temp_path, 'wb') as f:
                f.write(uploaded_file.getvalue())
            
            # Load
            load_result = loader.load_reference_file(temp_path)
            df_raw = load_result.numeric
            
            # Clean up temp file
            temp_path.unlink()
        
        st.success(f"Loaded {len(df_raw)} frames successfully")
        
        # Feature engineering
        with st.spinner("Engineering features..."):
            engineer = FeatureEngineer()
            df_features = engineer.transform(df_raw)
        
        st.success(f"Generated {len(df_features.columns)} features")
        
        # Detection
        with st.spinner("Running fault detection..."):
            if use_ensemble:
                models_dir = ARTIFACTS_DIR / "ml_models"
                detector = EnsembleDetector(models_dir=models_dir, use_shap=show_shap)
                result = detector.detect_with_explanation(df_features)
            else:
                from src.detection.detector import FaultDetector
                detector = FaultDetector()
                result = detector.run_detection(df_features)
                # Convert to explainable result format
                from src.ml.ensemble_detector import ExplainableResult
                result = ExplainableResult(
                    fault_detected=result.fault_detected,
                    confidence="medium",
                    reasons=result.reasons,
                    metrics=result.metrics,
                    top_anomaly_features=[]
                )
        
        # Display result
        st.markdown("---")
        st.header("Detection Result")
        
        if result.fault_detected:
            st.error(f"FAULT DETECTED (Confidence: {result.confidence.upper()})")
        else:
            st.success("NO FAULT DETECTED")
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            rich_ratio = result.metrics.get('rich_idle_ratio', 0.0)
            st.metric("Rich-Idle Ratio", f"{rich_ratio:.1%}")
        
        with col2:
            voltage_min = result.metrics.get('low_voltage_min', 0.0)
            st.metric("Min Voltage", f"{voltage_min:.2f}V")
        
        with col3:
            st.metric("Confidence", result.confidence.upper())
        
        with col4:
            if 'ensemble_anomaly_ratio' in result.metrics:
                ensemble_ratio = result.metrics['ensemble_anomaly_ratio']
                st.metric("ML Anomaly Rate", f"{ensemble_ratio:.1%}")
        
        # Reasons
        st.markdown("### Detection Reasons")
        for reason in result.reasons:
            st.markdown(f"- {reason}")
        
        # Visualizations
        st.markdown("---")
        st.header("Analysis Visualizations")
        
        # Timeline plot
        st.subheader("Fault Indicator Timeline")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        
        # Rich-idle score
        if 'rich_idle_score' in df_features.columns:
            ax1.plot(df_features.index, df_features['rich_idle_score'], 
                    label='Rich-Idle Score', color='#d62728', linewidth=1.5)
            ax1.axhline(y=2, color='gray', linestyle='--', alpha=0.5, label='Threshold')
            ax1.set_ylabel('Rich-Idle Score', fontsize=10)
            ax1.legend(loc='upper right', fontsize=8)
            ax1.grid(True, alpha=0.3)
        
        # Voltage
        if 'Tensão do módulo' in df_raw.columns:
            ax2.plot(df_raw.index, df_raw['Tensão do módulo'],
                    label='Battery Voltage', color='#1f77b4', linewidth=1.5)
            ax2.axhline(y=12.0, color='gray', linestyle='--', alpha=0.5, label='Threshold (12V)')
            ax2.set_ylabel('Voltage (V)', fontsize=10)
            ax2.set_xlabel('Frame Index', fontsize=10)
            ax2.legend(loc='upper right', fontsize=8)
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # Feature importance (if available)
        if result.top_anomaly_features:
            st.subheader("Top Anomalous Features")
            
            # Extract feature names and scores
            features = [f[0][:40] for f in result.top_anomaly_features[:10]]  # Truncate names
            scores = [abs(f[1]) for f in result.top_anomaly_features[:10]]
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.barh(features, scores, color='#ff7f0e')
            ax.set_xlabel('Anomaly Contribution', fontsize=10)
            ax.set_title('Feature Attribution (SHAP or Z-Score)', fontsize=12)
            ax.grid(True, alpha=0.3, axis='x')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        # Distribution plots
        with st.expander("Feature Distributions"):
            col1, col2 = st.columns(2)
            
            with col1:
                if 'Ajuste de combustível de curto prazo - Banco 1' in df_raw.columns:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    stft = df_raw['Ajuste de combustível de curto prazo - Banco 1'].dropna()
                    ax.hist(stft, bins=30, color='#2ca02c', alpha=0.7, edgecolor='black')
                    ax.axvline(x=-8, color='red', linestyle='--', label='Threshold (-8%)')
                    ax.set_xlabel('Short-Term Fuel Trim (%)', fontsize=10)
                    ax.set_ylabel('Frequency', fontsize=10)
                    ax.set_title('STFT Distribution', fontsize=11)
                    ax.legend(fontsize=8)
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    plt.close()
            
            with col2:
                if 'Sonda lambda - Banco 1, sensor 1' in df_raw.columns:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    lambda_vals = df_raw['Sonda lambda - Banco 1, sensor 1'].dropna()
                    ax.hist(lambda_vals, bins=30, color='#9467bd', alpha=0.7, edgecolor='black')
                    ax.axvline(x=0.8, color='red', linestyle='--', label='Threshold (0.8V)')
                    ax.set_xlabel('Lambda Sensor (V)', fontsize=10)
                    ax.set_ylabel('Frequency', fontsize=10)
                    ax.set_title('Lambda Sensor Distribution', fontsize=11)
                    ax.legend(fontsize=8)
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    plt.close()
        
        # Raw data preview
        with st.expander("View Raw Data"):
            st.dataframe(df_raw.head(50), height=300)
        
        # Feature data preview
        with st.expander("View Engineered Features"):
            st.dataframe(df_features.head(50), height=300)
    
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.exception(e)

else:
    # Instructions
    st.info("Upload a CSV file to begin analysis")
    
    st.markdown("### Expected Format")
    st.markdown("""
    CSV should contain OBD-II scanner data with columns:
    - `Temperatura do líquido de arrefecimento do motor - CTS` (Coolant temp)
    - `Rotação do motor - RPM` (Engine RPM)
    - `Carga calculada do motor` (Engine load)
    - `Altitude`
    - `Ajuste de combustível de curto prazo - Banco 1` (STFT)
    - `Sonda lambda - Banco 1, sensor 1` (Lambda sensor)
    - `Tensão do módulo` (Battery voltage)
    
    **Try the example**: `datasets/fault_example.csv`
    """)
    
    st.markdown("### How It Works")
    st.markdown("""
    1. **Data Validation**: Checks column coverage and format
    2. **Feature Engineering**: Generates 41 derived features
    3. **ML Detection**: Ensemble voting across IsolationForest, LOF, Mahalanobis
    4. **Explainability**: SHAP attribution shows which sensors triggered fault
    5. **Confidence Scoring**: High/Medium/Low based on model agreement
    """)
