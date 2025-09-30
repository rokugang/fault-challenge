"""
Production utilities for robust ML deployment.

Author: Rohit Gangupantulu
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime
import json

import pandas as pd
import numpy as np

from ..logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class HealthCheck:
    """Model health check result."""
    model_loaded: bool
    model_path: Optional[str]
    last_prediction_time: Optional[float]
    predictions_count: int
    error_count: int
    avg_prediction_time_ms: float


class ProductionWrapper:
    """
    Wraps ML detector with production-grade error handling, monitoring, and fallback.
    
    Features:
    - Graceful degradation if model fails
    - Prediction time monitoring
    - Error rate tracking
    - Input validation
    - Automatic fallback to rules
    """
    
    def __init__(self, detector, fallback_detector=None):
        self.detector = detector
        self.fallback_detector = fallback_detector
        
        # Monitoring
        self.predictions_count = 0
        self.error_count = 0
        self.prediction_times = []
        self.last_prediction_time = None
        
        # Circuit breaker (if error rate > 50%, use fallback)
        self.circuit_breaker_threshold = 0.5
        self.circuit_open = False
    
    def _validate_input(self, df: pd.DataFrame) -> tuple[bool, Optional[str]]:
        """Validate input before prediction."""
        if df is None or df.empty:
            return False, "Empty dataframe"
        
        if len(df) < 3:
            return False, f"Insufficient data ({len(df)} rows, need ≥3)"
        
        # Check for all NaN
        if df.isna().all().all():
            return False, "All values are NaN"
        
        return True, None
    
    def _should_use_fallback(self) -> bool:
        """Check if circuit breaker is open."""
        if self.predictions_count < 10:
            return False  # Not enough data
        
        error_rate = self.error_count / self.predictions_count
        return error_rate > self.circuit_breaker_threshold
    
    def predict(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Production-safe prediction with monitoring and fallback.
        """
        start_time = time.time()
        self.predictions_count += 1
        
        # Validate input
        valid, error_msg = self._validate_input(df)
        if not valid:
            logger.warning(f"Input validation failed: {error_msg}")
            return {
                "fault_detected": False,
                "error": error_msg,
                "fallback_used": False,
                "confidence": "none"
            }
        
        # Check circuit breaker
        use_fallback = self._should_use_fallback()
        
        try:
            # Try primary detector
            if not use_fallback:
                result = self.detector.run_detection(df)
                prediction_time = (time.time() - start_time) * 1000  # ms
                
                self.prediction_times.append(prediction_time)
                self.last_prediction_time = time.time()
                
                # Keep only last 100 times for avg
                if len(self.prediction_times) > 100:
                    self.prediction_times.pop(0)
                
                return {
                    "fault_detected": result.fault_detected,
                    "reasons": result.reasons,
                    "metrics": result.metrics,
                    "fallback_used": False,
                    "prediction_time_ms": prediction_time,
                    "confidence": result.metrics.get("confidence", "unknown")
                }
            else:
                logger.warning(f"Circuit breaker open (error rate {self.error_count}/{self.predictions_count}), using fallback")
                raise Exception("Circuit breaker open")
        
        except Exception as e:
            logger.error(f"Primary detector failed: {e}")
            self.error_count += 1
            
            # Try fallback
            if self.fallback_detector:
                try:
                    result = self.fallback_detector.run_detection(df)
                    prediction_time = (time.time() - start_time) * 1000
                    
                    logger.info("Fallback detector succeeded")
                    return {
                        "fault_detected": result.fault_detected,
                        "reasons": result.reasons + ["(fallback detector used)"],
                        "metrics": result.metrics,
                        "fallback_used": True,
                        "prediction_time_ms": prediction_time,
                        "confidence": "degraded"
                    }
                except Exception as fallback_error:
                    logger.error(f"Fallback also failed: {fallback_error}")
            
            # Complete failure
            return {
                "fault_detected": False,
                "error": str(e),
                "fallback_used": False,
                "confidence": "error"
            }
    
    def health_check(self) -> HealthCheck:
        """Get system health metrics."""
        model_loaded = self.detector is not None
        model_path = None
        
        if hasattr(self.detector, 'model') and self.detector.model:
            model_path = "loaded"
        
        avg_time = np.mean(self.prediction_times) if self.prediction_times else 0.0
        
        return HealthCheck(
            model_loaded=model_loaded,
            model_path=model_path,
            last_prediction_time=self.last_prediction_time,
            predictions_count=self.predictions_count,
            error_count=self.error_count,
            avg_prediction_time_ms=avg_time
        )
    
    def reset_circuit_breaker(self):
        """Manually reset circuit breaker."""
        self.error_count = 0
        self.predictions_count = 0
        self.circuit_open = False
        logger.info("Circuit breaker reset")
    
    def export_metrics(self, output_path: Path):
        """Export monitoring metrics to JSON."""
        health = self.health_check()
        
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "model_loaded": health.model_loaded,
            "predictions_count": health.predictions_count,
            "error_count": health.error_count,
            "error_rate": health.error_count / max(health.predictions_count, 1),
            "avg_prediction_time_ms": health.avg_prediction_time_ms,
            "last_prediction": datetime.fromtimestamp(health.last_prediction_time).isoformat() if health.last_prediction_time else None,
            "circuit_breaker_open": self.circuit_open
        }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Metrics exported to {output_path}")
