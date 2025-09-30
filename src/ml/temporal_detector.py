"""
Temporal Window-Based Fault Detection

Aggregates fault indicators across time windows to reduce false positives.
Author: Rohit Gangupantulu
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np
import pandas as pd

from src.detection.detector import FaultDetector, DetectionResult


@dataclass
class TemporalWindow:
    """Results for a single time window."""
    start_idx: int
    end_idx: int
    duration_sec: float
    fault_detected: bool
    rich_ratio: float
    voltage_min: float
    confidence: str


class TemporalFaultDetector(FaultDetector):
    """
    Fault detector with temporal window aggregation.
    
    Innovation: Instead of treating each frame independently, aggregates
    fault indicators over 30-second windows to catch intermittent faults
    and reduce noise-induced false positives.
    
    Args:
        window_size_sec: Window duration in seconds (default 30s)
        overlap_ratio: Overlap between consecutive windows (default 0.5 = 50%)
        min_windows_fault: Minimum windows that must show fault (default 2)
    """
    
    def __init__(
        self,
        window_size_sec: float = 30.0,
        overlap_ratio: float = 0.5,
        min_windows_fault: int = 2
    ):
        super().__init__()
        self.window_size_sec = window_size_sec
        self.overlap_ratio = overlap_ratio
        self.min_windows_fault = min_windows_fault
    
    def _estimate_sampling_rate(self, df: pd.DataFrame) -> float:
        """
        Estimate sampling rate from data or row frequency.
        
        Method 1: If timestamps exist, calculate from time deltas
        Method 2: If no timestamps, estimate from row count (assume typical OBD-II scan duration)
        Fallback: 1 Hz (conservative for OBD-II)
        """
        # Try timestamp-based estimation
        if "timestamp" in df.columns:
            try:
                timestamps = pd.to_datetime(df["timestamp"])
                time_diffs = timestamps.diff().dt.total_seconds().dropna()
                if len(time_diffs) > 0:
                    median_diff = time_diffs.median()
                    if median_diff > 0:
                        return 1.0 / median_diff
            except:
                pass
        
        # Try heuristic: OBD-II scans usually 60-300 seconds
        # If we have many samples, assume faster sampling
        n_samples = len(df)
        if n_samples > 300:
            # Likely 2-5 Hz for long scans
            return 3.0
        elif n_samples > 100:
            # Likely 1-2 Hz for medium scans
            return 1.5
        
        # Conservative fallback: 1 Hz
        return 1.0
    
    def _create_windows(
        self, 
        df: pd.DataFrame, 
        sampling_rate: float
    ) -> List[Tuple[int, int]]:
        """
        Create overlapping time windows.
        
        Returns:
            List of (start_idx, end_idx) tuples
        """
        window_size_frames = int(self.window_size_sec * sampling_rate)
        step_size = int(window_size_frames * (1 - self.overlap_ratio))
        
        windows = []
        n_frames = len(df)
        
        start_idx = 0
        while start_idx < n_frames:
            end_idx = min(start_idx + window_size_frames, n_frames)
            windows.append((start_idx, end_idx))
            
            if end_idx >= n_frames:
                break
            
            start_idx += step_size
        
        return windows
    
    def _evaluate_window(
        self, 
        df: pd.DataFrame, 
        start_idx: int, 
        end_idx: int
    ) -> TemporalWindow:
        """Evaluate fault detection for a single window."""
        window_df = df.iloc[start_idx:end_idx]
        
        # Run standard detection on window
        result = super().evaluate(window_df)
        
        # Extract metrics
        rich_ratio = result.metrics.get("rich_idle_ratio", 0.0)
        voltage_min = result.metrics.get("low_voltage_min", float('inf'))
        
        # Determine confidence based on strength of signals
        if rich_ratio > 0.10 and voltage_min < 11.5:
            confidence = "high"
        elif rich_ratio > 0.05 or voltage_min < 12.0:
            confidence = "medium"
        else:
            confidence = "low"
        
        duration = (end_idx - start_idx) / self._estimate_sampling_rate(df)
        
        return TemporalWindow(
            start_idx=start_idx,
            end_idx=end_idx,
            duration_sec=duration,
            fault_detected=result.fault_detected,
            rich_ratio=rich_ratio,
            voltage_min=voltage_min,
            confidence=confidence
        )
    
    def evaluate_temporal(self, df: pd.DataFrame) -> Dict:
        """
        Evaluate with temporal windowing.
        
        Returns:
            Dictionary with:
            - fault_detected: Overall fault decision
            - windows: List of TemporalWindow results
            - fault_window_count: Number of windows showing fault
            - total_windows: Total number of windows
            - reasons: List of explanations
        """
        sampling_rate = self._estimate_sampling_rate(df)
        windows = self._create_windows(df, sampling_rate)
        
        # Evaluate each window
        window_results = [
            self._evaluate_window(df, start, end)
            for start, end in windows
        ]
        
        # Count fault windows
        fault_windows = [w for w in window_results if w.fault_detected]
        fault_count = len(fault_windows)
        total_count = len(window_results)
        
        # Overall decision: require min_windows_fault to trigger
        overall_fault = fault_count >= self.min_windows_fault
        
        # Build reasons
        reasons = []
        if overall_fault:
            reasons.append(
                f"Temporal fault detected: {fault_count}/{total_count} windows show fault pattern"
            )
            
            # Highlight strongest windows
            fault_windows.sort(key=lambda w: w.rich_ratio + (1.0 if w.voltage_min < 12 else 0), reverse=True)
            for i, w in enumerate(fault_windows[:3]):
                reasons.append(
                    f"  Window {i+1}: frames {w.start_idx}-{w.end_idx}, "
                    f"rich={w.rich_ratio:.1%}, voltage={w.voltage_min:.1f}V"
                )
        else:
            reasons.append(
                f"No sustained fault: only {fault_count}/{total_count} windows flagged "
                f"(need ≥{self.min_windows_fault})"
            )
        
        # Aggregate metrics
        if fault_windows:
            avg_rich = np.mean([w.rich_ratio for w in fault_windows])
            min_voltage = min([w.voltage_min for w in fault_windows])
        else:
            avg_rich = 0.0
            min_voltage = float('inf')
        
        return {
            "fault_detected": overall_fault,
            "windows": window_results,
            "fault_window_count": fault_count,
            "total_windows": total_count,
            "reasons": reasons,
            "metrics": {
                "avg_rich_ratio_in_fault_windows": avg_rich,
                "min_voltage_across_fault_windows": min_voltage,
                "fault_window_ratio": fault_count / total_count if total_count > 0 else 0.0
            }
        }
    
    def evaluate(self, df: pd.DataFrame) -> DetectionResult:
        """
        Override evaluate() to use temporal windows.
        
        Returns standard DetectionResult for compatibility.
        """
        temporal_result = self.evaluate_temporal(df)
        
        return DetectionResult(
            fault_detected=temporal_result["fault_detected"],
            reasons=temporal_result["reasons"],
            metrics=temporal_result["metrics"]
        )
