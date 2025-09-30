from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from ..logging_utils import get_logger

logger = get_logger(__name__)

STFT_COL = "Ajuste de combustível de curto prazo - Banco 1"
STFT_SENSOR_COL = "Ajuste de combustível de curto prazo - Banco 1, sensor 1"
LAMBDA_COL = "Sonda lambda - Banco 1, sensor 1"
MAP_COL = "Pressão no coletor de admissão - MAP"
RPM_COL = "Rotação do motor - RPM"
VOLTAGE_COL = "Tensão do módulo"
LOAD_COL = "Carga calculada do motor"


@dataclass
class FeatureEngineer:
    rolling_window: int = 5
    ema_span: int = 15

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self._check_input(df)
        features = pd.DataFrame(index=df.index)

        if STFT_COL in df:
            stft = df[STFT_COL]
            features["stft_pct"] = stft
            features["stft_rolling_mean"] = stft.rolling(self.rolling_window, min_periods=1).mean()
            features["stft_rolling_std"] = stft.rolling(self.rolling_window, min_periods=1).std(ddof=0)
            features["stft_ema"] = stft.ewm(span=self.ema_span, adjust=False).mean()
            features["stft_low_flag"] = (stft <= -8).astype(int)

        if STFT_SENSOR_COL in df and STFT_COL not in df:
            stft_sensor = df[STFT_SENSOR_COL]
            features["stft_sensor_pct"] = stft_sensor
            features["stft_sensor_low_flag"] = (stft_sensor <= -8).astype(int)

        if LAMBDA_COL in df:
            lambda_v = df[LAMBDA_COL]
            features["lambda_v"] = lambda_v
            features["lambda_rolling_mean"] = lambda_v.rolling(self.rolling_window, min_periods=1).mean()
            features["lambda_rich_flag"] = (lambda_v >= 0.8).astype(int)

        if MAP_COL in df:
            map_val = df[MAP_COL]
            features["map_mbar"] = map_val
            features["map_rolling_mean"] = map_val.rolling(self.rolling_window, min_periods=1).mean()
            features["map_rolling_std"] = map_val.rolling(self.rolling_window, min_periods=1).std(ddof=0)

        if RPM_COL in df:
            rpm = df[RPM_COL]
            features["rpm"] = rpm
            features["is_idle"] = (rpm <= 1000).astype(int)
            rpm_diff = rpm.diff().fillna(0)
            features["rpm_delta"] = rpm_diff
            features["rpm_jerk"] = rpm_diff.diff().fillna(0)

        if VOLTAGE_COL in df:
            voltage = df[VOLTAGE_COL]
            features["module_voltage"] = voltage
            features["voltage_low_flag"] = (voltage < 12.2).astype(int)
            features["voltage_rolling_min"] = voltage.rolling(self.rolling_window, min_periods=1).min()

        if LOAD_COL in df:
            load = df[LOAD_COL]
            features["calc_load_pct"] = load
            features["calc_load_rolling_mean"] = load.rolling(self.rolling_window, min_periods=1).mean()

        if STFT_COL in df and LAMBDA_COL in df and RPM_COL in df:
            features["rich_idle_score"] = (
                features.get("stft_low_flag", 0)
                + features.get("lambda_rich_flag", 0)
                + features.get("is_idle", 0)
            )

        if VOLTAGE_COL in df:
            features["low_battery_score"] = (
                features.get("voltage_low_flag", 0)
                + (features.get("voltage_rolling_min", 0) < 12.2).astype(int)
            )

        result = pd.concat([df, features], axis=1)
        logger.info("Generated %d engineered features", result.shape[1] - df.shape[1])
        return result

    def _check_input(self, df: pd.DataFrame) -> None:
        if df.empty:
            raise ValueError("Input DataFrame is empty")
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")


def export_features(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Persisted features to %s", path)
