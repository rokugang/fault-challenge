from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from .data.loader import DataLoader
from .features.engineering import FeatureEngineer, export_features
from .logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class PipelineConfig:
    reference_path: Path
    output_dir: Path
    save_features: bool = True


class ETLFeaturePipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.loader = DataLoader()
        self.engineer = FeatureEngineer()

    def run(self) -> pd.DataFrame:
        logger.info("Starting pipeline for %s", self.config.reference_path)
        load_result = self.loader.load_reference_file(self.config.reference_path)
        enriched = self.engineer.transform(load_result.numeric)
        if self.config.save_features:
            out_path = self.config.output_dir / f"{self.config.reference_path.stem}_features.csv"
            export_features(enriched, out_path)
        return enriched


def run_pipeline(reference_path: Path, output_dir: Optional[Path] = None, save_features: bool = True) -> pd.DataFrame:
    output = output_dir or Path("artifacts/features")
    config = PipelineConfig(reference_path=reference_path, output_dir=output, save_features=save_features)
    pipeline = ETLFeaturePipeline(config)
    return pipeline.run()
