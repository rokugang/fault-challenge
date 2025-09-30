from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile

from .detection.evaluation import evaluate_file
from .logging_utils import get_logger

app = FastAPI(title="Doutor Diagnostics API", version="0.1.0")
logger = get_logger(__name__)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/detect")
async def detect(file: UploadFile = File(...)) -> dict:
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    try:
        contents = await file.read()
        temp_dir = Path(tempfile.mkdtemp())
        temp_path = temp_dir / file.filename
        temp_path.write_bytes(contents)
        logger.info("Received file %s for detection", file.filename)
        result = evaluate_file(temp_path, save_features=False)
    except Exception as exc:  # pragma: no cover
        logger.exception("Detection failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        try:
            if "temp_path" in locals() and temp_path.exists():
                temp_path.unlink(missing_ok=True)
            if "temp_dir" in locals() and temp_dir.exists():
                temp_dir.rmdir()
        except OSError:
            pass

    return result
