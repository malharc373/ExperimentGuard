"""FastAPI service exposing the ExperimentGuard decision engine over HTTP.

Endpoints:
    GET  /health              -> liveness probe
    POST /analyze             -> analyse a CSV upload, return JSON verdict
    POST /analyze/report      -> analyse a CSV upload, return the HTML report
    POST /analyze/records     -> analyse a JSON list of event records

Run locally:
    uvicorn api:app --reload
    # then: curl -F "file=@data/winner.csv" localhost:8000/analyze

All statistics live in the ``experimentguard`` package; this module only handles
transport, parameter parsing, and error mapping.
"""

from __future__ import annotations

import io

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from experimentguard import __version__
from experimentguard.pipeline import analyze, result_to_dict
from experimentguard.report import build_html_report
from experimentguard.data import DataValidationError

app = FastAPI(
    title="ExperimentGuard",
    version=__version__,
    description="A/B test decision engine for fintech payment-flow experiments.",
)


class AnalyzeParams(BaseModel):
    """Analysis knobs shared across endpoints (sent as query parameters)."""

    planned_mde: float = Field(0.01, gt=0, lt=1, description="planned absolute MDE")
    alpha: float = Field(0.05, gt=0, lt=1, description="significance level")
    power: float = Field(0.8, gt=0, lt=1, description="target power")
    bootstrap_resamples: int = Field(10_000, gt=0, le=100_000)


class RecordsRequest(BaseModel):
    records: list[dict] = Field(..., description="list of event rows matching the schema")


def _read_csv_upload(raw: bytes) -> pd.DataFrame:
    try:
        return pd.read_csv(io.BytesIO(raw), parse_dates=["event_date"])
    except Exception as exc:  # malformed CSV / missing date column
        raise HTTPException(status_code=400, detail=f"could not parse CSV: {exc}")


def _run(df: pd.DataFrame, params: AnalyzeParams):
    try:
        return analyze(
            df,
            planned_mde=params.planned_mde,
            alpha=params.alpha,
            power=params.power,
            bootstrap_resamples=params.bootstrap_resamples,
        )
    except DataValidationError as exc:
        raise HTTPException(status_code=422, detail=f"data validation failed: {exc}")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.post("/analyze")
async def analyze_csv(
    file: UploadFile = File(..., description="experiment CSV"),
    planned_mde: float = Query(0.01, gt=0, lt=1),
    alpha: float = Query(0.05, gt=0, lt=1),
    power: float = Query(0.8, gt=0, lt=1),
    bootstrap_resamples: int = Query(10_000, gt=0, le=100_000),
) -> dict:
    """Analyse an uploaded experiment CSV and return the JSON verdict."""
    df = _read_csv_upload(await file.read())
    params = AnalyzeParams(
        planned_mde=planned_mde, alpha=alpha, power=power,
        bootstrap_resamples=bootstrap_resamples,
    )
    return result_to_dict(_run(df, params))


@app.post("/analyze/report", response_class=HTMLResponse)
async def analyze_csv_report(
    file: UploadFile = File(...),
    planned_mde: float = Query(0.01, gt=0, lt=1),
    alpha: float = Query(0.05, gt=0, lt=1),
    power: float = Query(0.8, gt=0, lt=1),
    bootstrap_resamples: int = Query(10_000, gt=0, le=100_000),
) -> str:
    """Analyse an uploaded CSV and return the one-page HTML report."""
    df = _read_csv_upload(await file.read())
    params = AnalyzeParams(
        planned_mde=planned_mde, alpha=alpha, power=power,
        bootstrap_resamples=bootstrap_resamples,
    )
    return build_html_report(_run(df, params))


@app.post("/analyze/records")
def analyze_records(
    body: RecordsRequest,
    planned_mde: float = Query(0.01, gt=0, lt=1),
    alpha: float = Query(0.05, gt=0, lt=1),
    power: float = Query(0.8, gt=0, lt=1),
    bootstrap_resamples: int = Query(10_000, gt=0, le=100_000),
) -> dict:
    """Analyse experiment rows supplied as JSON records."""
    if not body.records:
        raise HTTPException(status_code=400, detail="records must be non-empty")
    df = pd.DataFrame(body.records)
    if "event_date" in df.columns:
        df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    params = AnalyzeParams(
        planned_mde=planned_mde, alpha=alpha, power=power,
        bootstrap_resamples=bootstrap_resamples,
    )
    return result_to_dict(_run(df, params))
