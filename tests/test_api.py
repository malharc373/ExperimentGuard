"""HTTP API tests using FastAPI's TestClient."""

import io

import pytest
from fastapi.testclient import TestClient

from api import app
from experimentguard.data import generate_synthetic_experiment

client = TestClient(app)


def _csv_bytes(**kwargs) -> bytes:
    df = generate_synthetic_experiment(**kwargs)
    return df.to_csv(index=False).encode()


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_analyze_winner_ships():
    files = {"file": ("winner.csv", io.BytesIO(_csv_bytes(seed=1)), "text/csv")}
    resp = client.post("/analyze?bootstrap_resamples=500", files=files)
    assert resp.status_code == 200
    body = resp.json()
    assert body["decision"]["verdict"] == "Ship"
    assert body["conversion"]["significant"] is True
    assert body["srm"]["srm_detected"] is False


def test_analyze_srm_blocks():
    files = {"file": ("srm.csv", io.BytesIO(_csv_bytes(control_ratio=0.55, seed=11)), "text/csv")}
    resp = client.post("/analyze?bootstrap_resamples=500", files=files)
    assert resp.status_code == 200
    assert resp.json()["decision"]["verdict"] == "Do not ship"
    assert resp.json()["srm"]["srm_detected"] is True


def test_report_endpoint_returns_html():
    files = {"file": ("winner.csv", io.BytesIO(_csv_bytes(seed=1)), "text/csv")}
    resp = client.post("/analyze/report?bootstrap_resamples=500", files=files)
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    assert "ExperimentGuard" in resp.text


def test_broken_data_returns_422():
    df = generate_synthetic_experiment(seed=1)
    df.loc[0, "converted"] = 5  # non-binary
    files = {"file": ("bad.csv", io.BytesIO(df.to_csv(index=False).encode()), "text/csv")}
    resp = client.post("/analyze?bootstrap_resamples=500", files=files)
    assert resp.status_code == 422
    assert "validation" in resp.json()["detail"]


def test_records_endpoint():
    df = generate_synthetic_experiment(seed=1)
    df["event_date"] = df["event_date"].dt.strftime("%Y-%m-%d")  # JSON clients send strings
    records = df.to_dict(orient="records")
    resp = client.post(
        "/analyze/records?bootstrap_resamples=500",
        json={"records": records},
    )
    assert resp.status_code == 200
    assert resp.json()["decision"]["verdict"] == "Ship"


def test_empty_records_rejected():
    resp = client.post("/analyze/records", json={"records": []})
    assert resp.status_code == 400


def test_invalid_param_rejected():
    files = {"file": ("winner.csv", io.BytesIO(_csv_bytes(seed=1)), "text/csv")}
    resp = client.post("/analyze?alpha=1.5", files=files)
    assert resp.status_code == 422  # query validation error
