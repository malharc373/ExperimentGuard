# ExperimentGuard — A/B Test Decision Engine

[![CI](https://github.com/malharc373/ExperimentGuard/actions/workflows/ci.yml/badge.svg)](https://github.com/malharc373/ExperimentGuard/actions/workflows/ci.yml)

A reproducible experimentation platform for fintech payment-flow changes. Given a
control/variant dataset, ExperimentGuard validates the randomisation, measures
conversion and transaction-value uplift with confidence intervals, monitors
payment-flow guardrails, corrects for multiple testing across segments, and
issues an explainable **Ship / Do not ship / Continue** recommendation.

> Built to complement Payment Pulse Lab: where that project explores fintech
> trends, this one proves statistical decision-making, data validation, and
> business communication.

## Why each piece exists

| Deliverable | Module | What it protects against |
|---|---|---|
| Sample-size / MDE calculator | `power.py` | Running an underpowered test you can't learn from |
| Sample-ratio-mismatch (SRM) check | `srm.py` | Trusting effects behind a broken randomisation |
| Conversion uplift + CI + two-proportion test | `effects.py` | Calling noise a win |
| Bootstrap CI for value uplift | `bootstrap.py` | Assuming normality on heavy-tailed money |
| Payment-success & latency guardrails | `guardrails.py` | Shipping a conversion win that degrades reliability |
| Segment analysis + BH correction | `segments.py` | False positives from slicing by platform/region |
| Rules-based decision engine | `decision.py` | Ad-hoc, unexplainable ship calls |
| One-page report (MD/HTML) | `report.py` | Results nobody can act on |

**Sample-ratio mismatch** means the observed control/variant split differs from
the intended one (e.g. 50/50). It is a randomisation-quality gate: Microsoft's
experimentation guidance is to pass an SRM check *before* analysing effects, and
this engine treats a detected SRM as an automatic Do-not-ship.

## Data schema

One row per user-event:

```
user_id, experiment_id, variant, event_date,
converted, transaction_value, region, platform,
payment_success, latency_ms
```

`validate_experiment_frame` enforces this schema and rejects broken data
(non-binary flags, negatives, unexpected arm labels, users spanning both arms).

## Decision precedence

1. **Data validity** — SRM detected → Do not ship
2. **Safety** — a guardrail (success rate / latency) breached → Do not ship
3. **Harm** — primary metric significantly *down* → Do not ship
4. **Win** — primary metric significantly *up*, guardrails intact → **Ship**
5. **Underpowered** — not significant and n < required → **Continue**
6. **Powered null** — not significant and n ≥ required → Do not ship

Every verdict ships with the ordered list of rules that fired.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# generate sample scenarios (winner / SRM / guardrail failure)
python scripts/generate_sample_data.py

# analyse one and write an HTML report
python run_analysis.py data/winner.csv --html data/report.html

# run the test suite
pytest -q

# optional interactive demo
streamlit run app.py
```

A rendered sample verdict lives at [`examples/report_winner.html`](examples/report_winner.html)
(open it in a browser).

## HTTP API

A FastAPI service ([api.py](api.py)) exposes the engine over HTTP:

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness probe |
| POST | `/analyze` | analyse a CSV upload → JSON verdict |
| POST | `/analyze/report` | analyse a CSV upload → one-page HTML report |
| POST | `/analyze/records` | analyse JSON event records |

Analysis knobs (`planned_mde`, `alpha`, `power`, `bootstrap_resamples`) are query
parameters. Broken data returns `422`; malformed CSV returns `400`.

```bash
uvicorn api:app --reload
curl -F "file=@data/winner.csv" "localhost:8000/analyze?bootstrap_resamples=500"
# interactive docs at http://localhost:8000/docs
```

## Docker

```bash
docker build -t experimentguard .
docker run --rm -p 8000:8000 experimentguard
curl localhost:8000/health
```

The image installs dependencies in a cached layer, ships only the package + API,
and defines a `HEALTHCHECK` against `/health`.

### As a library

```python
import pandas as pd
from experimentguard import analyze, build_report

df = pd.read_csv("data/winner.csv", parse_dates=["event_date"])
result = analyze(df, planned_mde=0.01, alpha=0.05, power=0.8)
print(result.decision.decision.value)   # "Ship"
print(build_report(result))             # full Markdown report
```

## Testing

`pytest` covers every statistical function (known-value checks, direction,
edge cases), the decision-rule precedence, the full pipeline on three synthetic
scenarios, and a suite of **intentionally broken-data** cases that must fail
validation loudly rather than corrupt the analysis.

```
pytest -q   # 62 tests
```

## Project layout

```
experimentguard/
  data.py        schema validation + synthetic generator
  power.py       sample-size & MDE
  srm.py         sample-ratio-mismatch check
  effects.py     conversion uplift / CI / two-proportion test
  bootstrap.py   value-uplift bootstrap CI
  guardrails.py  success-rate & latency regression checks
  segments.py    per-segment analysis + BH correction
  decision.py    rules-based Ship/No-ship/Continue engine
  report.py      Markdown / HTML report
  pipeline.py    end-to-end orchestration + JSON serialisation
run_analysis.py  CLI entry point
api.py           FastAPI service
Dockerfile       container image for the API
app.py           Streamlit demo
scripts/         sample-data generator
tests/           unit + broken-data + pipeline + API tests
```

## Stack

Python · Pandas · NumPy · SciPy · FastAPI · Docker · pytest · Streamlit (optional demo)
