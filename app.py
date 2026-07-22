"""Minimal Streamlit demo for ExperimentGuard.

Run: ``streamlit run app.py``

Upload an experiment CSV (or use the built-in synthetic sample), tune the power
settings, and see the decision, evidence, and full report. The UI is a thin
layer over ``experimentguard.pipeline.analyze`` — all logic lives in the package.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from experimentguard.pipeline import analyze
from experimentguard.report import build_report
from experimentguard.data import (
    generate_synthetic_experiment,
    validate_experiment_frame,
    DataValidationError,
)
from experimentguard.decision import Decision

st.set_page_config(page_title="ExperimentGuard", page_icon="🛡️", layout="centered")
st.title("🛡️ ExperimentGuard")
st.caption("A/B test decision engine for fintech payment-flow experiments")

with st.sidebar:
    st.header("Settings")
    mde = st.number_input("Planned MDE (absolute)", 0.001, 0.1, 0.01, 0.001, format="%.3f")
    alpha = st.number_input("Alpha", 0.001, 0.2, 0.05, 0.005, format="%.3f")
    power = st.number_input("Power", 0.5, 0.99, 0.8, 0.05)
    resamples = st.select_slider("Bootstrap resamples", [1000, 5000, 10000], value=5000)

source = st.radio("Data source", ["Synthetic sample", "Upload CSV"], horizontal=True)

df: pd.DataFrame | None = None
if source == "Synthetic sample":
    scenario = st.selectbox(
        "Scenario",
        ["Winner", "Sample-ratio mismatch", "Guardrail failure (latency)"],
    )
    if scenario == "Winner":
        df = generate_synthetic_experiment(seed=1)
    elif scenario == "Sample-ratio mismatch":
        df = generate_synthetic_experiment(control_ratio=0.55, seed=11)
    else:
        df = generate_synthetic_experiment(variant_latency_ms=310.0, seed=13)
else:
    upload = st.file_uploader("Experiment CSV", type="csv")
    if upload is not None:
        df = pd.read_csv(upload, parse_dates=["event_date"])

if df is not None:
    try:
        validate_experiment_frame(df)
    except DataValidationError as exc:
        st.error(f"Data validation failed: {exc}")
        st.stop()

    result = analyze(df, planned_mde=mde, alpha=alpha, power=power,
                     bootstrap_resamples=int(resamples))
    decision = result.decision.decision

    color = {
        Decision.SHIP: "green",
        Decision.DO_NOT_SHIP: "red",
        Decision.CONTINUE: "orange",
    }[decision]
    st.markdown(f"### Decision: :{color}[**{decision.value}**]")
    for reason in result.decision.reasons:
        st.write(f"- {reason}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Conversion uplift", f"{result.conversion.absolute_uplift:+.4f}",
                f"{result.conversion.relative_uplift * 100:+.1f}%")
    col2.metric("Value uplift", f"{result.bootstrap.mean_difference:+.2f}")
    col3.metric("SRM p-value", f"{result.srm.p_value:.3g}")

    st.divider()
    st.markdown(build_report(result))
else:
    st.info("Choose a synthetic scenario or upload a CSV to run the analysis.")
