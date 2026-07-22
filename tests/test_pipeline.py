"""End-to-end pipeline tests over synthetic scenarios."""

from experimentguard.data import generate_synthetic_experiment
from experimentguard.pipeline import analyze
from experimentguard.report import build_report, build_html_report
from experimentguard.decision import Decision


def test_winner_scenario_ships():
    df = generate_synthetic_experiment(seed=1)
    result = analyze(df, planned_mde=0.01, bootstrap_resamples=1000)
    assert result.decision.decision == Decision.SHIP
    assert not result.srm.srm_detected
    assert result.guardrails.passed


def test_srm_scenario_blocks():
    df = generate_synthetic_experiment(control_ratio=0.55, seed=11)
    result = analyze(df, bootstrap_resamples=1000)
    assert result.srm.srm_detected
    assert result.decision.decision == Decision.DO_NOT_SHIP


def test_latency_regression_blocks():
    df = generate_synthetic_experiment(variant_latency_ms=320.0, seed=13)
    result = analyze(df, bootstrap_resamples=1000)
    assert not result.guardrails.passed
    assert result.decision.decision == Decision.DO_NOT_SHIP


def test_report_renders_all_sections():
    df = generate_synthetic_experiment(seed=1)
    result = analyze(df, bootstrap_resamples=1000)
    md = build_report(result)
    for heading in ["Decision", "Data validity", "Power", "conversion",
                    "Transaction-value", "Guardrails", "Segment"]:
        assert heading in md
    html = build_html_report(result)
    assert html.startswith("<!doctype html>")
    assert "ExperimentGuard" in html
