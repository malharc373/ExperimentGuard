"""End-to-end orchestration: frame in, full analysis + decision out.

``analyze`` wires the individual statistical modules together in the order the
decision engine expects: validate -> SRM -> power -> primary effect -> bootstrap
value -> guardrails -> segments -> decision. It returns a single
``AnalysisResult`` that both the CLI and the report layer consume.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .data import validate_experiment_frame, ValidationReport
from .power import required_sample_size, PowerResult
from .srm import check_srm, SRMResult
from .effects import conversion_uplift, ConversionResult
from .bootstrap import bootstrap_value_uplift, BootstrapResult
from .guardrails import check_guardrails, GuardrailResult
from .segments import segment_analysis, SegmentResult
from .decision import decide, DecisionInputs, DecisionOutcome


@dataclass
class AnalysisResult:
    experiment_id: str
    validation: ValidationReport
    power: PowerResult
    srm: SRMResult
    conversion: ConversionResult
    bootstrap: BootstrapResult
    guardrails: GuardrailResult
    segments: list[SegmentResult] = field(default_factory=list)
    decision: DecisionOutcome | None = None


def analyze(
    df: pd.DataFrame,
    baseline_rate: float | None = None,
    planned_mde: float = 0.01,
    alpha: float = 0.05,
    power: float = 0.8,
    control_label: str = "control",
    variant_label: str = "variant",
    segment_dimensions=("platform", "region"),
    bootstrap_resamples: int = 10_000,
) -> AnalysisResult:
    """Run the full ExperimentGuard analysis on a validated experiment frame.

    Args:
        df: tidy experiment frame (see ``data.REQUIRED_COLUMNS``).
        baseline_rate: control conversion for the power calc; inferred if None.
        planned_mde: absolute uplift the experiment was powered to detect.
        alpha / power: test significance and target power.
        control_label / variant_label: arm labels.
        segment_dimensions: columns to run corrected segment analysis on.
        bootstrap_resamples: resamples for the transaction-value CI.
    """
    validation = validate_experiment_frame(df, control_label, variant_label)

    control = df[df["variant"] == control_label]
    variant = df[df["variant"] == variant_label]

    if baseline_rate is None:
        baseline_rate = float(control["converted"].mean())

    power_res = required_sample_size(baseline_rate, planned_mde, alpha, power)

    srm_res = check_srm(len(control), len(variant), 0.5)

    conv_res = conversion_uplift(
        control_conversions=int(control["converted"].sum()),
        control_n=len(control),
        variant_conversions=int(variant["converted"].sum()),
        variant_n=len(variant),
        alpha=alpha,
    )

    # Value uplift is measured over converting users (those who transacted).
    control_val = control.loc[control["converted"] == 1, "transaction_value"].to_numpy()
    variant_val = variant.loc[variant["converted"] == 1, "transaction_value"].to_numpy()
    boot_res = bootstrap_value_uplift(
        control_val, variant_val, n_resamples=bootstrap_resamples
    )

    guard_res = check_guardrails(
        control_success=int(control["payment_success"].sum()),
        control_n=len(control),
        variant_success=int(variant["payment_success"].sum()),
        variant_n=len(variant),
        control_latency=control["latency_ms"].to_numpy(),
        variant_latency=variant["latency_ms"].to_numpy(),
        alpha=alpha,
    )

    segments = [
        segment_analysis(df, dim, control_label=control_label,
                         variant_label=variant_label, alpha=alpha)
        for dim in segment_dimensions
        if dim in df.columns
    ]

    outcome = decide(
        DecisionInputs(
            srm=srm_res,
            conversion=conv_res,
            guardrails=guard_res,
            required_per_arm=power_res.per_arm_sample_size,
        )
    )

    experiment_id = str(df["experiment_id"].iloc[0]) if "experiment_id" in df else "unknown"

    return AnalysisResult(
        experiment_id=experiment_id,
        validation=validation,
        power=power_res,
        srm=srm_res,
        conversion=conv_res,
        bootstrap=boot_res,
        guardrails=guard_res,
        segments=segments,
        decision=outcome,
    )


def result_to_dict(result: AnalysisResult) -> dict:
    """Serialise an ``AnalysisResult`` into a plain, JSON-safe dictionary.

    Used by the HTTP API; keeps the wire format in one place so it stays in sync
    with the dataclasses above.
    """
    return {
        "experiment_id": result.experiment_id,
        "decision": {
            "verdict": result.decision.decision.value,
            "reasons": result.decision.reasons,
        },
        "validation": {
            "n_rows": result.validation.n_rows,
            "n_users": result.validation.n_users,
            "arms": result.validation.arms,
            "warnings": result.validation.warnings,
        },
        "power": {
            "baseline_rate": result.power.baseline_rate,
            "mde_absolute": result.power.mde_absolute,
            "mde_relative": result.power.mde_relative,
            "per_arm_sample_size": result.power.per_arm_sample_size,
            "total_sample_size": result.power.total_sample_size,
            "alpha": result.power.alpha,
            "power": result.power.power,
        },
        "srm": {
            "srm_detected": result.srm.srm_detected,
            "observed_control_ratio": result.srm.observed_control_ratio,
            "p_value": result.srm.p_value,
        },
        "conversion": {
            "control_rate": result.conversion.control_rate,
            "variant_rate": result.conversion.variant_rate,
            "absolute_uplift": result.conversion.absolute_uplift,
            "relative_uplift": result.conversion.relative_uplift,
            "ci_low": result.conversion.ci_low,
            "ci_high": result.conversion.ci_high,
            "p_value": result.conversion.p_value,
            "significant": result.conversion.significant,
        },
        "transaction_value": {
            "control_mean": result.bootstrap.control_mean,
            "variant_mean": result.bootstrap.variant_mean,
            "mean_difference": result.bootstrap.mean_difference,
            "ci_low": result.bootstrap.ci_low,
            "ci_high": result.bootstrap.ci_high,
            "significant": result.bootstrap.significant,
        },
        "guardrails": {
            "passed": result.guardrails.passed,
            "breaches": result.guardrails.breaches,
            "details": result.guardrails.details,
        },
        "segments": [
            {
                "dimension": seg.dimension,
                "method": seg.method,
                "rows": [
                    {
                        "value": row.value,
                        "control_n": row.control_n,
                        "variant_n": row.variant_n,
                        "absolute_uplift": row.absolute_uplift,
                        "p_value": row.p_value,
                        "p_value_corrected": row.p_value_corrected,
                        "significant": row.significant,
                    }
                    for row in seg.rows
                ],
            }
            for seg in result.segments
        ],
    }
