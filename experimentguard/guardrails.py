"""Guardrail checks: metrics that must not regress even if the primary metric wins.

A payment-flow change can lift conversion while quietly degrading reliability.
Two guardrails protect against shipping such a change:

    * payment-success rate must not drop by more than a tolerance (checked with a
      one-sided two-proportion test plus an absolute-drop tolerance), and
    * latency (mean and p95) must not rise beyond a tolerance.

Each guardrail returns a pass/fail plus the numbers behind it so the report can
explain *why* a guardrail tripped.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import stats


@dataclass
class GuardrailResult:
    passed: bool
    breaches: list[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)

    def summary(self) -> str:
        if self.passed:
            return "Guardrails PASSED"
        return "Guardrails BREACHED: " + "; ".join(self.breaches)


def _success_rate_regressed(
    control_success: int,
    control_n: int,
    variant_success: int,
    variant_n: int,
    abs_tolerance: float,
    alpha: float,
) -> tuple[bool, dict]:
    p_c = control_success / control_n
    p_v = variant_success / variant_n
    drop = p_c - p_v  # positive drop == regression

    p_pool = (control_success + variant_success) / (control_n + variant_n)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / control_n + 1 / variant_n))
    # One-sided test: is the variant rate significantly *lower*?
    if se == 0:
        p_value = 1.0
    else:
        z = (p_v - p_c) / se
        p_value = stats.norm.cdf(z)  # P(variant <= control)

    regressed = bool(drop > abs_tolerance and p_value < alpha)
    return regressed, {
        "control_rate": p_c,
        "variant_rate": p_v,
        "absolute_drop": drop,
        "abs_tolerance": abs_tolerance,
        "p_value": float(p_value),
    }


def _latency_regressed(
    control_latency,
    variant_latency,
    rel_tolerance: float,
    alpha: float,
) -> tuple[bool, dict]:
    control = np.asarray(control_latency, dtype=float)
    variant = np.asarray(variant_latency, dtype=float)
    c_mean, v_mean = control.mean(), variant.mean()
    c_p95 = float(np.percentile(control, 95))
    v_p95 = float(np.percentile(variant, 95))

    rel_increase = (v_mean - c_mean) / c_mean if c_mean > 0 else float("inf")
    # One-sided Welch t-test: is variant latency higher?
    t_stat, p_two = stats.ttest_ind(variant, control, equal_var=False)
    p_one = p_two / 2 if t_stat > 0 else 1 - p_two / 2

    regressed = bool(rel_increase > rel_tolerance and p_one < alpha)
    return regressed, {
        "control_mean_ms": float(c_mean),
        "variant_mean_ms": float(v_mean),
        "control_p95_ms": c_p95,
        "variant_p95_ms": v_p95,
        "relative_increase": float(rel_increase),
        "rel_tolerance": rel_tolerance,
        "p_value": float(p_one),
    }


def check_guardrails(
    control_success: int,
    control_n: int,
    variant_success: int,
    variant_n: int,
    control_latency,
    variant_latency,
    success_abs_tolerance: float = 0.005,
    latency_rel_tolerance: float = 0.10,
    alpha: float = 0.05,
) -> GuardrailResult:
    """Evaluate payment-success and latency guardrails.

    Args:
        control_success / variant_success: successful payments per arm.
        control_n / variant_n: total payment attempts per arm.
        control_latency / variant_latency: array-likes of per-event latency (ms).
        success_abs_tolerance: allowed absolute drop in success rate (e.g. 0.005).
        latency_rel_tolerance: allowed relative rise in mean latency (e.g. 0.10).
        alpha: significance level for the one-sided regression tests.

    Raises:
        ValueError: on non-positive arm sizes or empty latency arrays.
    """
    if control_n <= 0 or variant_n <= 0:
        raise ValueError("arm sizes must be positive")
    if len(control_latency) == 0 or len(variant_latency) == 0:
        raise ValueError("latency arrays must be non-empty")

    breaches: list[str] = []
    details: dict = {}

    success_bad, success_detail = _success_rate_regressed(
        control_success, control_n, variant_success, variant_n,
        success_abs_tolerance, alpha,
    )
    details["payment_success"] = success_detail
    if success_bad:
        breaches.append(
            f"payment-success dropped {success_detail['absolute_drop']:.4f} "
            f"(> {success_abs_tolerance})"
        )

    latency_bad, latency_detail = _latency_regressed(
        control_latency, variant_latency, latency_rel_tolerance, alpha,
    )
    details["latency"] = latency_detail
    if latency_bad:
        breaches.append(
            f"latency rose {latency_detail['relative_increase'] * 100:.1f}% "
            f"(> {latency_rel_tolerance * 100:.0f}%)"
        )

    return GuardrailResult(passed=len(breaches) == 0, breaches=breaches, details=details)
