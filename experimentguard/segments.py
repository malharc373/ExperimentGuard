"""Per-segment conversion analysis with multiple-testing correction.

Slicing the experiment by platform or region turns one hypothesis into many, and
testing many hypotheses at a fixed alpha inflates the false-positive rate. We run
a two-proportion test within each segment, then apply a Benjamini-Hochberg
correction to control the false discovery rate across the family of segment
tests. Only corrected-significant segments should drive a decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .effects import conversion_uplift


@dataclass
class SegmentRow:
    segment: str
    value: str
    control_n: int
    variant_n: int
    absolute_uplift: float
    p_value: float
    p_value_corrected: float
    significant: bool  # after correction


@dataclass
class SegmentResult:
    dimension: str
    method: str
    rows: list[SegmentRow] = field(default_factory=list)

    def significant_rows(self) -> list[SegmentRow]:
        return [r for r in self.rows if r.significant]

    def summary(self) -> str:
        sig = self.significant_rows()
        if not sig:
            return f"No {self.dimension} segment significant after {self.method} correction"
        parts = [f"{r.value} ({r.absolute_uplift:+.4f}, q={r.p_value_corrected:.3g})" for r in sig]
        return f"Significant {self.dimension} segments: " + ", ".join(parts)


def _benjamini_hochberg(p_values: list[float]) -> list[float]:
    """Return BH-adjusted p-values (q-values), preserving input order."""
    m = len(p_values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    prev = 1.0
    # Walk from largest p-value to smallest, enforcing monotonicity.
    for rank in range(m, 0, -1):
        idx = order[rank - 1]
        q = p_values[idx] * m / rank
        prev = min(prev, q)
        adjusted[idx] = min(prev, 1.0)
    return adjusted


def segment_analysis(
    df: pd.DataFrame,
    dimension: str,
    variant_col: str = "variant",
    converted_col: str = "converted",
    control_label: str = "control",
    variant_label: str = "variant",
    alpha: float = 0.05,
    min_arm_size: int = 30,
) -> SegmentResult:
    """Run a corrected per-segment conversion analysis along ``dimension``.

    Args:
        df: tidy experiment frame with one row per user.
        dimension: column to segment by (e.g. "platform" or "region").
        variant_col / converted_col: column names for arm and conversion flag.
        control_label / variant_label: values identifying the two arms.
        alpha: family-wise significance level applied to corrected p-values.
        min_arm_size: skip segments where either arm has fewer users than this.

    Raises:
        ValueError: if required columns are missing.
    """
    for col in (dimension, variant_col, converted_col):
        if col not in df.columns:
            raise ValueError(f"missing required column: {col}")

    raw_rows: list[SegmentRow] = []
    p_values: list[float] = []

    for value, chunk in df.groupby(dimension):
        control = chunk[chunk[variant_col] == control_label]
        variant = chunk[chunk[variant_col] == variant_label]
        if len(control) < min_arm_size or len(variant) < min_arm_size:
            continue
        res = conversion_uplift(
            control_conversions=int(control[converted_col].sum()),
            control_n=len(control),
            variant_conversions=int(variant[converted_col].sum()),
            variant_n=len(variant),
            alpha=alpha,
        )
        raw_rows.append(
            SegmentRow(
                segment=dimension,
                value=str(value),
                control_n=len(control),
                variant_n=len(variant),
                absolute_uplift=res.absolute_uplift,
                p_value=res.p_value,
                p_value_corrected=res.p_value,  # placeholder, filled below
                significant=False,
            )
        )
        p_values.append(res.p_value)

    corrected = _benjamini_hochberg(p_values)
    for row, q in zip(raw_rows, corrected):
        row.p_value_corrected = q
        row.significant = bool(q < alpha)

    return SegmentResult(dimension=dimension, method="benjamini-hochberg", rows=raw_rows)
