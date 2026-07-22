"""Data loading, schema validation, and synthetic experiment generation.

The expected schema (one row per user-event) is::

    user_id, experiment_id, variant, event_date,
    converted, transaction_value, region, platform,
    payment_success, latency_ms

``validate_experiment_frame`` is deliberately strict: it is the first thing the
pipeline runs, and it is where the "intentionally broken data" test cases are
meant to fail loudly rather than silently corrupt the analysis.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "user_id",
    "experiment_id",
    "variant",
    "event_date",
    "converted",
    "transaction_value",
    "region",
    "platform",
    "payment_success",
    "latency_ms",
]

BINARY_COLUMNS = ["converted", "payment_success"]


class DataValidationError(ValueError):
    """Raised when an experiment frame violates the expected schema/constraints."""


@dataclass
class ValidationReport:
    n_rows: int
    n_users: int
    arms: list[str]
    warnings: list[str]


def validate_experiment_frame(
    df: pd.DataFrame,
    control_label: str = "control",
    variant_label: str = "variant",
) -> ValidationReport:
    """Validate schema and value constraints, raising on hard errors.

    Hard errors (raise ``DataValidationError``):
        * missing required columns,
        * empty frame,
        * binary columns outside {0, 1},
        * negative transaction_value or latency_ms,
        * an arm other than the two expected labels,
        * duplicate user_id within an arm's assignment.

    Soft issues are returned as warnings (e.g. nulls present, only one arm).
    """
    warnings: list[str] = []

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(f"missing required columns: {missing}")

    if len(df) == 0:
        raise DataValidationError("experiment frame is empty")

    for col in BINARY_COLUMNS:
        bad = set(pd.unique(df[col].dropna())) - {0, 1, True, False}
        if bad:
            raise DataValidationError(f"column '{col}' has non-binary values: {bad}")

    for col in ("transaction_value", "latency_ms"):
        if (df[col].dropna() < 0).any():
            raise DataValidationError(f"column '{col}' contains negative values")

    arms = sorted(str(a) for a in pd.unique(df["variant"].dropna()))
    unexpected = set(arms) - {control_label, variant_label}
    if unexpected:
        raise DataValidationError(f"unexpected arm labels: {sorted(unexpected)}")
    if len(arms) < 2:
        warnings.append(f"only one arm present: {arms}")

    # A user should belong to a single arm; cross-arm contamination is a bug.
    per_user_arms = df.groupby("user_id")["variant"].nunique()
    contaminated = per_user_arms[per_user_arms > 1]
    if len(contaminated) > 0:
        raise DataValidationError(
            f"{len(contaminated)} user_id(s) appear in more than one arm"
        )

    for col in df.columns:
        n_null = int(df[col].isna().sum())
        if n_null:
            warnings.append(f"column '{col}' has {n_null} null value(s)")

    return ValidationReport(
        n_rows=len(df),
        n_users=int(df["user_id"].nunique()),
        arms=arms,
        warnings=warnings,
    )


def load_experiment_csv(path: str, **validate_kwargs) -> pd.DataFrame:
    """Load a CSV and validate it before returning the frame."""
    df = pd.read_csv(path, parse_dates=["event_date"])
    validate_experiment_frame(df, **validate_kwargs)
    return df


def generate_synthetic_experiment(
    n_per_arm: int = 15000,
    experiment_id: str = "exp_checkout_v2",
    control_conversion: float = 0.120,
    variant_conversion: float = 0.138,
    control_success: float = 0.985,
    variant_success: float = 0.984,
    control_latency_ms: float = 240.0,
    variant_latency_ms: float = 250.0,
    value_mean: float = 42.0,
    variant_value_lift: float = 1.5,
    control_ratio: float = 0.5,
    regions=("north", "south", "east", "west"),
    platforms=("ios", "android", "web"),
    seed: int = 7,
) -> pd.DataFrame:
    """Generate a realistic, tunable synthetic experiment frame.

    Defaults describe a mild winner: +1.2pp conversion, a small value lift, and
    guardrails just barely intact. Tweak the arguments to manufacture SRM,
    guardrail breaches, or null results for demos and tests.
    """
    rng = np.random.default_rng(seed)

    n_control = int(round((n_per_arm * 2) * control_ratio))
    n_variant = n_per_arm * 2 - n_control

    def _arm(n, label, p_conv, p_succ, lat_mean, val_lift):
        converted = rng.binomial(1, p_conv, size=n)
        success = rng.binomial(1, p_succ, size=n)
        # Log-normal transaction value; only converted users transact (>0).
        base = rng.lognormal(mean=np.log(value_mean), sigma=0.6, size=n) + val_lift
        value = np.where(converted == 1, base, 0.0)
        latency = rng.gamma(shape=4.0, scale=lat_mean / 4.0, size=n)
        return pd.DataFrame(
            {
                "user_id": [f"{label}_{i}" for i in range(n)],
                "experiment_id": experiment_id,
                "variant": label,
                "event_date": pd.Timestamp("2026-07-01")
                + pd.to_timedelta(rng.integers(0, 14, size=n), unit="D"),
                "converted": converted,
                "transaction_value": np.round(value, 2),
                "region": rng.choice(regions, size=n),
                "platform": rng.choice(platforms, size=n),
                "payment_success": success,
                "latency_ms": np.round(latency, 1),
            }
        )

    control = _arm(n_control, "control", control_conversion, control_success,
                   control_latency_ms, 0.0)
    variant = _arm(n_variant, "variant", variant_conversion, variant_success,
                   variant_latency_ms, variant_value_lift)

    df = pd.concat([control, variant], ignore_index=True)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
