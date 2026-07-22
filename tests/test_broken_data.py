"""Intentionally broken-data cases: validation must fail loudly, not silently."""

import pandas as pd
import pytest

from experimentguard.data import (
    validate_experiment_frame,
    generate_synthetic_experiment,
    DataValidationError,
    REQUIRED_COLUMNS,
)


def _good_frame(n=200):
    return generate_synthetic_experiment(n_per_arm=n // 2, seed=3)


def test_valid_frame_passes():
    report = validate_experiment_frame(_good_frame())
    assert report.n_rows > 0
    assert set(report.arms) == {"control", "variant"}


def test_missing_column_raises():
    df = _good_frame().drop(columns=["latency_ms"])
    with pytest.raises(DataValidationError, match="missing required columns"):
        validate_experiment_frame(df)


def test_empty_frame_raises():
    empty = pd.DataFrame(columns=REQUIRED_COLUMNS)
    with pytest.raises(DataValidationError, match="empty"):
        validate_experiment_frame(empty)


def test_non_binary_converted_raises():
    df = _good_frame()
    df.loc[0, "converted"] = 2
    with pytest.raises(DataValidationError, match="non-binary"):
        validate_experiment_frame(df)


def test_negative_transaction_value_raises():
    df = _good_frame()
    df.loc[0, "transaction_value"] = -10.0
    with pytest.raises(DataValidationError, match="negative"):
        validate_experiment_frame(df)


def test_negative_latency_raises():
    df = _good_frame()
    df.loc[0, "latency_ms"] = -5.0
    with pytest.raises(DataValidationError, match="negative"):
        validate_experiment_frame(df)


def test_unexpected_arm_label_raises():
    df = _good_frame()
    df.loc[0, "variant"] = "treatment_2"
    with pytest.raises(DataValidationError, match="unexpected arm"):
        validate_experiment_frame(df)


def test_user_in_two_arms_raises():
    df = _good_frame()
    # Force one user id to exist in both arms.
    shared = df.loc[df["variant"] == "control", "user_id"].iloc[0]
    idx = df.index[df["variant"] == "variant"][0]
    df.loc[idx, "user_id"] = shared
    with pytest.raises(DataValidationError, match="more than one arm"):
        validate_experiment_frame(df)


def test_null_values_are_warned_not_fatal():
    df = _good_frame()
    df.loc[0, "region"] = None
    report = validate_experiment_frame(df)
    assert any("region" in w for w in report.warnings)
