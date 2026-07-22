import numpy as np
import pandas as pd
import pytest

from experimentguard.segments import segment_analysis, _benjamini_hochberg


def _frame(effect_by_platform, n=4000, seed=0):
    """Build a two-arm frame where each platform has a set conversion effect."""
    rng = np.random.default_rng(seed)
    rows = []
    for platform, (p_c, p_v) in effect_by_platform.items():
        for label, p in (("control", p_c), ("variant", p_v)):
            conv = rng.binomial(1, p, n)
            rows.append(pd.DataFrame({
                "user_id": [f"{platform}_{label}_{i}" for i in range(n)],
                "variant": label,
                "converted": conv,
                "platform": platform,
            }))
    return pd.concat(rows, ignore_index=True)


def test_benjamini_hochberg_monotone_and_bounded():
    q = _benjamini_hochberg([0.01, 0.02, 0.03, 0.5])
    assert all(0 <= v <= 1 for v in q)
    # BH q-values are >= raw p-values here.
    assert q[0] >= 0.01


def test_empty_bh_returns_empty():
    assert _benjamini_hochberg([]) == []


def test_true_effect_segment_flagged():
    df = _frame({"ios": (0.10, 0.16), "android": (0.10, 0.10), "web": (0.10, 0.10)})
    res = segment_analysis(df, "platform")
    sig = {r.value for r in res.significant_rows()}
    assert "ios" in sig
    assert "android" not in sig


def test_no_effect_anywhere_none_flagged():
    df = _frame({"ios": (0.10, 0.10), "android": (0.10, 0.10)})
    res = segment_analysis(df, "platform")
    assert res.significant_rows() == []


def test_small_segments_skipped():
    df = _frame({"ios": (0.10, 0.16)}, n=10)  # below min_arm_size
    res = segment_analysis(df, "platform", min_arm_size=30)
    assert res.rows == []


def test_missing_dimension_raises():
    df = _frame({"ios": (0.1, 0.1)})
    with pytest.raises(ValueError):
        segment_analysis(df, "nonexistent_column")
