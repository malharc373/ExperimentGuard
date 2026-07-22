import numpy as np
import pytest

from experimentguard.guardrails import check_guardrails


def _latency(mean, n=3000, seed=0):
    rng = np.random.default_rng(seed)
    return rng.gamma(4.0, mean / 4.0, n)


def test_all_pass():
    res = check_guardrails(
        control_success=9850, control_n=10000,
        variant_success=9840, variant_n=10000,
        control_latency=_latency(240), variant_latency=_latency(245, seed=1),
    )
    assert res.passed
    assert res.breaches == []


def test_latency_regression_breaches():
    res = check_guardrails(
        control_success=9850, control_n=10000,
        variant_success=9845, variant_n=10000,
        control_latency=_latency(240), variant_latency=_latency(320, seed=1),
    )
    assert not res.passed
    assert any("latency" in b for b in res.breaches)


def test_success_rate_regression_breaches():
    res = check_guardrails(
        control_success=9850, control_n=10000,
        variant_success=9600, variant_n=10000,  # 98.5% -> 96%
        control_latency=_latency(240), variant_latency=_latency(242, seed=1),
    )
    assert not res.passed
    assert any("payment-success" in b for b in res.breaches)


def test_tiny_success_drop_within_tolerance_passes():
    res = check_guardrails(
        control_success=9850, control_n=10000,
        variant_success=9848, variant_n=10000,
        control_latency=_latency(240), variant_latency=_latency(241, seed=1),
    )
    assert res.passed


def test_empty_latency_raises():
    with pytest.raises(ValueError):
        check_guardrails(100, 100, 100, 100, [], [1, 2, 3])
