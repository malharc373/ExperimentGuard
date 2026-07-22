import pytest

from experimentguard.effects import conversion_uplift


def test_no_difference_not_significant():
    res = conversion_uplift(1000, 10000, 1000, 10000)
    assert res.absolute_uplift == pytest.approx(0.0)
    assert not res.significant
    assert res.p_value > 0.05


def test_clear_uplift_significant():
    # 12% -> 14% on 10k per arm is a strong, detectable lift.
    res = conversion_uplift(1200, 10000, 1400, 10000)
    assert res.absolute_uplift == pytest.approx(0.02, abs=1e-9)
    assert res.significant
    assert res.p_value < 0.01
    assert res.ci_low > 0  # CI excludes zero


def test_relative_uplift_computation():
    res = conversion_uplift(1000, 10000, 1100, 10000)
    # abs 0.01 over baseline 0.10 == 10% relative
    assert res.relative_uplift == pytest.approx(0.1, rel=1e-6)


def test_ci_brackets_point_estimate():
    res = conversion_uplift(1200, 10000, 1400, 10000)
    assert res.ci_low < res.absolute_uplift < res.ci_high


def test_negative_uplift_direction():
    res = conversion_uplift(1400, 10000, 1200, 10000)
    assert res.absolute_uplift < 0
    assert res.significant


def test_conversions_exceeding_n_raises():
    with pytest.raises(ValueError):
        conversion_uplift(11000, 10000, 1000, 10000)


def test_zero_arm_size_raises():
    with pytest.raises(ValueError):
        conversion_uplift(0, 0, 1, 100)
