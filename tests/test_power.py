import math

import pytest

from experimentguard.power import required_sample_size, minimum_detectable_effect


def test_required_sample_size_matches_known_value():
    # Standard textbook check: p=0.10, MDE=0.02, alpha=0.05, power=0.80
    # lands in the ~3.8k-4k per-arm range.
    res = required_sample_size(0.10, 0.02, alpha=0.05, power=0.80)
    assert 3500 <= res.per_arm_sample_size <= 4500
    assert res.total_sample_size == 2 * res.per_arm_sample_size


def test_smaller_mde_requires_more_users():
    big = required_sample_size(0.10, 0.02)
    small = required_sample_size(0.10, 0.01)
    assert small.per_arm_sample_size > big.per_arm_sample_size


def test_mde_relative_property():
    res = required_sample_size(0.10, 0.02)
    assert res.mde_relative == pytest.approx(0.2)


def test_mde_and_sample_size_are_roughly_inverse():
    # Feed the sample size back in; recovered MDE should be near the original.
    n = required_sample_size(0.10, 0.02).per_arm_sample_size
    recovered = minimum_detectable_effect(0.10, n)
    assert recovered.mde_absolute == pytest.approx(0.02, rel=0.15)


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_invalid_baseline_rate_raises(bad):
    with pytest.raises(ValueError):
        required_sample_size(bad, 0.01)


def test_mde_pushing_variant_out_of_range_raises():
    with pytest.raises(ValueError):
        required_sample_size(0.98, 0.05)  # 0.98 + 0.05 > 1


def test_non_positive_mde_raises():
    with pytest.raises(ValueError):
        required_sample_size(0.1, 0.0)
