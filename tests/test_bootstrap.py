import numpy as np
import pytest

from experimentguard.bootstrap import bootstrap_value_uplift


def test_identical_distributions_ci_contains_zero():
    rng = np.random.default_rng(0)
    control = rng.lognormal(3, 0.5, 2000)
    variant = rng.lognormal(3, 0.5, 2000)
    res = bootstrap_value_uplift(control, variant, n_resamples=2000, random_state=1)
    assert res.ci_low < 0 < res.ci_high
    assert not res.significant


def test_shifted_variant_detected():
    rng = np.random.default_rng(0)
    control = rng.lognormal(3, 0.5, 3000)
    variant = rng.lognormal(3, 0.5, 3000) + 5.0  # clear upward shift
    res = bootstrap_value_uplift(control, variant, n_resamples=3000, random_state=1)
    assert res.mean_difference > 0
    assert res.ci_low > 0
    assert res.significant


def test_reproducible_with_seed():
    a = bootstrap_value_uplift([1, 2, 3, 4], [2, 3, 4, 5], n_resamples=500, random_state=42)
    b = bootstrap_value_uplift([1, 2, 3, 4], [2, 3, 4, 5], n_resamples=500, random_state=42)
    assert a.ci_low == b.ci_low and a.ci_high == b.ci_high


def test_empty_arm_raises():
    with pytest.raises(ValueError):
        bootstrap_value_uplift([], [1, 2, 3])


def test_bad_confidence_level_raises():
    with pytest.raises(ValueError):
        bootstrap_value_uplift([1, 2], [1, 2], confidence_level=1.5)
