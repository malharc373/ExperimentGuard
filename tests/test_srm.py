import pytest

from experimentguard.srm import check_srm


def test_balanced_split_no_srm():
    res = check_srm(5000, 5000, expected_control_ratio=0.5)
    assert not res.srm_detected
    assert res.p_value > 0.05


def test_clear_mismatch_detected():
    # 5500/4500 on 10k is a gross deviation from 50/50.
    res = check_srm(5500, 4500, expected_control_ratio=0.5)
    assert res.srm_detected
    assert res.p_value < 0.001


def test_observed_ratio_reported():
    res = check_srm(6000, 4000)
    assert res.observed_control_ratio == pytest.approx(0.6)


def test_expected_ratio_other_than_half():
    # Intended 90/10; observed exactly 90/10 -> no SRM.
    res = check_srm(9000, 1000, expected_control_ratio=0.9)
    assert not res.srm_detected


def test_zero_total_raises():
    with pytest.raises(ValueError):
        check_srm(0, 0)


@pytest.mark.parametrize("ratio", [0.0, 1.0, -0.2, 1.1])
def test_bad_ratio_raises(ratio):
    with pytest.raises(ValueError):
        check_srm(100, 100, expected_control_ratio=ratio)


def test_negative_counts_raise():
    with pytest.raises(ValueError):
        check_srm(-1, 100)
