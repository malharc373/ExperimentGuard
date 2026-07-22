"""Decision-engine rule precedence tests using lightweight stub inputs."""

import numpy as np

from experimentguard.decision import decide, DecisionInputs, Decision
from experimentguard.srm import check_srm
from experimentguard.effects import conversion_uplift
from experimentguard.guardrails import check_guardrails


def _latency(mean, n=3000, seed=0):
    rng = np.random.default_rng(seed)
    return rng.gamma(4.0, mean / 4.0, n)


def _good_guardrails():
    return check_guardrails(
        9850, 10000, 9845, 10000,
        _latency(240), _latency(243, seed=1),
    )


def _clean_srm():
    return check_srm(10000, 10000)


def test_ship_on_clear_win():
    inputs = DecisionInputs(
        srm=_clean_srm(),
        conversion=conversion_uplift(1200, 10000, 1400, 10000),
        guardrails=_good_guardrails(),
        required_per_arm=5000,
    )
    assert decide(inputs).decision == Decision.SHIP


def test_srm_overrides_a_win():
    inputs = DecisionInputs(
        srm=check_srm(5500, 4500),  # SRM present
        conversion=conversion_uplift(1200, 10000, 1400, 10000),  # would ship
        guardrails=_good_guardrails(),
        required_per_arm=5000,
    )
    out = decide(inputs)
    assert out.decision == Decision.DO_NOT_SHIP
    assert "mismatch" in out.reasons[0].lower()


def test_guardrail_breach_overrides_a_win():
    bad_guard = check_guardrails(
        9850, 10000, 9845, 10000,
        _latency(240), _latency(320, seed=1),  # latency regressed
    )
    inputs = DecisionInputs(
        srm=_clean_srm(),
        conversion=conversion_uplift(1200, 10000, 1400, 10000),
        guardrails=bad_guard,
        required_per_arm=5000,
    )
    assert decide(inputs).decision == Decision.DO_NOT_SHIP


def test_significant_drop_is_do_not_ship():
    inputs = DecisionInputs(
        srm=_clean_srm(),
        conversion=conversion_uplift(1400, 10000, 1200, 10000),  # down
        guardrails=_good_guardrails(),
        required_per_arm=5000,
    )
    assert decide(inputs).decision == Decision.DO_NOT_SHIP


def test_underpowered_null_is_continue():
    inputs = DecisionInputs(
        srm=_clean_srm(),
        conversion=conversion_uplift(120, 1000, 128, 1000),  # not sig, small n
        guardrails=_good_guardrails(),
        required_per_arm=5000,  # need far more
    )
    out = decide(inputs)
    assert out.decision == Decision.CONTINUE


def test_powered_null_is_do_not_ship():
    inputs = DecisionInputs(
        srm=_clean_srm(),
        conversion=conversion_uplift(1200, 10000, 1205, 10000),  # flat, big n
        guardrails=_good_guardrails(),
        required_per_arm=5000,
    )
    assert decide(inputs).decision == Decision.DO_NOT_SHIP
