"""Sample-size and minimum-detectable-effect (MDE) calculators.

These answer the two planning questions you ask *before* running an experiment
(and re-check afterwards to know whether the test was adequately powered):

    * "How many users per arm do I need to detect a given uplift?"  -> required_sample_size
    * "Given the users I have, what is the smallest uplift I could detect?" -> minimum_detectable_effect

Both use the normal approximation for a two-proportion test, which is the
standard planning tool for conversion-rate experiments.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy import stats


@dataclass
class PowerResult:
    baseline_rate: float
    mde_absolute: float          # absolute change in conversion probability
    alpha: float
    power: float
    per_arm_sample_size: int     # rounded up
    total_sample_size: int

    @property
    def mde_relative(self) -> float:
        """MDE expressed as a fraction of the baseline rate (e.g. 0.05 = +5%)."""
        if self.baseline_rate == 0:
            return float("nan")
        return self.mde_absolute / self.baseline_rate


def _z(alpha: float, power: float) -> tuple[float, float]:
    """Return (z_{alpha/2}, z_{power}) critical values for a two-sided test."""
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_power = stats.norm.ppf(power)
    return z_alpha, z_power


def required_sample_size(
    baseline_rate: float,
    mde_absolute: float,
    alpha: float = 0.05,
    power: float = 0.8,
) -> PowerResult:
    """Users required *per arm* to detect ``mde_absolute`` uplift on conversion.

    Uses the two-proportion normal approximation with separate variances for the
    control (p) and variant (p + mde) rates.

    Args:
        baseline_rate: control conversion probability, in (0, 1).
        mde_absolute: absolute uplift to detect, e.g. 0.01 for a 1pp lift.
        alpha: two-sided significance level.
        power: desired statistical power (1 - beta).

    Raises:
        ValueError: if inputs are out of range or the variant rate leaves (0, 1).
    """
    if not 0 < baseline_rate < 1:
        raise ValueError("baseline_rate must be in (0, 1)")
    if mde_absolute <= 0:
        raise ValueError("mde_absolute must be positive")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if not 0 < power < 1:
        raise ValueError("power must be in (0, 1)")

    p1 = baseline_rate
    p2 = baseline_rate + mde_absolute
    if not 0 < p2 < 1:
        raise ValueError("baseline_rate + mde_absolute must stay within (0, 1)")

    z_alpha, z_power = _z(alpha, power)
    numerator = (z_alpha + z_power) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2))
    per_arm = numerator / (mde_absolute ** 2)
    per_arm = int(math.ceil(per_arm))

    return PowerResult(
        baseline_rate=baseline_rate,
        mde_absolute=mde_absolute,
        alpha=alpha,
        power=power,
        per_arm_sample_size=per_arm,
        total_sample_size=per_arm * 2,
    )


def minimum_detectable_effect(
    baseline_rate: float,
    per_arm_sample_size: int,
    alpha: float = 0.05,
    power: float = 0.8,
) -> PowerResult:
    """Smallest absolute uplift detectable with ``per_arm_sample_size`` per arm.

    Uses the pooled-variance approximation (variance evaluated at the baseline
    rate), which is the closed-form inverse of the sample-size formula and is
    the conventional way to report an MDE.
    """
    if not 0 < baseline_rate < 1:
        raise ValueError("baseline_rate must be in (0, 1)")
    if per_arm_sample_size <= 0:
        raise ValueError("per_arm_sample_size must be positive")

    z_alpha, z_power = _z(alpha, power)
    p = baseline_rate
    mde = (z_alpha + z_power) * math.sqrt(2 * p * (1 - p) / per_arm_sample_size)

    return PowerResult(
        baseline_rate=baseline_rate,
        mde_absolute=mde,
        alpha=alpha,
        power=power,
        per_arm_sample_size=per_arm_sample_size,
        total_sample_size=per_arm_sample_size * 2,
    )
