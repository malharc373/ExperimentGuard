"""Primary conversion-metric analysis: uplift, confidence interval, and test.

For the primary metric (``converted``) we report:

    * absolute and relative uplift,
    * a confidence interval on the absolute difference (unpooled/Wald SE, which
      is the correct SE for an *interval* around the observed difference), and
    * a two-proportion z-test p-value (pooled SE, the correct SE for testing the
      null of no difference).

Using the pooled SE for the test and the unpooled SE for the CI is deliberate;
mixing them up is a common A/B testing bug.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy import stats


@dataclass
class ConversionResult:
    control_n: int
    variant_n: int
    control_rate: float
    variant_rate: float
    absolute_uplift: float
    relative_uplift: float
    ci_low: float
    ci_high: float
    z_statistic: float
    p_value: float
    alpha: float
    significant: bool

    def summary(self) -> str:
        direction = "up" if self.absolute_uplift >= 0 else "down"
        sig = "significant" if self.significant else "not significant"
        return (
            f"Conversion {self.control_rate:.4f} -> {self.variant_rate:.4f} "
            f"({direction} {abs(self.relative_uplift) * 100:.2f}%), "
            f"95% CI [{self.ci_low:+.4f}, {self.ci_high:+.4f}], "
            f"p={self.p_value:.4g} ({sig})"
        )


def conversion_uplift(
    control_conversions: int,
    control_n: int,
    variant_conversions: int,
    variant_n: int,
    alpha: float = 0.05,
) -> ConversionResult:
    """Two-proportion analysis of a conversion metric.

    Args:
        control_conversions: successes in control.
        control_n: users in control.
        variant_conversions: successes in variant.
        variant_n: users in variant.
        alpha: two-sided significance level for the test and (1-alpha) CI.

    Raises:
        ValueError: on non-positive arm sizes or conversions outside [0, n].
    """
    if control_n <= 0 or variant_n <= 0:
        raise ValueError("arm sizes must be positive")
    if not 0 <= control_conversions <= control_n:
        raise ValueError("control_conversions must be within [0, control_n]")
    if not 0 <= variant_conversions <= variant_n:
        raise ValueError("variant_conversions must be within [0, variant_n]")

    p_c = control_conversions / control_n
    p_v = variant_conversions / variant_n
    diff = p_v - p_c

    # Pooled SE for the hypothesis test (null: equal proportions).
    p_pool = (control_conversions + variant_conversions) / (control_n + variant_n)
    se_pool = math.sqrt(p_pool * (1 - p_pool) * (1 / control_n + 1 / variant_n))
    if se_pool == 0:
        z = 0.0
        p_value = 1.0
    else:
        z = diff / se_pool
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    # Unpooled SE for the confidence interval around the observed difference.
    se_unpooled = math.sqrt(
        p_c * (1 - p_c) / control_n + p_v * (1 - p_v) / variant_n
    )
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci_low = diff - z_crit * se_unpooled
    ci_high = diff + z_crit * se_unpooled

    relative = diff / p_c if p_c > 0 else float("nan")

    return ConversionResult(
        control_n=control_n,
        variant_n=variant_n,
        control_rate=p_c,
        variant_rate=p_v,
        absolute_uplift=diff,
        relative_uplift=relative,
        ci_low=ci_low,
        ci_high=ci_high,
        z_statistic=float(z),
        p_value=float(p_value),
        alpha=alpha,
        significant=bool(p_value < alpha),
    )
