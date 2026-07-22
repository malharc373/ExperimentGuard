"""Sample-Ratio-Mismatch (SRM) check.

SRM means the observed split between control and variant differs from the split
you *intended* (e.g. you configured 50/50 but observed 50.8/49.2). It is a
randomisation-quality problem: a mismatch this size by chance is very unlikely,
so its presence usually points to a logging bug, a redirect that drops users, or
bot filtering that hits one arm harder. Microsoft's experimentation guidance is
to pass an SRM check *before* trusting any effect analysis.

We test observed arm counts against the expected split with a chi-square
goodness-of-fit test and flag a mismatch when p < threshold (default 0.001,
the widely used SRM alarm level).
"""

from __future__ import annotations

from dataclasses import dataclass

from scipy import stats


@dataclass
class SRMResult:
    control_count: int
    variant_count: int
    expected_control_ratio: float
    observed_control_ratio: float
    chi_square: float
    p_value: float
    threshold: float
    srm_detected: bool

    def summary(self) -> str:
        verdict = "SRM DETECTED" if self.srm_detected else "no SRM"
        return (
            f"{verdict}: observed control ratio "
            f"{self.observed_control_ratio:.4f} vs expected "
            f"{self.expected_control_ratio:.4f} "
            f"(chi2={self.chi_square:.3f}, p={self.p_value:.4g})"
        )


def check_srm(
    control_count: int,
    variant_count: int,
    expected_control_ratio: float = 0.5,
    threshold: float = 0.001,
) -> SRMResult:
    """Chi-square goodness-of-fit test of arm allocation vs the intended split.

    Args:
        control_count: number of users assigned to control.
        variant_count: number of users assigned to variant.
        expected_control_ratio: intended fraction of users in control (0..1).
        threshold: p-value below which SRM is flagged.

    Raises:
        ValueError: on non-positive totals or a ratio outside (0, 1).
    """
    if control_count < 0 or variant_count < 0:
        raise ValueError("counts must be non-negative")
    total = control_count + variant_count
    if total == 0:
        raise ValueError("total sample size must be positive")
    if not 0 < expected_control_ratio < 1:
        raise ValueError("expected_control_ratio must be in (0, 1)")

    expected_control = total * expected_control_ratio
    expected_variant = total * (1 - expected_control_ratio)

    chisq, p_value = stats.chisquare(
        f_obs=[control_count, variant_count],
        f_exp=[expected_control, expected_variant],
    )

    return SRMResult(
        control_count=control_count,
        variant_count=variant_count,
        expected_control_ratio=expected_control_ratio,
        observed_control_ratio=control_count / total,
        chi_square=float(chisq),
        p_value=float(p_value),
        threshold=threshold,
        srm_detected=bool(p_value < threshold),
    )
