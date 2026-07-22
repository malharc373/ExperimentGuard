"""Bootstrap confidence interval for transaction-value uplift.

Transaction value is heavy-tailed and non-normal, so a t-interval on the mean is
unreliable. Instead we resample each arm with replacement, recompute the
difference in means many times, and read the confidence interval off the
percentiles of that bootstrap distribution. This is distribution-free and needs
no assumption about the shape of the value distribution.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BootstrapResult:
    control_mean: float
    variant_mean: float
    mean_difference: float
    ci_low: float
    ci_high: float
    confidence_level: float
    n_resamples: int
    significant: bool  # True when the CI excludes zero

    def summary(self) -> str:
        sig = "significant" if self.significant else "not significant"
        pct = int(round(self.confidence_level * 100))
        return (
            f"Mean value {self.control_mean:.2f} -> {self.variant_mean:.2f} "
            f"(diff {self.mean_difference:+.2f}), "
            f"{pct}% CI [{self.ci_low:+.2f}, {self.ci_high:+.2f}] ({sig})"
        )


def bootstrap_value_uplift(
    control_values,
    variant_values,
    n_resamples: int = 10_000,
    confidence_level: float = 0.95,
    random_state: int | None = 42,
) -> BootstrapResult:
    """Percentile bootstrap CI for the difference in mean transaction value.

    Args:
        control_values: 1-D array-like of control transaction values.
        variant_values: 1-D array-like of variant transaction values.
        n_resamples: number of bootstrap resamples.
        confidence_level: e.g. 0.95 for a 95% interval.
        random_state: seed for reproducibility (None for nondeterministic).

    Raises:
        ValueError: if either arm is empty or arguments are out of range.
    """
    control = np.asarray(control_values, dtype=float)
    variant = np.asarray(variant_values, dtype=float)
    if control.size == 0 or variant.size == 0:
        raise ValueError("both arms must contain at least one value")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be in (0, 1)")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")

    rng = np.random.default_rng(random_state)
    n_c, n_v = control.size, variant.size

    # Vectorised resampling: draw all resample indices at once.
    c_idx = rng.integers(0, n_c, size=(n_resamples, n_c))
    v_idx = rng.integers(0, n_v, size=(n_resamples, n_v))
    c_means = control[c_idx].mean(axis=1)
    v_means = variant[v_idx].mean(axis=1)
    diffs = v_means - c_means

    alpha = 1 - confidence_level
    ci_low, ci_high = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])

    return BootstrapResult(
        control_mean=float(control.mean()),
        variant_mean=float(variant.mean()),
        mean_difference=float(variant.mean() - control.mean()),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        significant=bool(ci_low > 0 or ci_high < 0),
    )
