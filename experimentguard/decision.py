"""Rules-based decision engine: Ship / Do not ship / Continue.

The engine is intentionally transparent — every verdict comes with the ordered
list of rules that fired, so a reviewer can see exactly *why* it landed on a
recommendation. The precedence is:

    1. Data-validity gate  : SRM detected            -> DO_NOT_SHIP (invalid randomisation)
    2. Safety gate         : guardrail breached      -> DO_NOT_SHIP
    3. Harm gate           : primary metric sig. down -> DO_NOT_SHIP
    4. Win                 : primary metric sig. up   -> SHIP
    5. Underpowered        : not sig. & n < required  -> CONTINUE
    6. Powered null        : not sig. & n >= required -> DO_NOT_SHIP (flat result)

Gates are evaluated in order and the first decisive rule wins.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .srm import SRMResult
from .effects import ConversionResult
from .guardrails import GuardrailResult


class Decision(str, Enum):
    SHIP = "Ship"
    DO_NOT_SHIP = "Do not ship"
    CONTINUE = "Continue"


@dataclass
class DecisionInputs:
    srm: SRMResult
    conversion: ConversionResult
    guardrails: GuardrailResult
    required_per_arm: int | None = None  # from the power analysis, if available


@dataclass
class DecisionOutcome:
    decision: Decision
    reasons: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return f"{self.decision.value} — " + "; ".join(self.reasons)


def decide(inputs: DecisionInputs) -> DecisionOutcome:
    """Apply the ordered decision rules and return the verdict plus rationale."""
    reasons: list[str] = []

    # 1. Data validity: never trust effects behind a broken randomisation.
    if inputs.srm.srm_detected:
        reasons.append(
            f"Sample-ratio mismatch (p={inputs.srm.p_value:.3g}); "
            "results are not trustworthy"
        )
        return DecisionOutcome(Decision.DO_NOT_SHIP, reasons)

    # 2. Safety: guardrails outrank a primary-metric win.
    if not inputs.guardrails.passed:
        reasons.append("Guardrail breached: " + "; ".join(inputs.guardrails.breaches))
        return DecisionOutcome(Decision.DO_NOT_SHIP, reasons)

    conv = inputs.conversion

    # 3. Harm: a significant *drop* in the primary metric.
    if conv.significant and conv.absolute_uplift < 0:
        reasons.append(
            f"Primary metric significantly down ({conv.absolute_uplift:+.4f}, "
            f"p={conv.p_value:.3g})"
        )
        return DecisionOutcome(Decision.DO_NOT_SHIP, reasons)

    # 4. Win: a significant lift with guardrails intact.
    if conv.significant and conv.absolute_uplift > 0:
        reasons.append(
            f"Primary metric significantly up ({conv.absolute_uplift:+.4f}, "
            f"p={conv.p_value:.3g}); guardrails intact"
        )
        return DecisionOutcome(Decision.SHIP, reasons)

    # 5 & 6. Not significant: distinguish underpowered from a genuine flat result.
    reasons.append(
        f"Primary metric not significant (p={conv.p_value:.3g}, "
        f"CI [{conv.ci_low:+.4f}, {conv.ci_high:+.4f}])"
    )
    observed_per_arm = min(conv.control_n, conv.variant_n)
    if inputs.required_per_arm is not None and observed_per_arm < inputs.required_per_arm:
        reasons.append(
            f"Underpowered: {observed_per_arm} < {inputs.required_per_arm} "
            "required per arm; keep collecting"
        )
        return DecisionOutcome(Decision.CONTINUE, reasons)

    reasons.append("Adequately powered but no effect detected; do not ship")
    return DecisionOutcome(Decision.DO_NOT_SHIP, reasons)
