"""ExperimentGuard: an A/B test decision engine for fintech payment-flow experiments.

The package is organised as a small pipeline of pure-ish statistical functions:

    data       -> loading, schema validation, synthetic data generation
    power      -> sample-size and minimum-detectable-effect calculators
    srm        -> sample-ratio-mismatch check (randomisation quality gate)
    effects    -> conversion uplift, confidence interval, two-proportion test
    bootstrap  -> bootstrap CI for transaction-value uplift
    guardrails -> payment-success and latency regression checks
    segments   -> per-segment analysis with multiple-testing correction
    decision   -> rules-based Ship / Do-not-ship / Continue engine
    report     -> one-page Markdown / HTML experiment report

Everything below the report layer returns plain dataclasses / dicts so the
functions stay easy to unit test.
"""

from .power import required_sample_size, minimum_detectable_effect, PowerResult
from .srm import check_srm, SRMResult
from .effects import conversion_uplift, ConversionResult
from .bootstrap import bootstrap_value_uplift, BootstrapResult
from .guardrails import check_guardrails, GuardrailResult
from .segments import segment_analysis, SegmentResult
from .decision import decide, Decision, DecisionInputs
from .report import build_report, build_html_report
from .pipeline import analyze, AnalysisResult

__all__ = [
    "required_sample_size",
    "minimum_detectable_effect",
    "PowerResult",
    "check_srm",
    "SRMResult",
    "conversion_uplift",
    "ConversionResult",
    "bootstrap_value_uplift",
    "BootstrapResult",
    "check_guardrails",
    "GuardrailResult",
    "segment_analysis",
    "SegmentResult",
    "decide",
    "Decision",
    "DecisionInputs",
    "build_report",
    "build_html_report",
    "analyze",
    "AnalysisResult",
]

__version__ = "0.1.0"
