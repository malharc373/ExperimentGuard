"""Generate sample experiment CSVs for demos and tests.

Writes three scenarios into ``data/``:
    * winner.csv         — mild significant conversion lift, guardrails intact
    * srm.csv            — broken randomisation (55/45 split)
    * guardrail_fail.csv — conversion up but latency regressed

Run: ``python scripts/generate_sample_data.py``
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experimentguard.data import generate_synthetic_experiment

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)

    winner = generate_synthetic_experiment(seed=1)
    winner.to_csv(os.path.join(DATA_DIR, "winner.csv"), index=False)

    srm = generate_synthetic_experiment(control_ratio=0.55, seed=11)
    srm.to_csv(os.path.join(DATA_DIR, "srm.csv"), index=False)

    guardrail = generate_synthetic_experiment(
        variant_latency_ms=310.0,  # ~29% latency regression
        seed=13,
    )
    guardrail.to_csv(os.path.join(DATA_DIR, "guardrail_fail.csv"), index=False)

    print(f"Wrote winner.csv, srm.csv, guardrail_fail.csv to {DATA_DIR}")


if __name__ == "__main__":
    main()
