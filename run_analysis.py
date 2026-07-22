"""Command-line entry point: analyse an experiment CSV and emit a report.

Examples:
    python run_analysis.py data/winner.csv
    python run_analysis.py data/srm.csv --html report.html
    python run_analysis.py data/winner.csv --mde 0.01 --alpha 0.05
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from experimentguard.pipeline import analyze
from experimentguard.report import build_report, build_html_report
from experimentguard.data import DataValidationError


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="ExperimentGuard A/B decision engine")
    parser.add_argument("csv", help="path to experiment CSV")
    parser.add_argument("--mde", type=float, default=0.01,
                        help="planned absolute minimum detectable effect (default 0.01)")
    parser.add_argument("--alpha", type=float, default=0.05, help="significance level")
    parser.add_argument("--power", type=float, default=0.8, help="target power")
    parser.add_argument("--html", metavar="PATH", help="also write an HTML report")
    parser.add_argument("--resamples", type=int, default=10_000,
                        help="bootstrap resamples (default 10000)")
    args = parser.parse_args(argv)

    try:
        df = pd.read_csv(args.csv, parse_dates=["event_date"])
    except FileNotFoundError:
        print(f"error: file not found: {args.csv}", file=sys.stderr)
        return 2

    try:
        result = analyze(
            df, planned_mde=args.mde, alpha=args.alpha, power=args.power,
            bootstrap_resamples=args.resamples,
        )
    except DataValidationError as exc:
        print(f"data validation failed: {exc}", file=sys.stderr)
        return 3

    print(build_report(result))

    if args.html:
        with open(args.html, "w") as fh:
            fh.write(build_html_report(result))
        print(f"\n[HTML report written to {args.html}]", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
