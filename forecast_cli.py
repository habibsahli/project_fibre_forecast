#!/usr/bin/env python3
"""Simple CLI wrapper for the forecasting pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys


def run_forecast(args) -> int:
    cmd = [sys.executable, "-m", "src.forecasting.forecasting_pipeline"]
    if args.no_lstm:
        cmd.append("--no-lstm")
    if args.extra_args:
        cmd.extend(args.extra_args)
    return subprocess.call(cmd)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="forecast",
        description="CLI wrapper for fibre forecasting pipeline",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run forecasting pipeline")
    run_parser.add_argument("--no-lstm", action="store_true", help="Skip LSTM training")
    run_parser.add_argument("extra_args", nargs=argparse.REMAINDER, help="Extra args for pipeline")
    run_parser.set_defaults(func=run_forecast)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
