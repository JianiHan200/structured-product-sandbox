#!/usr/bin/env python3
"""
Run every analysis in order on whatever feed engine.canonical points to.

    python engine/run_all.py            # outputs go to results/

1. fund_master       one master list of funds (three fund types)
2. financing         equity financing spread over SOFR
3. historical_caps   actual caps from N-PORT, model caps, validation, monthly cap history
4. first_product     annual S&P 500 buffers at 10% / 15%: terms, cost decomposition, outcomes
5. robustness        non-overlapping cohorts, block bootstrap, check against actual fund NAVs

The SVI smile is fitted on first use and cached in data/derived/ (about 5-10 minutes the first
time on a laptop; `python engine/surface.py` pre-fits it on its own).
"""
import runpy
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = ["fund_master", "financing", "historical_caps", "first_product", "robustness"]

if __name__ == "__main__":
    for s in sys.argv[1:] or STEPS:
        t = time.time()
        print(f"\n===== {s} =====", flush=True)
        runpy.run_path(str(HERE / f"{s}.py"), run_name="__main__")
        print(f"({time.time() - t:.0f}s)", flush=True)
