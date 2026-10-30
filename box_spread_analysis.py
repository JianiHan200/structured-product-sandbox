#!/usr/bin/env python3
"""
Box-spread financing proxy: monthly series and chart (slide 5 of the Oct-15 update).

Input  (produced by pull_all.py):
    data/5_financing/spx_box_implied_financing.parquet   SPX box-implied rate per (date, expiry)
    data/3_risk_free/fred_treasury_curve.csv             Treasury constant-maturity yields
Output:
    data/5_financing/box_spread_monthly_1y.csv           monthly medians, bp
    data/5_financing/box_spread_vs_treasury.png          chart

Method
    r_box   = -ln(DF)/T, DF from regressing (C - P) on K across strikes (dividends cancel),
              computed in pull_all.box_rates(); expiries 6-18 months, >= 10 strikes.
    r_tsy   = Treasury CMT interpolated to the option's maturity (6M-1Y-2Y), converted from
              semi-annual bond-equivalent to continuous compounding: 2*ln(1 + y/200).
    spread  = r_box - r_tsy, in bp; monthly median across all (date, expiry) points.
    Reference: van Binsbergen, Diamond & Grotteria (2022), "Risk-free interest rates", JFE.

    python box_spread_analysis.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent / "data"


def monthly_spread() -> pd.DataFrame:
    b = pd.read_parquet(DATA / "5_financing" / "spx_box_implied_financing.parquet")
    b["date"] = pd.to_datetime(b["date"])
    s = b[(b["T"] > 0.5) & (b["T"] < 1.5) & (b["n"] >= 10)].copy()

    t = pd.read_csv(DATA / "3_risk_free" / "fred_treasury_curve.csv", parse_dates=["date"])
    s = s.merge(t[["date", "DGS6MO", "DGS1", "DGS2"]], on="date", how="left")
    s = s.dropna(subset=["DGS6MO", "DGS1", "DGS2"])
    y = np.where(s["T"] <= 1,
                 s.DGS6MO + (s.DGS1 - s.DGS6MO) * (s["T"] - 0.5) / 0.5,
                 s.DGS1 + (s.DGS2 - s.DGS1) * (s["T"] - 1))
    s["r_tsy"] = 2 * np.log(1 + y / 200)
    s["box_minus_tsy_bp"] = (s.r_box - s.r_tsy) * 1e4

    s["m"] = s.date.dt.to_period("M")
    m = s.groupby("m").agg(box_minus_tsy_bp=("box_minus_tsy_bp", "median"),
                           box_minus_om_zero_bp=("spread_bp", "median"),
                           box_rate=("r_box", "median"), tsy_rate=("r_tsy", "median"),
                           obs=("n", "size")).reset_index()
    m["m"] = m["m"].astype(str)
    return m


def chart(m: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = pd.PeriodIndex(m.m, freq="M").to_timestamp() + pd.Timedelta(days=14)
    v = m.box_minus_tsy_bp.to_numpy()
    roll = pd.Series(v, index=d).rolling(12, min_periods=6, center=True).mean()

    fig, ax = plt.subplots(figsize=(13, 5.5), dpi=150)
    for a, b in [("1998-08-01", "1998-12-31"), ("2007-08-01", "2009-03-31"), ("2020-03-01", "2020-04-30")]:
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color="#F3E3D3", lw=0)
    ax.axhline(0, color="#8A94A3", lw=1)
    ax.axhline(np.nanmean(v), color="#D9822B", lw=1.5, ls="--", label=f"Average {np.nanmean(v):.0f} bp")
    ax.plot(d, v, color="#9FB3C8", lw=1, label="Monthly median")
    ax.plot(roll.index, roll.values, color="#1F4E79", lw=2.5, label="12-month average")
    ax.set_ylabel("SPX box rate − Treasury (bp)")
    ax.set_title("Market-implied equity financing spread over Treasuries, 1996–2025", loc="left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, loc="upper right")
    ax.text(0, -0.13, "Source: OptionMetrics (WRDS), FRED. Shaded: 1998 LTCM, 2007–09 GFC, Mar–Apr 2020.",
            transform=ax.transAxes, fontsize=8, color="#6A7179")
    fig.tight_layout()
    fig.savefig(path)


def main() -> None:
    need = DATA / "5_financing" / "spx_box_implied_financing.parquet"
    if not need.exists():
        raise SystemExit(f"{need.relative_to(DATA.parent)} not found: run  python pull_all.py --user <WRDS username>  "
                         "first (the box-spread proxy needs OptionMetrics option quotes).")
    m = monthly_spread()
    out = DATA / "5_financing" / "box_spread_monthly_1y.csv"
    m.round(4).to_csv(out, index=False)
    print(f"-> {out.relative_to(DATA.parent)}  months={len(m)}")
    print(f"average {m.box_minus_tsy_bp.mean():.1f} bp; "
          f"peak {m.box_minus_tsy_bp.max():.0f} bp in {m.loc[m.box_minus_tsy_bp.idxmax(), 'm']}")
    try:
        png = DATA / "5_financing" / "box_spread_vs_treasury.png"
        chart(m, png)
        print(f"-> {png.relative_to(DATA.parent)}")
    except ImportError:
        print("(pip install matplotlib for the chart)")


if __name__ == "__main__":
    main()
