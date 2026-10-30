"""
Equity financing spread measured against SOFR (sponsor request, Oct 2026).

Spread = SPX box-spread implied rate  -  overnight funding benchmark, same date.
Benchmark: SOFR from 2 Apr 2018; before that EFFR (from Jul 2000) and the effective fed
funds rate (before Jul 2000), shifted by the average SOFR-EFFR basis measured over the
overlap, so the series is on a SOFR-equivalent basis throughout.

Two tenor buckets are reported:
  short : box expiries of 1-3 months
  1y    : box expiries of 6-18 months (matches the annual products)

Version 2 (Oct 28): tenor-matched benchmark. Comparing a 1-year box rate with today's
overnight SOFR mixes in the market's expectation of where SOFR is heading (+170 bp in 2022,
-70 bp in 2024 in version 1). Each box rate is now also compared with a SOFR rate for the SAME
period, in this order of preference:
  1. canon.ois_curve, when the feed provides SOFR OIS / Term SOFR (e.g. Bloomberg USOSFR*):
     the market's ex-ante term rate, interpolated to the box's exact tenor;
  2. otherwise the SOFR actually realised over the box's life, compounded daily (EFFR + basis
     before 2018). This is the rate an OIS would have settled at; ex post it differs from the
     ex-ante OIS rate only by the rate surprise, which averages out over many observations.
     Only boxes that have already expired can be measured this way.
Rates are continuously compounded; SOFR (act/360, simple) is converted accordingly.

Outputs (results/): financing_spread_sofr_monthly.csv, financing_spread_sofr.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.canonical import ROOT, connect  # noqa: E402

RESULTS = ROOT / "results"
BUCKETS = {"short": (25 / 365, 95 / 365), "1y": (0.5, 1.5)}


def sofr_equivalent(con) -> tuple[pd.DataFrame, float]:
    f = con.sql("select * from canon.funding order by date").df()
    f["date"] = pd.to_datetime(f["date"])
    f = f.set_index("date").sort_index()
    overlap = f.dropna(subset=["sofr", "effr"])
    basis = float((overlap["sofr"] - overlap["effr"]).mean())
    pre = f["effr"].combine_first(f["fed_funds"])
    bench = f["sofr"].combine_first(pre + basis)
    out = pd.DataFrame({"bench_simple": bench,
                        "bench_source": np.where(f["sofr"].notna(), "SOFR",
                                                 np.where(f["effr"].notna(), "EFFR+basis", "FedFunds+basis"))})
    out["bench_cc"] = np.log1p(out["bench_simple"] / 360) * 365
    return out.dropna(subset=["bench_simple"]), basis


def term_benchmark(con, bench: pd.DataFrame, b: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """SOFR-equivalent rate (continuously compounded) for each box's own period [date, expiry)."""
    has_ois = con.sql("select count(*) from information_schema.tables where table_schema = 'canon' "
                      "and table_name = 'ois_curve'").fetchone()[0] > 0
    rate = pd.Series(np.nan, index=b.index)
    source = pd.Series("", index=b.index)
    if has_ois:
        o = con.sql("select date, tenor_days, rate from canon.ois_curve").df()
        o["date"] = pd.to_datetime(o["date"])
        curves = {d: g.sort_values("tenor_days") for d, g in o.groupby("date")}
        for i, r in b.iterrows():
            g = curves.get(r["date"])
            if g is not None:
                days = r["tenor_years"] * 365
                simple = float(np.interp(days, g.tenor_days, g.rate))
                rate[i], source[i] = np.log1p(simple * days / 360) / r["tenor_years"], "OIS"
    # realised compounding: every calendar day accrues the last published rate / 360
    daily = bench["bench_simple"].resample("D").ffill()
    cum = np.log1p(daily / 360).cumsum()
    last = daily.index.max()
    start = pd.to_datetime(b["date"]) - pd.Timedelta(days=1)
    end = pd.to_datetime(b["expiry"]) - pd.Timedelta(days=1)
    ok = rate.isna() & (end <= last) & (start >= cum.index.min())
    L1 = cum.reindex(start[ok]).to_numpy()
    L2 = cum.reindex(end[ok]).to_numpy()
    rate[ok] = (L2 - L1) / b.loc[ok, "tenor_years"].to_numpy()
    source[ok] = "realised"
    return rate, source


def spread_series(con) -> tuple[pd.DataFrame, float]:
    bench, basis = sofr_equivalent(con)
    b = con.sql("select * from canon.box_financing where n_strikes >= 10").df()
    b["date"] = pd.to_datetime(b["date"])
    b["expiry"] = pd.to_datetime(b["expiry"])
    b = b.merge(bench, left_on="date", right_index=True, how="inner").reset_index(drop=True)
    b["term_cc"], b["term_source"] = term_benchmark(con, bench, b)
    frames = []
    for name, (lo, hi) in BUCKETS.items():
        x = b[(b.tenor_years >= lo) & (b.tenor_years <= hi)].copy()
        x["spread_bp"] = (x["r_box"] - x["bench_cc"]) * 1e4             # v1: vs overnight rate
        x["spread_term_bp"] = (x["r_box"] - x["term_cc"]) * 1e4          # v2: vs same-tenor rate
        x = x[x["spread_bp"].abs() < 500]  # drop broken quotes
        x.loc[x["spread_term_bp"].abs() >= 500, "spread_term_bp"] = np.nan
        x["month"] = x["date"].dt.to_period("M").astype(str)
        g = x.groupby("month").agg(spread_bp=("spread_bp", "median"), spread_term_bp=("spread_term_bp", "median"),
                                   r_box=("r_box", "median"), bench=("bench_cc", "median"),
                                   term_bench=("term_cc", "median"), obs=("spread_bp", "size"),
                                   bench_source=("bench_source", "last"),
                                   term_source=("term_source", "last")).reset_index()
        g["bucket"] = name
        frames.append(g)
    return pd.concat(frames, ignore_index=True), basis


def period_table(m: pd.DataFrame) -> pd.DataFrame:
    periods = {"1996-2007": ("1996-01", "2007-12"), "GFC 2008-09": ("2008-01", "2009-06"),
               "2010-2017": ("2010-01", "2017-12"), "SOFR era 2018-2025": ("2018-04", "2025-12"),
               "  of which 2023-2025": ("2023-01", "2025-12"), "Full sample": ("1996-01", "2025-12")}
    rows = []
    for p, (a, z) in periods.items():
        r = {"period": p}
        for bkt in BUCKETS:
            x = m[(m.bucket == bkt) & (m.month >= a) & (m.month <= z)]
            r[f"{bkt}_vs_overnight_bp"] = round(x["spread_bp"].mean(), 1)
            r[f"{bkt}_vs_term_bp"] = round(x["spread_term_bp"].mean(), 1)
        rows.append(r)
    return pd.DataFrame(rows)


def chart(m: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 4.8), dpi=150)
    for a, z in [("1998-08", "1998-12"), ("2007-08", "2009-03"), ("2020-03", "2020-04")]:
        ax.axvspan(pd.Period(a).to_timestamp(), pd.Period(z).to_timestamp(), color="#EEE6DC", lw=0)
    ax.axvline(pd.Timestamp("2018-04-02"), color="#8A94A3", lw=1, ls=":")
    ax.text(pd.Timestamp("2018-06-01"), 0.97, "SOFR from Apr 2018\n(EFFR + basis before)", transform=ax.get_xaxis_transform(),
            va="top", fontsize=8, color="#6A7179")
    ax.axhline(0, color="#8A94A3", lw=1)
    for bkt, col, color, lw, ls, label in [
            ("1y", "spread_bp", "#B8BEC6", 1.0, "--", "6-18 months vs overnight SOFR (v1)"),
            ("1y", "spread_term_bp", "#D9822B", 1.6, "-", "6-18 months vs same-tenor SOFR (v2)"),
            ("short", "spread_term_bp", "#1F4E79", 1.8, "-", "1-3 months vs same-tenor SOFR (v2)")]:
        s = m[m.bucket == bkt]
        d = pd.PeriodIndex(s.month, freq="M").to_timestamp()
        ax.plot(d, s[col].rolling(3, min_periods=1, center=True).mean(), color=color, lw=lw, ls=ls, label=label)
    ax.set_ylabel("Box rate − SOFR (bp), 3-month smoothed")
    ax.set_title("Equity financing spread over SOFR, SPX box spreads, 1996–2025", loc="left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(path)


if __name__ == "__main__":
    RESULTS.mkdir(exist_ok=True)
    m, basis = spread_series(connect())
    m.to_csv(RESULTS / "financing_spread_sofr_monthly.csv", index=False)
    print(f"SOFR - EFFR basis over the overlap: {basis * 1e4:.1f} bp (added to EFFR / fed funds before Apr 2018)\n")
    print(period_table(m).to_string(index=False))
    chart(m, RESULTS / "financing_spread_sofr.png")
    print(f"\n-> {RESULTS / 'financing_spread_sofr_monthly.csv'}\n-> {RESULTS / 'financing_spread_sofr.png'}")
