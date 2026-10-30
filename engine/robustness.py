"""
Robustness checks for the first-product back-test (Oct 28, version 2).

1. Non-overlapping cohorts. Monthly launches overlap (each covers 12 months), so adjacent
   vintages share 11 months of returns. Splitting them by launch month gives 12 cohorts of
   back-to-back, non-overlapping annual periods; the spread of results across cohorts shows
   how much depends on the launch month.
2. Moving-block bootstrap. Resampling 12-month blocks of the monthly vintage series keeps the
   overlap structure and gives confidence intervals for the headline statistics.
3. Fund check. For every completed outcome period of the four-leg SPY buffer funds with N-PORT
   data, compare the fund's actual NAV return with
     (a) the buffer payoff at the fund's actual cap (tests the payoff and fee model), and
     (b) the payoff at the engine's cap (SVI less the fitted execution cost; tests the whole engine).

Outputs (results/): robustness_cohorts.csv, robustness_bootstrap.csv, robustness_fund_check.csv,
                    robustness_fund_check.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.canonical import ROOT, connect  # noqa: E402
from engine.pricing import buffer_payoff  # noqa: E402

RESULTS = ROOT / "results"
FEES = {"First Trust (FT Vest)": 0.0085, "Innovator": 0.0079}
BLOCK, REPS, SEED = 12, 5000, 7


def _stats(x: pd.DataFrame) -> dict:
    return {"buffer_mean": x.buffer_return_net.mean(), "spx_tr_mean": x.spx_total_return.mean(),
            "buffer_vol": x.buffer_return_net.std(), "spx_tr_vol": x.spx_total_return.std(),
            "buffer_worst": x.buffer_return_net.min(), "buffer_neg_share": (x.buffer_return_net < 0).mean()}


def cohorts(v: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for b, g in v[v.buffer_return_net.notna()].groupby("buffer"):
        g = g.assign(m=pd.to_datetime(g.vintage).dt.month)
        per = pd.DataFrame([{"buffer": b, "launch_month": m, "years": len(x), **_stats(x)} for m, x in g.groupby("m")])
        rows.append(per)
    return pd.concat(rows, ignore_index=True)


def bootstrap(v: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    out = []
    for b, g in v[v.buffer_return_net.notna()].groupby("buffer"):
        g = g.sort_values("vintage").reset_index(drop=True)
        n = len(g)
        nblk = int(np.ceil(n / BLOCK))
        draws = []
        for _ in range(REPS):
            starts = rng.integers(0, n - BLOCK + 1, nblk)
            idx = np.concatenate([np.arange(s, s + BLOCK) for s in starts])[:n]
            x = g.iloc[idx]
            st = _stats(x)
            st["return_gap"] = st["buffer_mean"] - st["spx_tr_mean"]
            st["vol_ratio"] = st["buffer_vol"] / st["spx_tr_vol"]
            draws.append(st)
        d = pd.DataFrame(draws)
        point = _stats(g)
        point["return_gap"] = point["buffer_mean"] - point["spx_tr_mean"]
        point["vol_ratio"] = point["buffer_vol"] / point["spx_tr_vol"]
        for k in d.columns:
            out.append({"buffer": b, "statistic": k, "estimate": point[k],
                        "ci_low_95": d[k].quantile(0.025), "ci_high_95": d[k].quantile(0.975)})
    return pd.DataFrame(out)


def fund_check(con) -> pd.DataFrame:
    f = RESULTS / "caps_model_vs_actual.csv"
    if not f.exists():
        return pd.DataFrame()
    c = pd.read_csv(f, parse_dates=["expiry", "est_start", "model_date"])
    nav = con.sql("select date, ticker, nav from canon.fund_nav").df()
    px = con.sql("select date, ticker, adj_close from canon.fund_price").df()
    spx = con.sql("select date, close from canon.spot where underlying = 'SPX' order by date").df()
    for d in (nav, px, spx):
        d["date"] = pd.to_datetime(d["date"])
    spx = spx.set_index("date").close
    rows = []
    for r in c.itertuples():
        start, end = r.model_date, r.expiry
        if end > spx.index.max():
            continue
        s0, s1 = spx.asof(start), spx.asof(end)
        R = s1 / s0 - 1
        level, src = None, None
        n = nav[nav.ticker.eq(r.ticker)].set_index("date").nav.sort_index()
        p = px[px.ticker.eq(r.ticker)].set_index("date").adj_close.sort_index()
        for ser, name in ((n, "NAV"), (p, "price")):
            if len(ser) and ser.index.min() <= start + pd.Timedelta(days=5) and ser.index.max() >= end:
                level, src = ser, name
                break
        if level is None:
            continue
        fund = level.asof(end) / level.asof(start) - 1
        fee = FEES.get(r.issuer, 0.008) * (end - start).days / 365
        pay_actual = float(buffer_payoff(R, r.gross_cap_actual, r.buffer, r.buffer_start)) - fee
        cap_model = r.gross_cap_final if np.isfinite(r.gross_cap_final) else np.inf
        pay_model = float(buffer_payoff(R, cap_model, r.buffer, r.buffer_start)) - fee
        rows.append({"ticker": r.ticker, "issuer": r.issuer, "buffer": r.buffer, "start": start.date(),
                     "end": end.date(), "source": src, "spx_return": R, "fund_return": fund,
                     "payoff_actual_cap": pay_actual, "payoff_model_cap": pay_model,
                     "gap_actual_cap_pp": (fund - pay_actual) * 100, "gap_model_cap_pp": (fund - pay_model) * 100})
    return pd.DataFrame(rows)


def chart(fc: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.2, 5.6), dpi=150)
    ax.scatter(fc.payoff_model_cap * 100, fc.fund_return * 100, s=16, color="#1F4E79",
               label=f"Engine cap: gap {fc.gap_model_cap_pp.abs().mean():.2f} pp mean abs")
    ax.scatter(fc.payoff_actual_cap * 100, fc.fund_return * 100, s=10, color="#D9822B", alpha=0.7,
               label=f"Payoff at actual cap: gap {fc.gap_actual_cap_pp.abs().mean():.2f} pp")
    lo = float(min(fc.fund_return.min(), fc.payoff_model_cap.min())) * 100 - 3
    hi = float(max(fc.fund_return.max(), fc.payoff_model_cap.max())) * 100 + 3
    ax.plot([lo, hi], [lo, hi], color="#8A94A3", lw=0.8, ls="--")
    ax.set_xlabel("Predicted outcome-period return (%)")
    ax.set_ylabel("Actual fund return, NAV (%)")
    ax.set_title(f"Fund check: {len(fc)} completed outcome periods, {fc.ticker.nunique()} funds", loc="left",
                 fontsize=10.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(RESULTS / "robustness_fund_check.png")


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    v = pd.read_csv(RESULTS / "first_product_vintages.csv", parse_dates=["vintage"])
    co = cohorts(v)
    co.to_csv(RESULTS / "robustness_cohorts.csv", index=False)
    print("Non-overlapping cohorts (one per launch month): range across the 12 cohorts")
    print(co.groupby("buffer")[["buffer_mean", "spx_tr_mean", "buffer_vol", "buffer_worst"]]
          .agg(["min", "median", "max"]).round(4).to_string())
    bs = bootstrap(v)
    bs.to_csv(RESULTS / "robustness_bootstrap.csv", index=False)
    print(f"\nMoving-block bootstrap ({BLOCK}-month blocks, {REPS} draws): 95% intervals")
    print(bs.round(4).to_string(index=False))
    fc = fund_check(connect())
    fc.to_csv(RESULTS / "robustness_fund_check.csv", index=False)
    if len(fc):
        print(f"\nFund check: {len(fc)} completed periods, {fc.ticker.nunique()} funds")
        print(fc[["gap_actual_cap_pp", "gap_model_cap_pp"]].describe().round(2).to_string())
        chart(fc)
    print("\n-> results/robustness_cohorts.csv, robustness_bootstrap.csv, robustness_fund_check.csv, "
          "robustness_fund_check.png")
