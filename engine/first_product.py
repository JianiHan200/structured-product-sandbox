"""
First product (sponsor guidance, Oct 2026): annual S&P 500 buffers at 10% and 15%.

For every monthly vintage from 1996 to Aug 2025 (first trading day of each month):
  1. Terms   - gross cap that the option surface supports, funded by the forgone
               dividends (engine.pricing.solve_cap) on the SVI smile (engine.surface), less an
               execution haircut estimated on the calibration half of the N-PORT sample
               (results/caps_model_vs_actual.csv); net cap = gross - fee.
  2. Cost    - embedded cost decomposition per 100 of notional at inception:
               protection bought (put spread), cap give-up (value of the upside sold),
               forgone dividends, management fee, and the financing spread over SOFR.
  3. Outcome - realised one-year return of the buffer versus the S&P 500 price and total
               return, classified as capped / participating / fully buffered / loss beyond buffer.
Monthly vintages overlap (each covers 12 months), so outcome statistics are descriptive.

Outputs (results/): first_product_vintages.csv, first_product_summary.csv, first_product_costs.csv,
                    first_product_outcomes.png, first_product_caps.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.canonical import ROOT, connect  # noqa: E402
from engine.pricing import buffer_payoff, dividend_budget, solve_cap  # noqa: E402
from engine.historical_caps import haircut_pp, load_cost_model  # noqa: E402
from engine.surface import load_markets_svi, monthly_starts, save_cache  # noqa: E402

RESULTS = ROOT / "results"
BUFFERS = (0.10, 0.15)
FEE = 0.0079            # expense ratio, typical of the listed comparables (Innovator 0.79%, FT Vest 0.85%)
CATS = ["Capped", "Participating", "Fully buffered", "Loss beyond buffer"]


def execution_haircut(m) -> float:
    """Execution cost in cap terms for market m (decimal), from the model fitted on the calibration
    half of the N-PORT sample (engine.historical_caps); zero when it has not been fitted."""
    cm = load_cost_model()
    return float(haircut_pp(cm, m.r, m.vol(m.F))) / 100 if cm else 0.0


def financing_spread() -> pd.Series:
    """1-year box rate over same-tenor SOFR (v2); falls back to the short bucket vs overnight SOFR."""
    f = RESULTS / "financing_spread_sofr_monthly.csv"
    if not f.exists():
        return pd.Series(dtype=float)
    m = pd.read_csv(f)
    y = m[m.bucket.eq("1y")].set_index("month")
    s = m[m.bucket.eq("short")].set_index("month").spread_bp
    term = y["spread_term_bp"] if "spread_term_bp" in y else pd.Series(dtype=float)
    return term.combine_first(s) / 1e4


def vintages(con) -> pd.DataFrame:
    starts = monthly_starts(con, "SPX")
    mk = load_markets_svi(con, "SPX", starts)
    save_cache()
    spot = con.sql("select date, close from canon.spot where underlying = 'SPX' order by date").df()
    spot = pd.Series(spot.close.to_numpy(), index=pd.to_datetime(spot.date))
    tr = con.sql("select date, tr_level from canon.total_return_index where underlying = 'SPX' order by date").df()
    tr = pd.Series(tr.tr_level.to_numpy(), index=pd.to_datetime(tr.date))
    vix = con.sql("select date, vix from canon.vol_index order by date").df()
    vix = pd.Series(vix.vix.to_numpy(), index=pd.to_datetime(vix.date))
    fin = financing_spread()

    rows = []
    for d in starts:
        if d not in mk:
            continue
        m = mk[d]
        haircut = execution_haircut(m)
        end = d + pd.DateOffset(years=1)
        done = end <= spot.index.max()
        R = spot.asof(end) / m.S - 1 if done else np.nan
        R_tr = tr.asof(end) / tr.asof(d) - 1 if done and d >= tr.index.min() else np.nan
        for b in BUFFERS:
            cap_mid = solve_cap(m, b)
            cap_g = cap_mid - haircut if np.isfinite(cap_mid) else np.inf
            prot = m.price("P", 1.0) - m.price("P", 1 - b)
            giveup = m.price("C", 1 + cap_g) if np.isfinite(cap_g) else 0.0
            s_fin = float(fin.get(f"{d:%Y-%m}", np.nan))
            gross = float(buffer_payoff(R, cap_g, b)) if done else np.nan
            net = gross - FEE if done else np.nan
            cat = (np.nan if not done else CATS[0] if R >= cap_g else CATS[1] if R >= 0
                   else CATS[2] if R >= -b else CATS[3])
            rows.append({
                "vintage": d, "buffer": b, "spx": m.S, "atm_vol": m.vol(m.F), "r": m.r, "q": m.q,
                "vix": float(vix.asof(d)) if len(vix) else np.nan,
                "gross_cap_mid": cap_mid, "execution_haircut": haircut, "gross_cap": cap_g, "net_cap": cap_g - FEE,
                "cost_protection_bought": prot, "cost_cap_giveup": giveup,
                "cost_forgone_dividends": dividend_budget(m), "cost_fee": FEE,
                "cost_financing_spread": s_fin * m.T,
                "spx_price_return": R, "spx_total_return": R_tr,
                "buffer_return_net": net, "outcome": cat})
    return pd.DataFrame(rows)


def summary(v: pd.DataFrame) -> pd.DataFrame:
    out = []
    regimes = {"All": lambda x: x, "VIX < 15": lambda x: x[x.vix < 15],
               "VIX 15-25": lambda x: x[(x.vix >= 15) & (x.vix < 25)], "VIX >= 25": lambda x: x[x.vix >= 25]}
    for b in BUFFERS:
        for reg, f in regimes.items():
            x = f(v[(v.buffer == b) & v.buffer_return_net.notna()])
            if not len(x):
                continue
            fin_cap = x.gross_cap.replace(np.inf, np.nan)
            r = {"buffer": f"{b:.0%}", "regime": reg, "vintages": len(x),
                 "median_net_cap": fin_cap.median() - FEE, "uncapped_share": np.isinf(x.gross_cap).mean(),
                 "buffer_mean": x.buffer_return_net.mean(), "spx_tr_mean": x.spx_total_return.mean(),
                 "spx_price_mean": x.spx_price_return.mean(),
                 "buffer_vol": x.buffer_return_net.std(), "spx_tr_vol": x.spx_total_return.std(),
                 "buffer_worst": x.buffer_return_net.min(), "spx_tr_worst": x.spx_total_return.min(),
                 "buffer_neg_share": (x.buffer_return_net < 0).mean(),
                 "spx_tr_neg_share": (x.spx_total_return < 0).mean()}
            for c in CATS:
                r[f"share_{c.lower().replace(' ', '_')}"] = (x.outcome == c).mean()
            out.append(r)
    return pd.DataFrame(out)


def cost_table(v: pd.DataFrame) -> pd.DataFrame:
    cols = ["cost_protection_bought", "cost_cap_giveup", "cost_forgone_dividends", "cost_fee", "cost_financing_spread"]
    rows = []
    for b in BUFFERS:
        x = v[v.buffer == b]
        for label, y in [("1996-2025 average", x), ("2023-2025 average", x[x.vintage >= "2023-01-01"])]:
            rows.append({"buffer": f"{b:.0%}", "period": label, **(y[cols].mean() * 100).round(2).to_dict()})
    return pd.DataFrame(rows)


def charts(v: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"Capped": "#1F4E79", "Participating": "#7FA7C9", "Fully buffered": "#D9822B", "Loss beyond buffer": "#B23A2E"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150, sharey=True)
    for ax, b in zip(axes, BUFFERS):
        x = v[(v.buffer == b) & v.buffer_return_net.notna()]
        for c in CATS:
            s = x[x.outcome == c]
            ax.scatter(s.spx_price_return * 100, s.buffer_return_net * 100, s=10, color=colors[c],
                       label=f"{c} ({len(s) / len(x):.0%})")
        lim = [-50, 70]
        ax.plot(lim, lim, color="#B8BEC6", lw=0.8, ls="--")
        ax.axhline(0, color="#8A94A3", lw=0.8)
        ax.axvline(0, color="#8A94A3", lw=0.8)
        ax.set_xlim(lim)
        ax.set_title(f"{b:.0%} buffer: {len(x)} monthly vintages, 1996–2025", loc="left", fontsize=11)
        ax.set_xlabel("S&P 500 price return over the year (%)")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    axes[0].set_ylabel("Buffer return, net of 0.79% fee (%)")
    fig.suptitle("Annual S&P 500 buffers: realised one-year outcomes by vintage", x=0.01, ha="left", fontsize=13)
    fig.tight_layout()
    fig.savefig(RESULTS / "first_product_outcomes.png")

    fig, ax = plt.subplots(figsize=(12, 4.5), dpi=150)
    for b, col in [(0.10, "#1F4E79"), (0.15, "#D9822B")]:
        x = v[v.buffer == b].set_index("vintage")
        ax.plot(x.index, x.gross_cap.replace(np.inf, np.nan).clip(upper=0.45) * 100 - FEE * 100, color=col, lw=1.8,
                label=f"Net cap, {b:.0%} buffer")
    ax2 = ax.twinx()
    x = v[v.buffer == 0.10].set_index("vintage")
    ax2.fill_between(x.index, x.vix, color="#B8BEC6", alpha=0.35, lw=0)
    ax2.set_ylabel("VIX at launch (shaded)", color="#6A7179")
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax.set_ylabel("Net cap (%), execution-adjusted")
    ax.set_title("What the option market would have paid: net caps for annual S&P 500 buffers by launch month",
                 loc="left", fontsize=11)
    ax.legend(frameon=False, loc="upper right")
    for a in (ax, ax2):
        a.spines["top"].set_visible(False)
    fig.tight_layout()
    fig.savefig(RESULTS / "first_product_caps.png")


if __name__ == "__main__":
    RESULTS.mkdir(exist_ok=True)
    con = connect()
    v = vintages(con)
    v.to_csv(RESULTS / "first_product_vintages.csv", index=False)
    s = summary(v)
    s.to_csv(RESULTS / "first_product_summary.csv", index=False)
    pd.set_option("display.width", 220)
    print(s.round(4).to_string(index=False))
    print("\nEmbedded cost per 100 notional (annual, %):")
    ct = cost_table(v)
    ct.to_csv(RESULTS / "first_product_costs.csv", index=False)
    print(ct.to_string(index=False))
    charts(v)
    print("\n-> results/first_product_vintages.csv, first_product_summary.csv, first_product_costs.csv, "
          "first_product_outcomes.png, first_product_caps.png")
