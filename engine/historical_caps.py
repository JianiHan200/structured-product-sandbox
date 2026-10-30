"""
Rebuild historical caps (sponsor priority, Oct 2026). Two independent routes:

1. ACTUAL caps from SEC N-PORT. Each filing lists the fund's FLEX legs. For one outcome
   period (one expiry): the purchased put sits at the buffer start, the written put at the
   buffer end, the written call at the cap. Reference price S0 = purchased-put strike /
   (1 - buffer start). Gross cap = written-call strike / S0 - 1.
2. MODEL caps: on the period's start date, solve the cap at which forgone dividends fund the
   overlay, pricing each FLEX leg off the volatility smile (engine.pricing).

Version 2 (Oct 28):
  * smile: SVI fitted to listed quotes (engine.surface), priced at the fund's EXACT expiry;
    the first version's linear delta-grid smile at a fixed 365-day tenor is kept for comparison;
  * validation sample: every SPY-referenced 1-year, four-leg buffer with N-PORT data
    (Innovator B / P / U and FT Vest F / G / D families), not five funds;
  * calibration and test kept apart: the execution-cost model is estimated on the earlier half
    of outcome periods and the error is reported on the later half only;
  * execution cost: the model-minus-actual gap is not constant. It rises with the level of
    rates and falls with volatility, so the haircut is a small linear model in (zero rate,
    ATM vol) fitted on the calibration half, inputs held inside their calibrated range:
        haircut_pp = c0 + c_r * r + c_vol * atm_vol      (results/execution_cost_model.csv)

Outputs (results/): caps_actual_nport.csv, caps_model_vs_actual.csv, caps_validation_summary.csv,
                    caps_model_history.csv, caps_history.png, caps_validation.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.canonical import ROOT, connect  # noqa: E402
from engine.pricing import UNDERLYING_FOR, load_markets, on_or_before, solve_cap, trading_dates  # noqa: E402
from engine.surface import load_markets_svi, monthly_starts, save_cache  # noqa: E402

RESULTS = ROOT / "results"
COST_MODEL = RESULTS / "execution_cost_model.csv"
FEES = {"First Trust (FT Vest)": 0.0085, "Innovator": 0.0079}
MODEL_BUFFERS = {"9% (Innovator B)": 0.09, "10% (FT Vest F)": 0.10, "15% (Innovator P / FT Vest G)": 0.15}
SIMPLE_BUFFERS = ("Buffer", "Moderate Buffer", "Deep Buffer", "Ultra buffer (-5% to -35%)")  # four-leg structures


def buffer_start(ticker: str, sub_strategy: str, terms_start: dict) -> float:
    """Where the buffer starts (0 for a standard buffer). Issuer terms first; Innovator's
    Ultra series (-5% to -35%) carry no buffer start in its term table."""
    v = terms_start.get(ticker)
    if v is not None and np.isfinite(v):
        return abs(float(v))
    return 0.05 if str(sub_strategy).startswith("Ultra") else 0.0


def actual_caps(con, master: pd.DataFrame) -> pd.DataFrame:
    legs = con.sql("select * from canon.fund_option_legs").df()
    terms = con.sql("""select ticker, buffer_start from canon.fund_terms
                       where snapshot = (select max(snapshot) from canon.fund_terms)""").df()
    tstart = dict(zip(terms.ticker, pd.to_numeric(terms.buffer_start, errors="coerce")))
    buffers = master[master.fund_type.eq("Buffer / defined outcome") & master.sub_strategy.isin(SIMPLE_BUFFERS)]
    buffers = buffers.set_index("ticker")
    rows = []
    for (t, exp), g in legs[legs.ticker.isin(buffers.index)].groupby(["ticker", "expiry"]):
        lp = g[(g.put_call == "Put") & (g.side == "Purchased")].strike
        sp = g[(g.put_call == "Put") & (g.side == "Written")].strike
        sc = g[(g.put_call == "Call") & (g.side == "Written")].strike
        if lp.empty or sc.empty:
            continue
        f = buffers.loc[t]
        a = buffer_start(t, f.sub_strategy, tstart)
        s0 = lp.max() / (1 - a)
        rows.append({
            "ticker": t, "issuer": f.issuer, "sub_strategy": f.sub_strategy, "buffer_pct": f.buffer_pct,
            "ref_ticker": g.ref_ticker.dropna().iloc[0] if g.ref_ticker.notna().any() else None,
            "expiry": pd.Timestamp(exp), "est_start": pd.Timestamp(exp) - pd.Timedelta(days=364),
            "ref_price_s0": s0, "buffer_start": a,
            "buffer": (lp.max() - sp.min()) / s0 if not sp.empty else np.nan,
            "gross_cap_actual": sc.min() / s0 - 1, "first_report": g.report_date.min()})
    out = pd.DataFrame(rows)
    # plausible 1-year buffer periods only (drops partial filings and stray legs)
    ok = out.buffer.between(0.04, 0.40) & out.gross_cap_actual.between(0.0, 0.8)
    return out[ok].sort_values(["ticker", "expiry"]).reset_index(drop=True)


def model_vs_actual(con, act: pd.DataFrame) -> pd.DataFrame:
    out = []
    for ref, g in act.groupby("ref_ticker"):
        und = UNDERLYING_FOR.get(ref)
        if und is None:
            continue
        idx = trading_dates(con, und)
        g = g[g.est_start <= idx.max()].copy()
        g["model_date"] = [on_or_before(idx, d) for d in g.est_start]
        g = g[g.model_date.notna() & g.buffer.notna()]
        g["T"] = (g.expiry - g.model_date).dt.days / 365.0
        lin = load_markets(con, und, sorted(set(g.model_date)))                 # v1: delta grid, 365 days
        svi = load_markets_svi(con, und, list(zip(g.model_date, g["T"])))     # v2: SVI, exact expiry
        for r in g.itertuples():
            row = r._asdict()
            row.pop("Index")
            if r.model_date in lin:
                row["gross_cap_linear"] = solve_cap(lin[r.model_date], r.buffer, r.buffer_start)
            m = svi.get((r.model_date, r.T))
            if m is not None:
                row["gross_cap_svi"] = solve_cap(m, r.buffer, r.buffer_start)
                row["r"], row["atm_vol"] = m.r, m.vol(m.F)
            out.append(row)
    save_cache()
    df = pd.DataFrame(out)
    for k in ("linear", "svi"):
        df[f"error_{k}_pp"] = (df[f"gross_cap_{k}"].replace(np.inf, np.nan) - df.gross_cap_actual) * 100
    split = df.est_start.median()
    df["sample"] = np.where(df.est_start < split, "calibration", "test")
    return df


def fit_cost_model(cal: pd.DataFrame) -> dict:
    """Execution haircut (pp of cap) = c0 + c_r * r + c_vol * atm_vol, least squares on calibration."""
    x = cal.dropna(subset=["error_svi_pp", "r", "atm_vol"])
    X = np.column_stack([np.ones(len(x)), x.r, x.atm_vol])
    c0, c_r, c_vol = np.linalg.lstsq(X, x.error_svi_pp, rcond=None)[0]
    return {"c0": c0, "c_r": c_r, "c_vol": c_vol, "r_min": x.r.min(), "r_max": x.r.max(),
            "vol_min": x.atm_vol.min(), "vol_max": x.atm_vol.max(), "n": len(x),
            "from": x.est_start.min().date(), "to": x.est_start.max().date()}


def haircut_pp(model: dict, r, vol):
    """Haircut in pp of cap; inputs clipped to the calibrated range, never negative."""
    r = np.clip(r, model["r_min"], model["r_max"])
    vol = np.clip(vol, model["vol_min"], model["vol_max"])
    return np.maximum(model["c0"] + model["c_r"] * r + model["c_vol"] * vol, 0.0)


def load_cost_model() -> dict | None:
    return pd.read_csv(COST_MODEL).iloc[0].to_dict() if COST_MODEL.exists() else None


def validation_summary(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    cm = fit_cost_model(df[df["sample"].eq("calibration")])
    df["haircut_pp"] = haircut_pp(cm, df.r, df.atm_vol)
    df["gross_cap_final"] = df.gross_cap_svi - df.haircut_pp / 100
    df["error_final_pp"] = df.error_svi_pp - df.haircut_pp
    cal = df[df["sample"].eq("calibration")]
    const = float(cal.error_svi_pp.mean())
    rows = []
    for samp, x in [("all", df), ("calibration", cal), ("test", df[df["sample"].eq("test")])]:
        for name, e in [("v1: linear smile, 365d", x.error_linear_pp),
                        ("v2: SVI, exact expiry", x.error_svi_pp),
                        ("v2 + constant haircut", x.error_svi_pp - const),
                        ("v2 + rate/vol execution cost (final)", x.error_final_pp)]:
            e = e.dropna()
            rows.append({"sample": samp, "model": name, "n": len(e),
                         "periods_from": x.est_start.min().date(), "periods_to": x.est_start.max().date(),
                         "mean_error_pp": e.mean(), "mae_pp": e.abs().mean(), "rmse_pp": np.sqrt((e ** 2).mean())})
    return pd.DataFrame(rows), cm


def model_history(con) -> pd.DataFrame:
    dates = monthly_starts(con, "SPX")
    mk = load_markets_svi(con, "SPX", dates)
    vix = con.sql("select date, vix from canon.vol_index").df()
    vix["date"] = pd.to_datetime(vix.date)
    vix = vix.set_index("date").vix.sort_index()
    rows = []
    for d in dates:
        if d not in mk:
            continue
        m = mk[d]
        row = {"date": d, "spx": m.S, "r": m.r, "q": m.q, "atm_vol": m.vol(m.F),
               "vix": float(vix.asof(d)) if len(vix) else np.nan}
        for label, b in MODEL_BUFFERS.items():
            row[f"gross_cap_{int(round(b * 100))}"] = solve_cap(m, b)
        rows.append(row)
    save_cache()
    return pd.DataFrame(rows)


def charts(hist: pd.DataFrame, cmp_: pd.DataFrame, master: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    blue, orange = "#1F4E79", "#D9822B"
    fig, ax = plt.subplots(figsize=(12, 5), dpi=150)
    clip = lambda s: s.replace(np.inf, 0.45).clip(upper=0.45) * 100  # noqa: E731
    ax.plot(hist.date, clip(hist.gross_cap_10), color=blue, lw=1.6, label="Model gross cap, 10% buffer (SVI)")
    ax.plot(hist.date, clip(hist.gross_cap_15), color=orange, lw=1.6, label="Model gross cap, 15% buffer (SVI)")
    if len(cmp_):
        std = cmp_[cmp_.buffer_start.eq(0)]
        for lo, hi, mk_, col, lab in [(0.085, 0.105, "o", blue, "9–10%"), (0.145, 0.155, "s", orange, "15%")]:
            s = std[std.buffer.between(lo, hi)]
            ax.scatter(s.model_date, s.gross_cap_actual * 100, marker=mk_, s=20, facecolor="white",
                       edgecolor=col, lw=1.2, zorder=3, label=f"Actual cap from N-PORT, {lab} buffers (n={len(s)})")
    cur = master[master.first_product_comparable & master.issuer.str.startswith("First Trust")].copy()
    cur["gross"] = cur.current_starting_cap_net + FEES["First Trust (FT Vest)"]
    for b, col in [(10.0, blue), (15.0, orange)]:
        s = cur[cur.buffer_pct.eq(b)]
        ax.scatter(pd.to_datetime(s.current_outcome_start), s.gross * 100, marker="D", s=14, color=col, zorder=3)
    ax.text(0.995, 0.02, "Diamonds: published caps of current FT Vest series (beyond OptionMetrics coverage)",
            transform=ax.transAxes, ha="right", fontsize=8, color="#6A7179")
    ax.set_ylabel("Gross cap, 1-year S&P 500 buffer (%)\n(clipped at 45%)")
    ax.set_title("Historical caps: SVI model vs actual caps from N-PORT holdings", loc="left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, loc="upper center", fontsize=8.5, ncol=2)
    fig.tight_layout()
    fig.savefig(RESULTS / "caps_history.png")

    # model vs actual, out-of-sample periods, v1 and v2
    t = cmp_[cmp_["sample"].eq("test")]
    fig, ax = plt.subplots(figsize=(6.2, 5.6), dpi=150)
    ax.scatter(t.gross_cap_actual * 100, t.gross_cap_linear.replace(np.inf, np.nan) * 100, s=16, color="#B8BEC6",
               label=f"v1 linear smile, 365d (MAE {t.error_linear_pp.abs().mean():.2f} pp)")
    ax.scatter(t.gross_cap_actual * 100, t.gross_cap_final.replace(np.inf, np.nan) * 100, s=16, color=blue,
               label=f"v2 SVI + execution cost (MAE {t.error_final_pp.abs().mean():.2f} pp)")
    vals = pd.concat([t.gross_cap_actual, t.gross_cap_final, t.gross_cap_linear]).replace(np.inf, np.nan) * 100
    lim = [max(0.0, float(np.nanpercentile(vals, 0.5)) - 2), float(np.nanpercentile(vals, 99.5)) + 2]
    ax.plot(lim, lim, color="#8A94A3", lw=0.8, ls="--")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Actual gross cap from N-PORT (%)")
    ax.set_ylabel("Model gross cap (%)")
    ax.set_title(f"Out-of-sample test: {len(t)} fund-periods from {t.est_start.min():%b %Y}", loc="left", fontsize=10.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(RESULTS / "caps_validation.png")


if __name__ == "__main__":
    RESULTS.mkdir(exist_ok=True)
    pd.set_option("display.width", 200)
    con = connect()
    master = pd.read_csv(RESULTS / "fund_master.csv")
    act = actual_caps(con, master)
    act.to_csv(RESULTS / "caps_actual_nport.csv", index=False)
    print(f"Actual caps rebuilt from N-PORT: {len(act)} fund-periods, {act.ticker.nunique()} funds")
    cmp_ = model_vs_actual(con, act)
    cmp_.to_csv(RESULTS / "caps_model_vs_actual.csv", index=False)
    summ, cm = validation_summary(cmp_)
    cmp_.to_csv(RESULTS / "caps_model_vs_actual.csv", index=False)
    pd.DataFrame([cm]).to_csv(COST_MODEL, index=False)
    summ.to_csv(RESULTS / "caps_validation_summary.csv", index=False)
    print("\nModel vs actual gross caps (percentage points):")
    print(summ.round(2).to_string(index=False))
    print(f"\nExecution cost (pp of cap) = {cm['c0']:.2f} + {cm['c_r']:.2f} * r + {cm['c_vol']:.2f} * atm_vol, "
          f"fitted on {cm['n']} periods {cm['from']} to {cm['to']}")
    hist = model_history(con)
    hist.to_csv(RESULTS / "caps_model_history.csv", index=False)
    caps = hist[["gross_cap_9", "gross_cap_10", "gross_cap_15"]]
    print(f"\nModel cap history: {len(hist)} months; uncapped months: {np.isinf(caps).sum().to_dict()}")
    print(caps.replace(np.inf, np.nan).describe().round(4).to_string())
    charts(hist, cmp_, master)
    print("\n-> results/caps_actual_nport.csv, caps_model_vs_actual.csv, caps_validation_summary.csv, "
          "caps_model_history.csv, caps_history.png, caps_validation.png")
