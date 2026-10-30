"""
One master list of funds: every ticker that appears in any fund dataset (issuer term tables,
prices, CRSP NAV, N-PORT), classified into the three fund types and flagged for which data
we hold. Output: results/fund_master.csv

Fund types (sponsor guidance, Oct 2026: all three, buffers and derivative income first):
  1. Buffer / defined outcome   (priority 1)
  2. Derivative income          (priority 1)
  3. Autocallable               (priority 2)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.canonical import ROOT, connect  # noqa: E402

RESULTS = ROOT / "results"

# Derivative-income and autocallable funds: issuer, strategy, underlying
OTHER_FUNDS = {
    "JEPI": ("JPMorgan", "Derivative income", "Equity-linked notes (covered calls)", "S&P 500"),
    "JEPQ": ("JPMorgan", "Derivative income", "Equity-linked notes (covered calls)", "Nasdaq-100"),
    "QYLD": ("Global X", "Derivative income", "Covered call (ATM, monthly)", "Nasdaq-100"),
    "XYLD": ("Global X", "Derivative income", "Covered call (ATM, monthly)", "S&P 500"),
    "RYLD": ("Global X", "Derivative income", "Covered call (ATM, monthly)", "Russell 2000"),
    "SPYI": ("NEOS", "Derivative income", "Call spread overlay", "S&P 500"),
    "QQQI": ("NEOS", "Derivative income", "Call spread overlay", "Nasdaq-100"),
    "IWMI": ("NEOS", "Derivative income", "Call spread overlay", "Russell 2000"),
    "GPIX": ("Goldman Sachs", "Derivative income", "Dynamic covered call", "S&P 500"),
    "GPIQ": ("Goldman Sachs", "Derivative income", "Dynamic covered call", "Nasdaq-100"),
    "DIVO": ("Amplify", "Derivative income", "Tactical covered call on dividend stocks", "Large-cap dividend stocks"),
    "ISPY": ("ProShares", "Derivative income", "Daily covered call", "S&P 500"),
    "QDTE": ("Roundhill", "Derivative income", "0DTE covered call", "Nasdaq-100"),
    "XDTE": ("Roundhill", "Derivative income", "0DTE covered call", "S&P 500"),
    "RDTE": ("Roundhill", "Derivative income", "0DTE covered call", "Russell 2000"),
    "SVOL": ("Simplify", "Derivative income", "Short VIX futures with tail hedge", "VIX futures"),
    "BALI": ("iShares", "Derivative income", "Covered call on active large-cap", "US large cap"),
    "QQQY": ("Defiance", "Derivative income", "0DTE put writing", "Nasdaq-100"),
    "IWMY": ("Defiance", "Derivative income", "0DTE put writing", "Russell 2000"),
    "TSLY": ("YieldMax", "Derivative income", "Synthetic covered call (single stock)", "TSLA"),
    "NVDY": ("YieldMax", "Derivative income", "Synthetic covered call (single stock)", "NVDA"),
    "CONY": ("YieldMax", "Derivative income", "Synthetic covered call (single stock)", "COIN"),
    "MSTY": ("YieldMax", "Derivative income", "Synthetic covered call (single stock)", "MSTR"),
    "APLY": ("YieldMax", "Derivative income", "Synthetic covered call (single stock)", "AAPL"),
    "AMZY": ("YieldMax", "Derivative income", "Synthetic covered call (single stock)", "AMZN"),
    "GOOY": ("YieldMax", "Derivative income", "Synthetic covered call (single stock)", "GOOGL"),
    "FBY": ("YieldMax", "Derivative income", "Synthetic covered call (single stock)", "META"),
    "YMAX": ("YieldMax", "Derivative income", "Fund of single-stock option income ETFs", "Basket of YieldMax ETFs"),
    "ULTY": ("YieldMax", "Derivative income", "Option income on a stock basket", "US equity basket"),
    "CAIE": ("Calamos", "Autocallable", "Laddered autocallable notes (synthetic)", "Autocall index"),
}

REF_NAMES = {"SPY": "S&P 500", "SPX": "S&P 500", "QQQ": "Nasdaq-100", "IWM": "Russell 2000",
             "EFA": "MSCI EAFE", "EEM": "MSCI EM", "TLT": "20+Y Treasuries", "GLD": "Gold",
             "RSP": "S&P 500 Equal Weight", "CBTX": "Bitcoin"}


def innovator_strategy(name: str) -> tuple[str, float | None]:
    """Sub-strategy and buffer size (%) from an Innovator fund name."""
    n = name.replace("™", "").replace("®", "")
    if "Defined Protection" in n:
        return "Defined protection (100% buffer)", 100.0
    if "Wealth Shield" in n:
        return "Buffer (quarterly)", 20.0
    if "Barrier" in n:
        return "Premium income with barrier", None
    if "Premium Income" in n:
        m = re.search(r"(\d+) Buffer", n)
        return "Premium income with buffer", float(m.group(1)) if m else None
    if "Dual Directional" in n:
        m = re.search(r"(\d+) Buffer", n)
        return "Dual directional buffer", float(m.group(1)) if m else None
    if "5 to 15 Buffer" in n:
        return "Buffer (-5% to -15%)", 10.0
    if "Floor" in n:
        m = re.search(r"(\d+) Floor", n)
        return "Floor", -float(m.group(1)) if m else None
    if "Accelerated" in n and "Buffer" not in n:
        return "Accelerated upside (no buffer)", 0.0
    if "Accelerated" in n:
        m = re.search(r"(\d+) Buffer", n)
        return "Accelerated with buffer", float(m.group(1)) if m else None
    if "Ultra Buffer" in n:
        return "Ultra buffer (-5% to -35%)", 30.0
    if "Power Buffer" in n:
        return "Buffer", 15.0
    m = re.search(r"(\d+) Buffer", n)
    if m:
        return "Buffer", float(m.group(1))
    if "Buffer" in n:
        return "Buffer", 9.0
    return "Other defined outcome", None


def period_label(start, end) -> str:
    if pd.isna(start) or pd.isna(end):
        return ""
    months = round((pd.Timestamp(end) - pd.Timestamp(start)).days / 30.44)
    return {3: "Quarterly", 6: "6-month", 12: "Annual", 24: "2-year"}.get(months, f"{months}-month")


def build(con) -> pd.DataFrame:
    terms = con.sql("""select * from canon.fund_terms
                       where snapshot = (select max(snapshot) from canon.fund_terms)""").df()
    rows = []
    for _, r in terms.iterrows():
        if r["issuer"] == "Innovator":
            sub, buf = innovator_strategy(str(r["name"]))
        else:
            sub = str(r["name"])
            bs, be = r.get("buffer_start"), r.get("buffer_end")
            buf = round(abs(float(be) - float(bs)) * 100, 1) if pd.notna(bs) and pd.notna(be) else None
            if "Digital" in sub:
                sub = "Buffer with digital return"
        rows.append({
            "ticker": r["ticker"], "fund_type": "Buffer / defined outcome", "priority": 1,
            "issuer": "First Trust (FT Vest)" if r["issuer"].startswith("First Trust") else r["issuer"],
            "sub_strategy": sub, "reference_asset": REF_NAMES.get(str(r["ref_asset"]), r["ref_asset"]),
            "reference_ticker": r["ref_asset"], "outcome_period": period_label(r["outcome_start"], r["outcome_end"]),
            "series_month": r["series"], "buffer_pct": buf,
            "current_outcome_start": r["outcome_start"], "current_outcome_end": r["outcome_end"],
            "current_starting_cap_net": r["starting_cap_net"],
        })
    for t, (issuer, ftype, sub, ref) in OTHER_FUNDS.items():
        rows.append({"ticker": t, "fund_type": ftype, "priority": 1 if ftype == "Derivative income" else 2,
                     "issuer": issuer, "sub_strategy": sub, "reference_asset": ref})
    m = pd.DataFrame(rows).drop_duplicates("ticker")

    # data availability, from every fund dataset
    px = con.sql("select ticker, min(date) price_start, max(date) price_end, count(*) price_days "
                 "from canon.fund_price group by 1").df()
    nav = con.sql("select ticker, min(date) nav_start, max(date) nav_end from canon.fund_nav group by 1").df()
    npt = con.sql("select ticker, count(distinct report_date) nport_reports, min(report_date) nport_start "
                  "from canon.fund_option_legs group by 1").df()
    m = m.merge(px, on="ticker", how="left").merge(nav, on="ticker", how="left").merge(npt, on="ticker", how="left")

    # tickers seen in a dataset but not classified (should be none; listed so nothing is silently dropped)
    seen = set(px.ticker) | set(nav.ticker) | set(npt.ticker)
    extra = sorted(seen - set(m.ticker))
    if extra:
        m = pd.concat([m, pd.DataFrame({"ticker": extra, "fund_type": "Unclassified", "priority": 3})])

    m["has_terms"] = m["fund_type"].eq("Buffer / defined outcome")
    m["has_price"] = m["price_start"].notna()
    m["has_nav"] = m["nav_start"].notna()
    m["has_nport"] = m["nport_reports"].fillna(0) > 0
    m["first_product_comparable"] = (
        m["fund_type"].eq("Buffer / defined outcome") & m["reference_ticker"].isin(["SPY", "SPX"])
        & m["outcome_period"].eq("Annual") & m["sub_strategy"].isin(["Buffer", "Moderate Buffer"])
        & m["buffer_pct"].between(9, 15.5))
    order = {"Buffer / defined outcome": 0, "Derivative income": 1, "Autocallable": 2, "Unclassified": 3}
    m = m.sort_values(["fund_type", "issuer", "reference_asset", "ticker"],
                      key=lambda s: s.map(order) if s.name == "fund_type" else s.astype(str))
    return m.reset_index(drop=True)


def summary(m: pd.DataFrame) -> pd.DataFrame:
    g = m.groupby("fund_type", sort=False).agg(
        funds=("ticker", "size"), with_terms=("has_terms", "sum"), with_price=("has_price", "sum"),
        with_nav=("has_nav", "sum"), with_nport=("has_nport", "sum"))
    g.loc["Total"] = g.sum()
    return g


if __name__ == "__main__":
    RESULTS.mkdir(exist_ok=True)
    m = build(connect())
    m.to_csv(RESULTS / "fund_master.csv", index=False)
    print(summary(m).to_string())
    print("\nBy issuer (buffers):")
    print(m[m.fund_type.eq("Buffer / defined outcome")].groupby(["issuer", "sub_strategy"]).size().to_string())
    print("\nFirst-product comparables (annual S&P 500 buffers, 9-15%):")
    print(m[m.first_product_comparable][["ticker", "issuer", "series_month", "buffer_pct",
                                          "current_starting_cap_net"]].to_string(index=False))
    print(f"\n-> {RESULTS / 'fund_master.csv'}")
