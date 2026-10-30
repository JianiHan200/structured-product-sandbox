"""
Canonical data layer: the engine reads ONLY the `canon` schema defined here.

Every canonical table has a fixed name and column set (see SCHEMA.md). A *feed* is a
set of adapters, one SQL query per canonical table, that maps a particular data source
onto those columns. To run the engine on a different source (Bloomberg exports, a vendor
feed, Wedbush's own OptionMetrics), add a feed to FEEDS with the same table names and
columns; no engine code changes.

All paths are relative to the data folder, so the project can be moved or copied to
another machine without rebuilding anything.

    from engine.canonical import connect
    con = connect()                      # default feed "wrds_public", data/ next to this repo
    con.sql("select * from canon.spot where underlying = 'SPX' limit 5")
"""
from __future__ import annotations

from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data"

# --------------------------------------------------------------------------------------
# Feed: OptionMetrics + CRSP via WRDS, plus public sources (what pull_all.py downloads)
# Placeholders: {d} = data folder (absolute, resolved at runtime)
# --------------------------------------------------------------------------------------
# OptionMetrics secid -> underlying name (latest name record per secid)
SECMAP = """
    select cast(secid as integer) as secid,
           case cast(secid as integer) when 108105 then 'SPX' when 102480 then 'NDX'
                                       when 102434 then 'RUT' else ticker end as underlying
    from (select secid, ticker, row_number() over (partition by secid order by effect_date desc) rn
          from '{d}/1_options_iv/optionm_secid_map.parquet') where rn = 1"""

WRDS_PUBLIC = {
    # spot level of each underlying (price index, no dividends)
    "spot": """
        select cast(s.date as date) as date, m.underlying, s.close
        from '{d}/2_total_return/optionm_underlying_prices.parquet' s
        join canon._secmap m on cast(s.secid as integer) = m.secid
        where s.close is not null
        union all
        select cast(date as date), case ticker when '^GSPC' then 'SPX' when '^NDX' then 'NDX'
                                               when '^RUT' then 'RUT' else ticker end, close
        from read_csv_auto('{d}/2_total_return/yahoo_underlyings_daily.csv.gz')
        where ticker in ('^GSPC','^NDX','^RUT','SPY','QQQ','IWM')
          and cast(date as date) > (select max(cast(date as date))
                                    from '{d}/2_total_return/optionm_underlying_prices.parquet')""",
    # total-return index level (dividends reinvested) where available
    "total_return_index": """
        select cast(date as date) as date, 'SPX' as underlying, close as tr_level
        from read_csv_auto('{d}/2_total_return/yahoo_underlyings_daily.csv.gz') where ticker = '^SP500TR'""",
    # implied volatility surface on a (tenor, delta) grid; strike = implied strike
    "vol_surface": """
        select cast(s.date as date) as date, m.underlying, cast(s.days as integer) as tenor_days,
               s.cp_flag as cp, s.delta / 100.0 as delta, s.impl_volatility as iv, s.impl_strike as strike
        from read_parquet('{d}/1_options_iv/optionm_vol_surface/*.parquet', union_by_name=true) s
        join canon._secmap m on cast(s.secid as integer) = m.secid
        where s.impl_volatility is not null""",
    # individual option quotes (end of day): used to fit the SVI smile (engine/surface.py)
    "option_quotes": """
        select cast(q.date as date) as date, m.underlying, cast(q.exdate as date) as expiry,
               q.cp_flag as cp, q.strike, q.best_bid as bid, q.best_offer as ask,
               q.impl_volatility as iv, q.delta, q.vega, q.open_interest
        from read_parquet('{d}/1_options_iv/optionm_option_quotes/*.parquet', union_by_name=true) q
        join canon._secmap m on cast(q.secid as integer) = m.secid
        where q.impl_volatility is not null""",
    # forward price of the underlying for each listed expiry
    "forward_curve": """
        select cast(f.date as date) as date, m.underlying, cast(f.expiration as date) as expiry,
               f.forwardprice as forward
        from read_parquet('{d}/1_options_iv/optionm_forward_prices/*.parquet', union_by_name=true) f
        join canon._secmap m on cast(f.secid as integer) = m.secid
        where f.forwardprice is not null""",
    # zero-coupon curve, continuously compounded, decimal
    "zero_curve": """
        select cast(date as date) as date, cast(days as integer) as tenor_days, rate / 100.0 as rate
        from '{d}/3_risk_free/optionm_zero_curve.parquet'""",
    # continuous dividend yield, decimal
    "div_yield": """
        select cast(date as date) as date,
               case cast(secid as integer) when 108105 then 'SPX' when 102480 then 'NDX'
                                           when 102434 then 'RUT' end as underlying,
               rate / 100.0 as q
        from '{d}/2_total_return/optionm_index_div_yield.parquet'""",
    # overnight funding benchmarks, decimal (SOFR from Apr 2018, EFFR from Jul 2000, Fed funds from 1954)
    "funding": """
        select coalesce(n.date, f.date) as date, n.SOFR / 100.0 as sofr, n.EFFR / 100.0 as effr,
               f.DFF / 100.0 as fed_funds
        from (select cast(date as date) as date, SOFR, EFFR
              from read_csv_auto('{d}/5_financing/nyfed_reference_rates.csv')) n
        full outer join (select cast(date as date) as date, DFF
                         from read_csv_auto('{d}/5_financing/fred_funding_rates.csv')) f
          on n.date = f.date""",
    # market-implied financing rate from SPX box spreads (pull_all.box_rates)
    "box_financing": """
        select cast(date as date) as date, cast(exdate as date) as expiry, T as tenor_years,
               n as n_strikes, r_box
        from '{d}/5_financing/spx_box_implied_financing.parquet'""",
    # market regime indicators: VIX close, NBER recession flag (monthly), HY spread
    "vol_index": """
        select cast(date as date) as date, VIX as vix, VIX3M as vix3m
        from read_csv_auto('{d}/1_options_iv/cboe_vix_term_structure.csv')""",
    "regime": """
        select cast(date as date) as date, USREC as recession, BAMLH0A0HYM2 as hy_spread
        from read_csv_auto('{d}/extra/regime_fred.csv')""",
    # fund-level series
    "fund_price": """
        select cast(date as date) as date, ticker, close, adj_close, dividends
        from read_csv_auto('{d}/4_etf/etf_prices_daily.csv.gz')""",
    "fund_nav": """
        select cast(n.date as date) as date, m.ticker, n.dnav as nav, n.dret as nav_return
        from '{d}/4_etf/crsp_mf_daily_nav.parquet' n
        join (select distinct crsp_fundno, ticker from '{d}/4_etf/crsp_mf_fund_map.parquet') m
          using (crsp_fundno)""",
    # dated snapshots of issuer term tables
    "fund_terms": """
        select *, regexp_extract(filename, '(\\d{4}-\\d{2}-\\d{2})', 1) as snapshot
        from read_csv_auto('{d}/4_etf/terms/*/defined_outcome_terms_*.csv',
                           union_by_name=true, filename=true)""",
    # option legs reported in SEC N-PORT
    "fund_option_legs": """
        select ticker, cast(repPdDate as date) as report_date,
               "derivativeInfo.optionSwaptionWarrantDeriv.putOrCall" as put_call,
               "derivativeInfo.optionSwaptionWarrantDeriv.writtenOrPur" as side,
               cast("derivativeInfo.optionSwaptionWarrantDeriv.exercisePrice" as double) as strike,
               cast("derivativeInfo.optionSwaptionWarrantDeriv.expDt" as date) as expiry,
               "derivativeInfo.optionSwaptionWarrantDeriv.descRefInstrmnt.otherRefInst.identifiers.ticker@value"
                   as ref_ticker,
               cast(balance as double) as contracts, cast(valUSD as double) as value_usd
        from read_csv_auto('{d}/4_etf/nport/nport_*.csv.gz', union_by_name=true, all_varchar=true)
        where "derivativeInfo.optionSwaptionWarrantDeriv.putOrCall" is not null""",
}

# Optional tables a richer feed can add (not in wrds_public). The engine uses them when present:
#   ois_curve(date, tenor_days, rate)  SOFR OIS / Term SOFR par rates, decimal, act/360 simple
#                                      -> engine/financing.py measures box spreads against it.
OPTIONAL_TABLES = ["ois_curve"]

FEEDS = {"wrds_public": WRDS_PUBLIC}


def connect(data_dir: str | Path | None = None, feed: str = "wrds_public",
            database: str = ":memory:") -> duckdb.DuckDBPyConnection:
    """Open DuckDB and register the canonical views for `feed` over files in `data_dir`.
    Views whose source files are missing are skipped (reported in canon._status)."""
    d = Path(data_dir or DEFAULT_DATA).resolve()
    con = duckdb.connect(database)
    con.execute("set enable_progress_bar = false")
    con.execute("create schema if not exists canon")
    status = []
    try:
        con.execute("create or replace view canon._secmap as " + SECMAP.replace("{d}", str(d)))
    except Exception as e:  # noqa: BLE001
        status.append(("_secmap", "missing", str(e).splitlines()[0][:120]))
    for name, sql in FEEDS[feed].items():
        try:
            con.execute(f"create or replace view canon.{name} as " + sql.replace("{d}", str(d)))
            status.append((name, "ok", ""))
        except Exception as e:  # noqa: BLE001
            con.execute(f"drop view if exists canon.{name}")
            status.append((name, "missing", str(e).splitlines()[0][:120]))
    con.execute("create or replace table canon._status(view_name varchar, status varchar, detail varchar)")
    con.executemany("insert into canon._status values (?, ?, ?)", status)
    return con


if __name__ == "__main__":
    c = connect()
    print(c.sql("select * from canon._status").df().to_string(index=False))
