# The Structured Product Sandbox — data pipeline

Wedbush Funds × Columbia Industry Projects, Fall 2026

One command builds the full dataset requested in the kickoff brief, organised by the brief's five data items, and loads it into a DuckDB database.

## Quick start

Needs Python 3.10+ and an internet connection. Tested on macOS (Apple Silicon).

```bash
cd structured_product_sandbox
pip install -r requirements.txt

# A) Free sources only (no account needed): about 30–60 minutes
python pull_all.py

# B) Full dataset, including OptionMetrics and CRSP (needs a WRDS account): several hours
python pull_all.py --user <your_wrds_username>

# Then: the financing-proxy chart from the Oct-15 update (needs B)
python box_spread_analysis.py
```

The first WRDS connection asks for your password, a Duo approval, and whether to save credentials in `~/.pgpass`. Nothing is stored by these scripts.

A long run can be stopped and restarted at any time: files already downloaded are skipped. For an overnight run on a Mac, use `caffeinate -i python pull_all.py --user <username>` so the machine doesn't sleep. Allow about 10 GB of disk space for the full dataset.

## What you get

Everything is written to `data/`, one folder per kickoff item.

| Folder / DuckDB schema | Kickoff item | Contents | History (as of Oct 2026) |
| --- | --- | --- | --- |
| `1_options_iv` | Option prices and IV surfaces | OptionMetrics option quotes (SPX, NDX, RUT, SPY, QQQ, IWM), volatility surfaces, standardised options and forward prices for 39 underlyings; Cboe VIX term structure, VXN, RVX, SKEW | 1996 – Aug 2025 (WRDS); Cboe to date |
| `2_total_return` | Total return with dividends | CRSP daily total and price returns, CRSP index; OptionMetrics prices, dividend yields and distributions; Yahoo prices and dividends | 1926 – present |
| `3_risk_free` | Risk-free curve | Fed Gürkaynak-Sack-Wright zero curve, Treasury CMT and bill rates (FRED and Treasury), OptionMetrics zero curve | 1961 – present |
| `4_etf` | ETF terms, holdings, NAV, price | Daily term snapshots (Innovator, FT Vest), ETF prices and dividends, CRSP daily NAV, SEC N-PORT holdings | Prices 2013 –; N-PORT 2019 –; terms from first snapshot |
| `5_financing` | Swap financing proxy | SOFR, EFFR, OBFR (NY Fed), Fed Funds, T-bill; N-PORT swap terms; **SPX box-spread implied financing rate** | 1996 – present |
| `extra` | — | Regime context: NBER recessions, HY spread, 10Y–2Y | — |

`data/inventory.csv` is regenerated on every run and lists, for every dataset, its first and last date, frequency, row count and number of securities.

## Querying the data (DuckDB)

`data/sandbox.duckdb` is a single-file database. There is no server to install. It has one schema per kickoff item: `s1_options_iv`, `s2_total_return`, `s3_risk_free`, `s4_etf`, `s5_financing`, plus `extra` and `main.inventory`. Small tables are stored in the file. The large option tables are views over the Parquet files, so they are queried in place, without loading them into memory.

```python
import duckdb
con = duckdb.connect("data/sandbox.duckdb", read_only=True)
con.sql("show all tables").show()

# SPX 30-day at-the-money implied vol, full history
con.sql("""
    select date, impl_volatility
    from s1_options_iv.optionm_vol_surface
    where secid = 108105 and days = 30 and delta = 50 and cp_flag = 'C'
    order by date
""").df()

# Term history of one buffer ETF across daily snapshots
con.sql("select snapshot, remaining_cap_net, remaining_buffer_net "
        "from s4_etf.defined_outcome_terms where ticker = 'POCT' order by snapshot").df()
```

OptionMetrics security ids: SPX 108105, NDX 102480, RUT 102434. The full map is in `s1_options_iv.optionm_secid_map`.

## Analysis engine (runs on any feed)

The engine reads only the canonical tables defined in `engine/SCHEMA.md`, never raw vendor files. To switch data source, add a feed in `engine/canonical.py`; nothing else changes. All paths are relative, so the folder can be copied to another machine and run as is.

```bash
python engine/run_all.py          # all five steps; first run also fits the SVI cache
```

| Step | Script | Output (results/) |
| --- | --- | --- |
| 1 | `fund_master.py` | `fund_master.csv`: one master list (buffer / defined outcome, derivative income, autocallable) |
| 2 | `financing.py` | `financing_spread_sofr_monthly.csv`, `financing_spread_sofr.png`: box-spread financing spread over SOFR |
| 3 | `historical_caps.py` | `caps_actual_nport.csv`, `caps_model_vs_actual.csv`, `caps_validation_summary.csv`, `execution_cost_model.csv`, `caps_model_history.csv`, `caps_history.png`, `caps_validation.png` |
| 4 | `first_product.py` | `first_product_vintages.csv`, `first_product_summary.csv`, `first_product_costs.csv`, two charts: annual S&P 500 buffers at 10% / 15% |
| 5 | `robustness.py` | `robustness_cohorts.csv`, `robustness_bootstrap.csv`, `robustness_fund_check.csv`, `robustness_fund_check.png` |

Single step: `python engine/run_all.py historical_caps`.

## Version 2 (Oct 28): what changed in the engine

| Area | Version 1 | Version 2 | Where |
| --- | --- | --- | --- |
| Validation sample | 5 FT Vest funds, 21 periods | 73 SPY buffer funds (Innovator B / P / U, FT Vest F / G / D), 339 periods; Innovator series matched by SEC series name | `pull_all.py` (`series_by_name`, `buffer_nport_universe`), `historical_caps.py` |
| Calibration vs test | same sample | execution cost fitted on periods before May 2023, error reported on later periods only | `historical_caps.py` |
| Volatility smile | linear in the delta grid, flat beyond 0.10 delta, fixed 365 days | SVI per listed expiry from option quotes, total variance interpolated to the fund's exact expiry, OptionMetrics forwards | `surface.py` |
| Execution cost | constant | linear in zero rate and ATM vol, inputs clipped to the calibrated range | `historical_caps.py` (`fit_cost_model`) |
| Financing benchmark | overnight SOFR | same-tenor SOFR: `canon.ois_curve` when a feed provides it, else SOFR realised over the box's life | `financing.py` |
| Back-test | overlapping monthly vintages only | + 12 non-overlapping cohorts, 12-month block bootstrap, check against actual fund NAVs | `robustness.py` |

SVI parameters are cached in `data/derived/svi_SPX.parquet` (`python engine/surface.py` builds it in about 5-10 minutes; otherwise it is built on first use).

## Useful options

| Command | What it does |
| --- | --- |
| `python pull_all.py --only 3 5` | Run only some kickoff items (1–5) |
| `python pull_all.py --skip-wrds` | Free sources only |
| `python pull_all.py --skip-wrds --only 4` | Today's ETF term snapshot. Run daily to build term history |
| `python pull_all.py --start 2008` | OptionMetrics from 2008 instead of 1996 (faster) |
| `python pull_all.py --single-name-quotes` | Also raw option quotes for the 28 single names (very large) |
| `python pull_all.py --nport-all` | N-PORT holdings for every ETF in the universe (slow) |
| `python pull_all.py --refresh` | Re-download files that already exist |
| `python pull_all.py --inventory` | Only rebuild the coverage report |
| `python pull_all.py --build-db` | Only rebuild `sandbox.duckdb` from the files on disk |

Environment variables (optional): `WRDS_USERNAME` instead of `--user`; `FRED_API_KEY` (free from fred.stlouisfed.org), used when FRED blocks scripted downloads. Every FRED series also has a non-FRED backup source.

## Financing proxy method

For each date and SPX expiry, regress call-minus-put mid prices on strike: C(K) − P(K) = DF·(F − K). The slope gives the discount factor DF, so r_box = −ln(DF)/T and the dividend term cancels. Subtracting the maturity-matched Treasury yield gives the market-implied cost of financing an equity position, following van Binsbergen, Diamond and Grotteria (2022). `box_spread_analysis.py` aggregates 6–18-month expiries into a monthly series (1996–Aug 2025: average 39 bp, peak 130 bp in Oct 2008).

## Known gaps

- OptionMetrics on WRDS currently ends 29 Aug 2025. A licensed feed is needed for later dates.
- Issuer websites only show live ETF terms, so term history starts at the first snapshot. Earlier series must be rebuilt from prospectus filings.
- FLEX option quotes and dealer swap financing quotes are not public.
- Term tables cover Innovator and FT Vest. Add other issuers' tickers to `EXTRA_DEFINED_OUTCOME` in `pull_all.py` to pull their prices, NAV and N-PORT holdings.
- Licensing: OptionMetrics and CRSP data come from WRDS and must not be redistributed. Share this code, not the `data/` folder.
