# Canonical data schema

The engine reads only the `canon` schema built by `engine/canonical.py`. To run on another data source (Bloomberg exports, a vendor feed, an in-house database), add a feed to `FEEDS` in `canonical.py`. A feed is one SQL query per table below, returning these column names and units. The engine code does not change.

Rates and yields are decimals and continuously compounded unless noted. Dates are `DATE`. `underlying` uses index names (`SPX`, `NDX`, `RUT`) or tickers.

| Table | Columns | Used by | Current source (`wrds_public`) |
| --- | --- | --- | --- |
| `spot` | date, underlying, close | pricing, back-tests | OptionMetrics secprd; Yahoo after OptionMetrics ends |
| `total_return_index` | date, underlying, tr_level | back-tests | Yahoo ^SP500TR |
| `vol_surface` | date, underlying, tenor_days, cp, delta (±0–1), iv, strike | cap solving, FLEX pricing | OptionMetrics vsurfd |
| `option_quotes` | date, underlying, expiry, cp, strike, bid, ask, iv, delta, vega, open_interest | SVI smile fit | OptionMetrics opprcd |
| `forward_curve` | date, underlying, expiry, forward | SVI moneyness, forwards | OptionMetrics fwdprd |
| `zero_curve` | date, tenor_days, rate | discounting, forwards | OptionMetrics zerocd |
| `div_yield` | date, underlying, q | forwards, dividend budget | OptionMetrics idxdvd |
| `funding` | date, sofr, effr, fed_funds (simple, act/360) | financing benchmark | NY Fed, FRED |
| `box_financing` | date, expiry, tenor_years, n_strikes, r_box | financing spread | computed from OptionMetrics quotes (`pull_all.box_rates`) |
| `vol_index` | date, vix, vix3m | regime split | Cboe |
| `regime` | date, recession, hy_spread | regime split | FRED |
| `fund_price` | date, ticker, close, adj_close, dividends | fund analysis | Yahoo |
| `fund_nav` | date, ticker, nav, nav_return | NAV reconciliation | CRSP Mutual Fund |
| `fund_terms` | issuer, ticker, …, starting_cap_net, buffer_start, buffer_end, snapshot | master list, caps | issuer websites, daily snapshots |
| `fund_option_legs` | ticker, report_date, put_call, side, strike, expiry, ref_ticker, contracts, value_usd | actual caps, FLEX legs | SEC N-PORT |

Optional, used automatically when a feed provides it:

| Table | Columns | Used by | Typical source |
| --- | --- | --- | --- |
| `ois_curve` | date, tenor_days, rate (simple, act/360) | financing spread vs same-tenor SOFR | Bloomberg SOFR OIS (USOSFR*), CME Term SOFR |

The minimum a new feed needs to price buffers is `spot`, `vol_surface`, `zero_curve` and `div_yield`; with `option_quotes` and `forward_curve` as well, the engine fits an SVI smile instead of interpolating the delta grid. All other tables are optional: a view whose source is missing is skipped and reported in `canon._status`.

All paths are resolved relative to the data folder at run time, so the project can be copied to another machine and run unchanged.
