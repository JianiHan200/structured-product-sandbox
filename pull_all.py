#!/usr/bin/env python3
"""
Structured Product Sandbox: pull the whole dataset, organised exactly like the
kickoff brief's Data section, then report what was actually obtained.

    data/1_options_iv/       Daily option prices + implied vol surfaces (SPX, NDX, RUT, large single names)
    data/2_total_return/     Index and single-stock total return series, with dividends
    data/3_risk_free/        Risk-free rate curve history
    data/4_etf/              Terms, holdings, daily NAV and price history for defined-outcome
                             and derivative-income ETFs
    data/5_financing/        Swap financing rates or a defensible proxy
    data/extra/              Regime context (recessions, credit spread) used by the simulator
    data/inventory.csv       Coverage of every dataset: start, end, frequency, rows, # securities
    data/sandbox.duckdb      Everything above, registered in DuckDB as it downloads
                             (schemas s1_options_iv ... s5_financing, extra)

Usage (run in Terminal on your own machine):

    pip install wrds pandas pyarrow requests yfinance lxml duckdb
    python pull_all.py --build-db                          # (re)build the DuckDB from files on disk
    python pull_all.py --user christinahan                 # everything, 1996 -> today
    python pull_all.py --user christinahan --only 3 5      # just sections 3 and 5
    python pull_all.py --skip-wrds                         # free sources only (no WRDS login)
    python pull_all.py --inventory                         # only rebuild the coverage tables

Large WRDS tables are saved one file per year and skipped if that year already
exists, so an interrupted run can simply be restarted. Use --refresh to re-pull.
"""
from __future__ import annotations

import argparse
import io
import re
import sys
import time
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
S1, S2, S3, S4, S5, SX = (DATA / d for d in
                          ["1_options_iv", "2_total_return", "3_risk_free", "4_etf", "5_financing", "extra"])
TODAY = date.today().isoformat()

# ------------------------------------------------------------------ universe
INDEX_UNDERLYINGS = ["SPX", "NDX", "RUT"]
INDEX_ETFS = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "GLD", "RSP"]
SINGLE_NAMES = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "AVGO", "TSLA", "JPM", "LLY",
                "V", "UNH", "XOM", "MA", "COST", "HD", "PG", "JNJ", "NFLX", "AMD",
                "CRM", "ORCL", "ADBE", "COIN", "MSTR", "PLTR", "BAC", "WMT"]
YAHOO_INDICES = ["^GSPC", "^SP500TR", "^NDX", "^RUT", "^RUTTR"]  # ^XNDX has no history on Yahoo
INCOME_ETFS = ["JEPI", "JEPQ", "QYLD", "XYLD", "RYLD", "SPYI", "QQQI", "IWMI", "GPIX", "GPIQ",
               "DIVO", "ISPY", "QDTE", "XDTE", "RDTE", "SVOL", "BALI", "QQQY", "IWMY",
               "TSLY", "NVDY", "CONY", "MSTY", "APLY", "AMZY", "GOOY", "FBY", "YMAX", "ULTY", "CAIE"]
# Defined-outcome ETFs from issuers not covered by the term-table scrapers (AllianzIM, iShares,
# PGIM, Calamos, TrueShares ...). Add tickers here to get their prices / NAV / N-PORT.
EXTRA_DEFINED_OUTCOME: list[str] = []
# N-PORT is several SEC requests per fund-quarter; income ETFs + one of each buffer family by default.
NPORT_DEFAULT = INCOME_ETFS + ["BJUL", "PJUL", "UJUL", "BALT", "ZALT", "FJUL", "DJUL", "GJUL", "JULM", "QJUN"]
NPORT_MAX_FILINGS = 40

FRED = {
    "vol": ["VIXCLS", "VXNCLS", "RVXCLS"],
    "treasury": ["DGS1MO", "DGS3MO", "DGS6MO", "DGS1", "DGS2", "DGS3", "DGS5", "DGS7",
                 "DGS10", "DGS20", "DGS30", "DTB3", "DTB6"],
    "funding": ["SOFR", "SOFR30DAYAVG", "SOFR90DAYAVG", "DFF", "DTB3", "OBFR"],
    "regime": ["USREC", "BAMLH0A0HYM2", "T10Y2Y", "SP500", "NASDAQ100"],
}
CBOE_SYMBOLS = ["VIX9D", "VIX", "VIX3M", "VIX6M", "VIX1Y", "SKEW", "VXN", "RVX"]

SEC_UA = "Columbia MAFN Structured Product Sandbox research jianihan670@gmail.com"
BROWSER_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")
HTTP = requests.Session()
HTTP.headers.update({"User-Agent": BROWSER_UA})

REFRESH = False


def log(m: str) -> None:
    print(m, flush=True)


def get(url: str, headers: dict | None = None, tries: int = 4) -> requests.Response:
    for k in range(tries):
        try:
            r = HTTP.get(url, headers=headers, timeout=90)
            if r.status_code == 200:
                return r
            if r.status_code in (403, 429, 503):
                time.sleep(2 * (k + 1))
                continue
            r.raise_for_status()
        except requests.RequestException:
            if k == tries - 1:
                raise
            time.sleep(2 * (k + 1))
    raise RuntimeError(f"HTTP failed after {tries} tries: {url}")


def write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)
    log(f"    -> {path.relative_to(ROOT)}  rows={len(df):,}")


def done(path: Path) -> bool:
    return path.exists() and not REFRESH


def sql_list(xs) -> str:
    return ",".join(f"'{x}'" for x in xs)


def terms_tickers() -> list[str]:
    files = sorted((S4 / "terms").glob("*/defined_outcome_terms_*.csv"))
    return pd.read_csv(files[-1])["ticker"].dropna().str.upper().tolist() if files else []


def etf_universe() -> list[str]:
    return sorted(set(INCOME_ETFS + EXTRA_DEFINED_OUTCOME + terms_tickers()))


# ================================================================== free sources
FRED_FAILS = 0  # consecutive failed series; after 2 we treat FRED as blocked and stop waiting on it


def fred_series(sid: str) -> pd.Series:
    """FRED download. Uses the official API when FRED_API_KEY is set (most reliable);
    otherwise the CSV endpoint, which FRED's CDN often blocks for scripts. Gives up fast
    once FRED looks blocked, because every FRED series also has a non-FRED source below."""
    import os

    global FRED_FAILS
    key = os.environ.get("FRED_API_KEY")
    if key:
        js = get(f"https://api.stlouisfed.org/fred/series/observations?series_id={sid}"
                 f"&api_key={key}&file_type=json").json()
        df = pd.DataFrame(js["observations"])[["date", "value"]]
        df["date"] = pd.to_datetime(df["date"])
        return pd.to_numeric(df.set_index("date")["value"], errors="coerce").rename(sid)
    if FRED_FAILS >= 2:
        raise RuntimeError("skipped (FRED blocked this session; backup source used instead)")
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
    last = None
    for k, ua in enumerate(["curl/8.4.0", "Mozilla/5.0", BROWSER_UA]):
        try:
            r = requests.get(url, headers={"User-Agent": ua, "Accept": "text/csv,*/*"}, timeout=25)
            if r.status_code == 200 and r.text.lstrip().lower().startswith(("observation_date", "date")):
                df = pd.read_csv(io.StringIO(r.text))
                df.columns = ["date", sid]
                df["date"] = pd.to_datetime(df["date"])
                time.sleep(1)
                FRED_FAILS = 0
                return pd.to_numeric(df.set_index("date")[sid], errors="coerce")
            last = f"HTTP {r.status_code}"
        except requests.RequestException as e:
            last = type(e).__name__
        time.sleep(3)
    FRED_FAILS += 1
    raise RuntimeError(f"unavailable ({last}); set FRED_API_KEY to use the FRED API")


# ---------------------------------------------------------------- non-FRED backups
def treasury_block(kind: str, path: Path, first_year: int) -> None:
    """U.S. Treasury daily rates, one CSV per year (par yield curve from 1990, bills from 2002)."""
    frames = []
    for y in range(first_year, date.today().year + 1):
        url = ("https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
               f"daily-treasury-rates.csv/{y}/all?type={kind}&field_tdr_date_value={y}&page&_format=csv")
        try:
            df = pd.read_csv(io.StringIO(get(url).text))
            if len(df):
                frames.append(df)
        except Exception as e:  # noqa: BLE001
            log(f"    Treasury {kind} {y}: {str(e)[:80]}")
    if frames:
        df = pd.concat(frames, ignore_index=True)
        df = df.rename(columns={df.columns[0]: "date"})
        df["date"] = pd.to_datetime(df["date"])
        write(df.sort_values("date"), path)


def nyfed_block(path: Path) -> None:
    """NY Fed reference rates: SOFR (2018-), EFFR (2000-), OBFR (2016-), percent per annum."""
    out = {}
    for name, ep, start in [("SOFR", "secured/sofr", 2018), ("EFFR", "unsecured/effr", 2000),
                            ("OBFR", "unsecured/obfr", 2016)]:
        parts = []
        for y in range(start, date.today().year + 1):
            url = (f"https://markets.newyorkfed.org/api/rates/{ep}/search.json"
                   f"?startDate={y}-01-01&endDate={y}-12-31")
            try:
                js = get(url).json()
                rows = js.get("refRates", [])
                parts += [(r["effectiveDate"], r.get("percentRate")) for r in rows]
            except Exception as e:  # noqa: BLE001
                log(f"    NY Fed {name} {y}: {str(e)[:80]}")
        if parts:
            s = pd.Series(dict(parts), name=name)
            s.index = pd.to_datetime(s.index)
            out[name] = pd.to_numeric(s, errors="coerce").sort_index()
    if out:
        write(pd.DataFrame(out).sort_index().reset_index().rename(columns={"index": "date"}), path)


def fred_block(ids: list[str], path: Path) -> None:
    cols = {}
    for sid in ids:
        try:
            cols[sid] = fred_series(sid)
        except Exception as e:  # noqa: BLE001
            log(f"    FRED {sid} FAILED: {e}")
    if cols:
        write(pd.DataFrame(cols).sort_index().reset_index().rename(columns={"index": "date"}), path)


def cboe_block(path: Path) -> None:
    frames = {}
    for sym in CBOE_SYMBOLS:
        try:
            df = pd.read_csv(io.StringIO(get(
                f"https://cdn.cboe.com/api/global/us_indices/daily_prices/{sym}_History.csv").text))
            df.columns = [c.strip().upper() for c in df.columns]
            df["DATE"] = pd.to_datetime(df["DATE"])
            frames[sym] = df.set_index("DATE")["CLOSE" if "CLOSE" in df.columns else df.columns[-1]]
        except Exception as e:  # noqa: BLE001
            log(f"    Cboe {sym} FAILED: {e}")
    if frames:
        write(pd.DataFrame(frames).sort_index().reset_index().rename(columns={"DATE": "date"}), path)


def gsw_block(path: Path) -> None:
    lines = get("https://www.federalreserve.gov/data/yield-curve-tables/feds200628.csv").text.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("Date"))
    df = pd.read_csv(io.StringIO("\n".join(lines[start:])))
    keep = ["Date"] + [c for c in df.columns if re.fullmatch(r"SVENY\d\d", c)]
    df = df[keep].rename(columns={"Date": "date"})
    df["date"] = pd.to_datetime(df["date"])
    write(df, path)


def yahoo_block(tickers: list[str], price_path: Path, div_path: Path) -> None:
    import yfinance as yf

    prices, divs, missing = [], [], []
    for i in range(0, len(tickers), 40):
        batch = tickers[i:i + 40]
        df = yf.download(batch, period="max", interval="1d", auto_adjust=False, actions=True,
                         group_by="ticker", threads=True, progress=False)
        for t in batch:
            try:
                sub = df[t] if isinstance(df.columns, pd.MultiIndex) else df
                sub = sub.dropna(how="all")
            except KeyError:
                sub = pd.DataFrame()
            if sub.empty or "Close" not in sub or sub["Close"].dropna().empty:
                missing.append(t)
                continue
            sub = sub.rename(columns=lambda c: c.lower().replace(" ", "_"))
            sub = sub.rename(columns={"stock_splits": "splits"})
            sub["ticker"] = t
            sub = sub.reset_index().rename(columns={"Date": "date"})
            prices.append(sub)
            if "dividends" in sub:
                divs.append(sub.loc[sub["dividends"] > 0, ["date", "ticker", "dividends"]])
        log(f"    yahoo {min(i + 40, len(tickers))}/{len(tickers)}")
    if prices:
        p = pd.concat(prices, ignore_index=True)
        cols = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume", "dividends", "splits"]
        write(p[[c for c in cols if c in p.columns]], price_path)
    if divs:
        write(pd.concat(divs, ignore_index=True), div_path)
    if missing:
        log(f"    yahoo: no data for {', '.join(missing)}")
        (price_path.parent / (price_path.name.split(".")[0] + "_missing.txt")).write_text("\n".join(missing))


# ---------------------------------------------------------------- ETF term tables
def norm(s: str) -> str:
    return re.sub(r"\d+$", "", re.sub(r"[^a-z0-9]", "", s.lower()))


def clean(s: str) -> str:
    return " ".join(s.split()).replace("$", "").replace(",", "")


INNOV_MAP = {"ticker": "ticker", "name": "name", "series": "series", "referenceasset": "ref",
             "fundprice": "px", "remainingcap": "rem_cap", "remaininginversecap": "rem_inv_cap",
             "remainingbuffer": "rem_buf", "downsidebeforebuffer": "dn_before_buf",
             "remainingoutcomeperiod": "rem_days", "startingcap": "start_cap",
             "outcomeperiodstartdate": "start", "outcomeperiodenddate": "end",
             "startingrefassetprice": "ref0", "startingetfshareprice": "etf0", "indexprice": "ref_now",
             "remainingdistribution": "rem_dist", "indextobarrier": "idx_to_barrier",
             "remainingtotalpotentialreturn": "rem_tot_ret"}


def scrape_innovator(outdir: Path) -> pd.DataFrame | None:
    import lxml.html

    html = get("https://www.innovatoretfs.com/define/etfs/").text
    doc = lxml.html.fromstring(html)
    rows, seen = [], set()
    for gi, t in enumerate(doc.xpath("//table")):
        trs = t.xpath(".//tr")
        if not trs:
            continue
        hdr = [norm(c.text_content()) for c in trs[0].xpath("./th|./td")]
        if not hdr or hdr[0] != "ticker" or "remainingoutcomeperiod" not in hdr:
            continue
        for tr in trs[1:]:
            cells = tr.xpath("./th|./td")
            rec = {"table": gi}
            for h, c in zip(hdr, cells):
                if h in INNOV_MAP:
                    rec[INNOV_MAP[h]] = clean(c.text_content())
            tk = rec.get("ticker", "").upper()
            if tk and tk not in seen:
                rec["ticker"] = tk
                seen.add(tk)
                rows.append(rec)
    if not rows:
        log("    Innovator: no product table found (site layout may have changed)")
        return None
    df = pd.DataFrame(rows)
    write(df, outdir / "innovator_terms.csv")
    return df


def scrape_ftvest(outdir: Path) -> pd.DataFrame | None:
    import lxml.html

    base = "https://www.ftportfolios.com"
    html = get(f"{base}/Retail/etf/targetoutcomefundlist.aspx").text
    doc = lxml.html.fromstring(html)
    out = []
    for t in doc.xpath("//table"):
        trs = t.xpath(".//tr")
        if len(trs) < 5:
            continue
        hdr = [norm(c.text_content()) for c in trs[0].xpath("./th|./td")]
        if not hdr or hdr[0] != "tickersymbol" or "remainingcapnet" not in hdr:
            continue
        for tr in trs[1:]:
            c = tr.xpath("./th|./td")
            if len(c) < 10:
                continue
            sp = lambda cell: [clean(s.text_content()) for s in cell.xpath(".//span")]  # noqa: E731
            tk = sp(c[0])
            if len(tk) < 13:
                continue
            fv = clean(c[4].text_content()).replace(" ", "").split("/")
            rv = clean(c[5].text_content()).replace(" ", "").split("/")
            cap, dn = sp(c[6]), sp(c[8])
            fill = lambda xs, n: (xs + [""] * n)[:n]  # noqa: E731
            out.append([tk[0], clean(c[1].text_content()), clean(c[2].text_content()),
                        clean(c[3].text_content()), *fill(fv, 2), *fill(rv, 2), *fill(cap, 2),
                        clean(c[7].text_content()), *fill(dn, 3),
                        re.sub(r"\D", "", c[9].text_content()), *tk[1:13]])
    if not out:
        log("    FT Vest: buffer table not found (site layout may have changed)")
        return None
    cols = ["ticker", "strategy", "series", "ref", "fund_px", "fund_ret", "ref_px", "ref_ret",
            "ref_to_cap", "rem_cap_net", "rem_buf_net", "ref_to_buf_end", "dn_col2_unlabeled",
            "fund_dn_before_buf_net", "rem_days", "start", "end", "start_cap_net", "start_buf_net",
            "etf0", "etf_cap_px", "ref0", "ref_cap_px", "buf_start_pct", "ref_buf_start_px",
            "buf_end_pct", "ref_buf_end_px"]
    df = pd.DataFrame(out, columns=cols).replace("------", "")
    write(df, outdir / "ftvest_terms.csv")
    return df


def pct(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace("%", "").str.strip()
                         .replace({"-": None, "nan": None, "": None}), errors="coerce") / 100


def combine_terms(inv: pd.DataFrame | None, ft: pd.DataFrame | None, outdir: Path, asof: str) -> None:
    parts = []
    if inv is not None:
        parts.append(pd.DataFrame({
            "issuer": "Innovator", "ticker": inv.ticker, "name": inv.get("name"), "series": inv.get("series"),
            "ref_asset": inv.get("ref"),
            "outcome_start": pd.to_datetime(inv.get("start"), format="%m/%d/%Y", errors="coerce"),
            "outcome_end": pd.to_datetime(inv.get("end"), format="%m/%d/%Y", errors="coerce"),
            "starting_cap_net": pct(inv.get("start_cap", pd.Series(index=inv.index, dtype=str))),
            "remaining_cap_net": pct(inv.get("rem_cap", pd.Series(index=inv.index, dtype=str))),
            "remaining_buffer_net": pct(inv.get("rem_buf", pd.Series(index=inv.index, dtype=str))),
            "buffer_start": np.nan, "buffer_end": np.nan,
            "fund_px": pd.to_numeric(inv.get("px"), errors="coerce"),
            "ref_px": pd.to_numeric(inv.get("ref_now"), errors="coerce"),
            "etf_px_at_start": pd.to_numeric(inv.get("etf0"), errors="coerce"),
            "ref_px_at_start": pd.to_numeric(inv.get("ref0"), errors="coerce"),
            "as_of": asof}))
    if ft is not None:
        parts.append(pd.DataFrame({
            "issuer": "First Trust (FT Vest)", "ticker": ft.ticker, "name": ft.strategy, "series": ft.series,
            "ref_asset": ft.ref,
            "outcome_start": pd.to_datetime(ft.start, format="%m/%d/%Y", errors="coerce"),
            "outcome_end": pd.to_datetime(ft.end, format="%m/%d/%Y", errors="coerce"),
            "starting_cap_net": pct(ft.start_cap_net), "remaining_cap_net": pct(ft.rem_cap_net),
            "remaining_buffer_net": pct(ft.rem_buf_net), "buffer_start": pct(ft.buf_start_pct),
            "buffer_end": pct(ft.buf_end_pct), "fund_px": pd.to_numeric(ft.fund_px, errors="coerce"),
            "ref_px": pd.to_numeric(ft.ref_px, errors="coerce"),
            "etf_px_at_start": pd.to_numeric(ft.etf0, errors="coerce"),
            "ref_px_at_start": pd.to_numeric(ft.ref0, errors="coerce"), "as_of": asof}))
    if parts:
        write(pd.concat(parts, ignore_index=True), outdir / f"defined_outcome_terms_{asof}.csv")


# ---------------------------------------------------------------- SEC N-PORT
def sec_get(url: str) -> requests.Response:
    time.sleep(0.15)
    return get(url, headers={"User-Agent": SEC_UA, "Accept-Encoding": "gzip, deflate"})


def strip_ns(tag: str) -> str:
    return tag.split("}", 1)[-1]


def flatten(el: ET.Element, prefix: str = "") -> dict:
    out = {f"{prefix}{strip_ns(el.tag)}@{k}": v for k, v in el.attrib.items()}
    kids = list(el)
    if not kids:
        if el.text and el.text.strip():
            out[f"{prefix}{strip_ns(el.tag)}"] = el.text.strip()
        return out
    for ch in kids:
        out.update(flatten(ch, f"{prefix}{strip_ns(el.tag)}."))
    return out


def parse_nport(xml: bytes) -> pd.DataFrame:
    root = ET.fromstring(xml)
    info = {}
    for el in root.iter():
        n = strip_ns(el.tag)
        if n in ("seriesId", "seriesName", "repPdDate", "repPdEnd", "totAssets", "netAssets") and el.text:
            info.setdefault(n, el.text.strip())
    rows = [{k.replace("invstOrSec.", "", 1): v for k, v in flatten(i).items()}
            for i in root.iter() if strip_ns(i.tag) == "invstOrSec"]
    df = pd.DataFrame(rows)
    for k, v in info.items():
        df[k] = v
    return df


MONTH_CODES = "JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC"
# Trusts whose ETFs are missing from SEC's ticker file: series are matched by name instead
NAME_MATCHED_TRUSTS = {"Innovator": "0001415726"}   # Innovator ETFs Trust


def buffer_nport_universe() -> list[str]:
    """SPY-referenced 1-year four-leg buffers: Innovator B / P / U, FT Vest F / G / D families."""
    pat = re.compile(rf"^[BPUFGD]({MONTH_CODES})$")
    return sorted(t for t in terms_tickers() if pat.match(t))


def series_by_name() -> dict[str, str]:
    """ticker -> SEC series id for name-matched trusts (series name = issuer + fund name + month)."""
    files = sorted((S4 / "terms").glob("*/defined_outcome_terms_*.csv"))
    if not files:
        return {}
    terms = pd.read_csv(files[-1])
    out = {}
    for issuer, cik in NAME_MATCHED_TRUSTS.items():
        html = sec_get(f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}"
                       f"&scd=series&count=100").text
        name2sid = {norm_name(n): sid for sid, n in re.findall(
            r'CIK=(S\d{9})&amp;scd=series&amp;view=mutual-fund">([^<]+)</a>', html)}
        for r in terms[terms.issuer.eq(issuer)].itertuples():
            key = norm_name(f"{issuer} {r.name} - {r.series}")
            if key in name2sid:
                out[str(r.ticker).upper()] = name2sid[key]
    return out


def norm_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower().replace("(r)", "").replace("™", "")).strip()


def nport_block(tickers: list[str], outdir: Path) -> None:
    js = sec_get("https://www.sec.gov/files/company_tickers_mf.json").json()
    f = js["fields"]
    mapping = {r[f.index("symbol")].upper(): dict(zip(f, r)) for r in js["data"]}
    by_name = {}
    if any(t not in mapping for t in tickers):
        try:
            by_name = series_by_name()
        except Exception as e:  # noqa: BLE001
            log(f"    N-PORT: series-name lookup failed ({e})")
    ns = {"a": "http://www.w3.org/2005/Atom"}
    for t in tickers:
        path = outdir / f"nport_{t}.csv.gz"
        if done(path):
            continue
        sid = mapping[t]["seriesId"] if t in mapping else by_name.get(t)
        if not sid:
            log(f"    N-PORT {t}: not in SEC fund list")
            continue
        try:
            feed = ET.fromstring(sec_get(
                f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={sid}&type=NPORT-P"
                f"&dateb=&owner=include&count=100&output=atom").content)
            links = [e.find("a:link", ns).get("href") for e in feed.findall("a:entry", ns)][:NPORT_MAX_FILINGS]
        except Exception as e:  # noqa: BLE001
            log(f"    N-PORT {t}: filing list failed ({e})")
            continue
        frames = []
        for idx in links:
            b = idx.rsplit("/", 1)[0]
            try:
                df = parse_nport(sec_get(f"{b}/primary_doc.xml").content)
                if "seriesId" in df and len(df) and df["seriesId"].iloc[0] != sid:
                    continue
                df["ticker"], df["filing"] = t, b
                frames.append(df)
            except Exception as e:  # noqa: BLE001
                log(f"      {t} {b}: {e}")
        if frames:
            write(pd.concat(frames, ignore_index=True), path)


# ================================================================== WRDS
class Wrds:
    def __init__(self, user: str, start: int, end: int, single_name_quotes: bool):
        import wrds

        self.db = wrds.Connection(wrds_username=user)
        self.start, self.end, self.sn_quotes = start, end, single_name_quotes
        self._sec: pd.DataFrame | None = None

    def q(self, sql: str) -> pd.DataFrame:
        return self.db.raw_sql(sql)

    def secids(self) -> pd.DataFrame:
        if self._sec is None:
            path = S1 / "optionm_secid_map.parquet"
            df = self.q(f"select * from optionm.secnmd where ticker in "
                        f"({sql_list(INDEX_UNDERLYINGS + INDEX_ETFS + SINGLE_NAMES)})")
            df.columns = [c.lower() for c in df.columns]
            write(df, path)
            self._sec = df.sort_values("effect_date").groupby("ticker").tail(1)
            miss = sorted(set(INDEX_UNDERLYINGS + INDEX_ETFS + SINGLE_NAMES) - set(self._sec.ticker))
            if miss:
                log(f"    secid not found: {miss}")
        return self._sec

    def ids(self, tickers=None) -> str:
        s = self.secids()
        if tickers is not None:
            s = s[s.ticker.isin(tickers)]
        return ",".join(str(int(x)) for x in s.secid.unique())

    def yearly(self, folder: Path, table: str, sql_fn) -> None:
        for y in range(self.start, self.end + 1):
            path = folder / f"{y}.parquet"
            if done(path):
                continue
            try:
                write(self.q(sql_fn(f"optionm.{table}{y}", y)), path)
            except Exception as e:  # noqa: BLE001
                log(f"    {table}{y}: skipped ({str(e).splitlines()[0][:90]})")

    def whole(self, path: Path, sql: str) -> None:
        if done(path):
            return
        try:
            write(self.q(sql), path)
        except Exception as e:  # noqa: BLE001
            log(f"    {path.name}: FAILED ({str(e).splitlines()[0][:120]})")

    # --- section 1
    def section1(self) -> None:
        all_ids = self.ids()
        self.yearly(S1 / "optionm_vol_surface", "vsurfd", lambda t, y: f"""
            select secid, date, days, delta, cp_flag, impl_volatility, impl_strike, impl_premium
            from {t} where secid in ({all_ids})""")
        self.yearly(S1 / "optionm_std_options", "stdopd", lambda t, y: f"""
            select * from {t} where secid in ({all_ids})""")
        self.yearly(S1 / "optionm_forward_prices", "fwdprd", lambda t, y: f"""
            select * from {t} where secid in ({all_ids})""")
        quote_ids = self.ids(INDEX_UNDERLYINGS + ["SPY", "QQQ", "IWM"]
                             + (SINGLE_NAMES if self.sn_quotes else []))
        # raw quotes: one file per MONTH (a full year of SPX can exceed laptop memory);
        # tenors under 7 days (0DTE/weeklies) are dropped, structured products never use them
        folder = S1 / "optionm_option_quotes"
        for y in range(self.start, self.end + 1):
            for m in range(1, 13):
                path = folder / f"{y}-{m:02d}.parquet"
                if done(path) or date(y, m, 1) > date.today():
                    continue
                nxt = f"{y + (m == 12)}-{m % 12 + 1:02d}-01"
                where = (f"from optionm.opprcd{y} where secid in ({quote_ids}) "
                         f"and date >= '{y}-{m:02d}-01' and date < '{nxt}' "
                         f"and exdate - date between 7 and 1100 and best_bid > 0")
                core = ("secid, date, exdate, cp_flag, strike_price/1000.0 as strike, best_bid, best_offer, "
                        "volume, open_interest, impl_volatility, delta, gamma, vega, theta")
                df, err = None, None
                for cols in [core + ", am_settlement, ss_flag, exercise_style", core + ", am_settlement", core]:
                    try:
                        df = self.q(f"select {cols} {where}")
                        break
                    except Exception as e:  # noqa: BLE001  column missing in this vintage -> fewer columns
                        err = str(e).splitlines()[0][:110]
                        if "does not exist" in err and f"opprcd{y}" in err:
                            break  # the year table itself is missing
                if df is None:
                    log(f"    opprcd{y}-{m:02d}: skipped ({err})")
                    if err and f"opprcd{y}" in err:
                        break
                    continue
                if len(df):
                    write(df, path)

    # --- section 2
    def section2(self) -> None:
        s = f"{self.start}-01-01"
        self.whole(S2 / "optionm_underlying_prices.parquet",
                   f"select * from optionm.secprd where secid in ({self.ids()}) and date >= '{s}'")
        self.whole(S2 / "optionm_index_div_yield.parquet",
                   f"select * from optionm.idxdvd where secid in ({self.ids(INDEX_UNDERLYINGS)}) and date >= '{s}'")
        self.whole(S2 / "optionm_distributions.parquet",
                   f"select * from optionm.distrd where secid in ({self.ids()})")
        tick = SINGLE_NAMES + INDEX_ETFS
        self._crsp(tick, S2 / "crsp_daily.parquet", S2 / "crsp_permno_map.parquet", 1926)
        self.whole(S2 / "crsp_index_daily.parquet",
                   "select date, vwretd, vwretx, ewretd, ewretx, sprtrn, spindx from crsp.dsi")

    def _crsp(self, tick, path, map_path, start_year) -> None:
        if done(path):
            return
        try:
            names = self.q(f"select distinct permno, ticker from crsp.stksecurityinfohist "
                           f"where ticker in ({sql_list(tick)})")
            perm = ",".join(str(int(p)) for p in names.permno.unique())
            df = self.q(f"""select permno, dlycaldt as date, dlyprc as prc, dlyret as ret, dlyretx as retx,
                                   dlycap as mktcap, dlyvol as vol
                            from crsp.dsf_v2 where permno in ({perm}) and dlycaldt >= '{start_year}-01-01'""")
        except Exception as e:  # noqa: BLE001
            log(f"    CRSP CIZ failed ({str(e).splitlines()[0][:80]}); using legacy crsp.dsf")
            names = self.q(f"select distinct permno, ticker from crsp.msenames where ticker in ({sql_list(tick)})")
            perm = ",".join(str(int(p)) for p in names.permno.unique())
            df = self.q(f"""select permno, date, prc, ret, retx, shrout, vol, cfacpr
                            from crsp.dsf where permno in ({perm}) and date >= '{start_year}-01-01'""")
        write(names, map_path)
        write(df, path)

    # --- section 3
    def section3(self) -> None:
        self.whole(S3 / "optionm_zero_curve.parquet",
                   f"select date, days, rate from optionm.zerocd where date >= '{self.start}-01-01'")

    # --- section 4
    def section4(self) -> None:
        tick = etf_universe()
        self._crsp(tick, S4 / "crsp_etf_daily.parquet", S4 / "crsp_etf_permno_map.parquet", 2000)
        fmap = S4 / "crsp_mf_fund_map.parquet"
        try:
            fund = self.q(f"select distinct crsp_fundno, ticker, fund_name, et_flag, first_offer_dt "
                          f"from crsp.fund_hdr where ticker in ({sql_list(tick)})")
        except Exception:  # noqa: BLE001
            fund = self.q(f"select distinct crsp_fundno, ticker, fund_name from crsp.fund_hdr "
                          f"where ticker in ({sql_list(tick)})")
        write(fund, fmap)
        if len(fund):
            ids = ",".join(str(int(x)) for x in fund.crsp_fundno.unique())
            self.whole(S4 / "crsp_mf_daily_nav.parquet",
                       f"select crsp_fundno, caldt as date, dnav, dret from crsp.daily_nav_ret "
                       f"where crsp_fundno in ({ids})")

    def close(self) -> None:
        self.db.close()


# ================================================================== derived: box-spread financing rate
def box_rates(quotes_dir: Path, zero_path: Path, out_path: Path, secid: int = 108105) -> None:
    """
    Market-implied financing rate from SPX put-call parity across strikes (box spread):
        C(K) - P(K) = DF * (F - K)   =>   regress (C - P) on K per (date, expiry):
        slope = -DF,  r_box = -ln(DF) / T,  implied forward F = intercept / DF.
    Dividends drop out, so r_box - Treasury zero rate is a clean equity-financing spread
    (see van Binsbergen, Diamond & Grotteria 2022). European SPX options only.
    """
    files = sorted(quotes_dir.glob("*.parquet"))
    if not files:
        log("    box rates: no option quote files yet (run section 1 first)")
        return
    zero = pd.read_parquet(zero_path) if zero_path.exists() else None
    if zero is not None:
        zero["date"] = pd.to_datetime(zero["date"])
        if zero["rate"].median() > 1:
            zero["rate"] = zero["rate"] / 100
    out = []
    for f in files:
        q = read_cols(f, ["secid", "date", "exdate", "cp_flag", "strike", "best_bid",
                          "best_offer", "delta", "am_settlement"])
        if "am_settlement" not in q:
            q["am_settlement"] = -1
        q = q[(q.secid == secid) & (q.best_bid > 0)]
        if q.empty:
            continue
        q["mid"] = (q.best_bid + q.best_offer) / 2
        q = q[(q.best_offer - q.best_bid) / q.mid < 0.25]
        piv = q.pivot_table(index=["date", "exdate", "am_settlement", "strike"], columns="cp_flag",
                            values=["mid", "delta"], aggfunc="first").dropna()
        piv.columns = [f"{a}_{b}" for a, b in piv.columns]
        piv = piv[(piv.delta_C > 0.15) & (piv.delta_C < 0.85)].reset_index()
        piv["y"] = piv.mid_C - piv.mid_P
        g = piv.groupby(["date", "exdate", "am_settlement"])
        st = g.agg(n=("strike", "size"), mk=("strike", "mean"), my=("y", "mean"))
        piv = piv.join(st, on=["date", "exdate", "am_settlement"])
        piv["dk"], piv["dy"] = piv.strike - piv.mk, piv.y - piv.my
        piv["kk"], piv["ky"] = piv.dk ** 2, piv.dk * piv.dy
        s = piv.groupby(["date", "exdate", "am_settlement"]).agg(kk=("kk", "sum"), ky=("ky", "sum")).join(st)
        s = s[s.n >= 5].reset_index()
        s["slope"] = s.ky / s.kk
        s["df"] = -s.slope
        s["T"] = (pd.to_datetime(s.exdate) - pd.to_datetime(s.date)).dt.days / 365.0
        s = s[(s.df > 0) & (s.df < 1.2) & (s["T"] > 20 / 365)]
        s["r_box"] = -np.log(s.df) / s["T"]
        s["fwd_implied"] = (s.my - s.slope * s.mk) / s.df
        out.append(s[["date", "exdate", "am_settlement", "n", "T", "r_box", "fwd_implied"]])
        log(f"    box rates {f.stem}: {len(s):,} date-expiry points")
    if not out:
        return
    b = pd.concat(out, ignore_index=True)
    b["date"] = pd.to_datetime(b["date"])
    if zero is not None:
        zr = {d: (g["days"].to_numpy(), g["rate"].to_numpy()) for d, g in zero.sort_values("days").groupby("date")}
        b["r_zero"] = [np.interp(t * 365, *zr[d]) if d in zr else np.nan for d, t in zip(b.date, b["T"])]
        b["spread_bp"] = (b.r_box - b.r_zero) * 1e4
    write(b, out_path)


def swap_legs(nport_dir: Path, out_path: Path) -> None:
    frames = []
    for f in nport_dir.glob("nport_*.csv.gz"):
        df = pd.read_csv(f, low_memory=False)
        cols = [c for c in df.columns if "swapDeriv" in c]
        if not cols:
            continue
        sw = df[df[cols].notna().any(axis=1)]
        keep = [c for c in ["ticker", "repPdEnd", "name", "lei", "valUSD", "pctVal"] if c in sw] + cols
        frames.append(sw[keep])
    if frames:
        write(pd.concat(frames, ignore_index=True), out_path)
    else:
        log("    no swap legs found in N-PORT files")


# ================================================================== inventory
DATE_COLS = ["date", "Date", "DATE", "caldt", "dlycaldt", "repPdDate", "as_of"]  # repPdDate = holdings as-of
ID_COLS = ["ticker", "secid", "permno", "crsp_fundno"]


def freq_label(dates: pd.Series) -> str:
    u = pd.Series(pd.to_datetime(dates.dropna().unique())).sort_values()
    if len(u) <= 1:
        return "snapshot"
    d = u.diff().dt.days.median()
    return ("daily" if d <= 4 else "weekly" if d <= 10 else "monthly" if d <= 40
            else "quarterly" if d <= 100 else "irregular")


def read_cols(path: Path, cols=None) -> pd.DataFrame:
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        names = pq.ParquetFile(path).schema.names
        return pd.read_parquet(path, columns=[c for c in names if cols is None or c in cols])
    return pd.read_csv(path, low_memory=False, usecols=(lambda c: c in cols) if cols else None)


def inventory() -> pd.DataFrame:
    rows = []
    sections = {S1: "1 Options & IV", S2: "2 Total return", S3: "3 Risk-free",
                S4: "4 ETFs", S5: "5 Financing", SX: "extra"}
    for sec, label in sections.items():
        if not sec.exists():
            continue
        is_data = lambda p: p.is_file() and (p.suffix in (".csv", ".parquet") or p.name.endswith(".csv.gz"))  # noqa: E731
        # a sub-folder of data files (yearly parquet, one N-PORT file per fund) counts as ONE dataset;
        # dated term snapshots are listed file by file
        units = [p for p in sorted(sec.iterdir()) if p.is_dir() and p.name != "terms"
                 and any(is_data(f) for f in p.iterdir())]
        units += [p for p in sorted(sec.rglob("*")) if is_data(p)
                  and not any(p.is_relative_to(u) for u in units)]
        for u in units:
            files = sorted(f for f in u.iterdir() if is_data(f)) if u.is_dir() else [u]
            try:
                parts = [read_cols(f, DATE_COLS + ID_COLS) for f in files]
                head = read_cols(files[0])
            except Exception as e:  # noqa: BLE001
                rows.append({"section": label, "dataset": str(u.relative_to(DATA)), "note": f"unreadable: {e}"})
                continue
            df = pd.concat(parts, ignore_index=True)
            dcol = next((c for c in DATE_COLS if c in df.columns), None)
            icol = next((c for c in ID_COLS if c in df.columns), None)
            # "wide" = one row per date, one column per series (FRED, Cboe, GSW)
            numeric_wide = (icol is None and dcol is not None and len(head.columns) > 2
                            and not df[dcol].duplicated().any())
            if numeric_wide:  # FRED / Cboe / GSW style: one row per series
                full = pd.concat([read_cols(f) for f in files], ignore_index=True)
                for c in full.columns:
                    if c == dcol:
                        continue
                    s = full[[dcol, c]].dropna()
                    if s.empty:
                        continue
                    rows.append({"section": label, "dataset": f"{u.relative_to(DATA)} :: {c}",
                                 "start": pd.to_datetime(s[dcol]).min().date(),
                                 "end": pd.to_datetime(s[dcol]).max().date(),
                                 "frequency": freq_label(s[dcol]), "rows": len(s), "securities": 1})
                continue
            d = pd.to_datetime(df[dcol], errors="coerce") if dcol else pd.Series(dtype="datetime64[ns]")
            rows.append({"section": label, "dataset": str(u.relative_to(DATA)),
                         "start": d.min().date() if len(d.dropna()) else None,
                         "end": d.max().date() if len(d.dropna()) else None,
                         "frequency": freq_label(d) if dcol else "n/a", "rows": len(df),
                         "securities": df[icol].nunique() if icol else None})
    inv = pd.DataFrame(rows)
    inv.to_csv(DATA / "inventory.csv", index=False)
    for label, g in inv.groupby("section", sort=False):
        log(f"\n=== {label} ===")
        log(g.drop(columns="section").to_string(index=False))
    log(f"\n-> data/inventory.csv")
    return inv


# ================================================================== DuckDB database
DB_PATH = DATA / "sandbox.duckdb"
SCHEMAS = {S1: "s1_options_iv", S2: "s2_total_return", S3: "s3_risk_free",
           S4: "s4_etf", S5: "s5_financing", SX: "extra"}
MATERIALIZE_MAX_MB = 400  # single files up to this size are copied in; bigger ones become views


def _tname(s: str) -> str:
    s = re.sub(r"(\.csv\.gz|\.csv|\.parquet)$", "", s)
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def _reader(path_or_glob: str, is_parquet: bool) -> str:
    p = path_or_glob.replace("'", "''")
    return (f"read_parquet('{p}', union_by_name=true)" if is_parquet
            else f"read_csv_auto('{p}', union_by_name=true, sample_size=-1)")


def sync_db(sections: list[Path] | None = None) -> None:
    """Register every downloaded dataset in data/sandbox.duckdb.

    schema per kickoff item (s1_options_iv ... s5_financing, extra);
    small/medium files -> real tables; partitioned folders and very large files -> views
    over the parquet/CSV files (no duplicate copy of 10+ GB of option data)."""
    try:
        import duckdb
    except ImportError:
        log("  (duckdb not installed: pip install duckdb, then run  python pull_all.py --build-db)")
        return
    try:
        con = duckdb.connect(str(DB_PATH))
    except Exception as e:  # noqa: BLE001  e.g. the database is open in another program
        log(f"  DuckDB busy ({str(e).splitlines()[0][:80]}); close other connections and run --build-db")
        return
    is_data = lambda p: p.is_file() and (p.suffix in (".csv", ".parquet") or p.name.endswith(".csv.gz"))  # noqa: E731
    n_tab = n_view = 0
    for sec, schema in SCHEMAS.items():
        if (sections and sec not in sections) or not sec.exists():
            continue
        con.execute(f"create schema if not exists {schema}")
        for item in sorted(sec.iterdir()):
            try:
                if item.is_dir() and item.name == "terms":
                    # all dated snapshots stacked into one history (as_of column = snapshot date)
                    for kind in ["defined_outcome_terms", "innovator_terms", "ftvest_terms"]:
                        files = list(item.glob(f"*/{kind}*.csv"))
                        if files:
                            g = str(item / "*" / f"{kind}*.csv")
                            con.execute(f"create or replace view {schema}.{kind} as "
                                        f"select *, regexp_extract(filename, '(\\d{{4}}-\\d{{2}}-\\d{{2}})', 1) "
                                        f"as snapshot from read_csv_auto('{g}', union_by_name=true, "
                                        f"filename=true, all_varchar=true)")
                            n_view += 1
                elif item.is_dir():
                    files = [f for f in item.iterdir() if is_data(f)]
                    if not files:
                        continue
                    pq = any(f.suffix == ".parquet" for f in files)
                    glob = str(item / ("*.parquet" if pq else "*.csv.gz"))
                    con.execute(f"create or replace view {schema}.{_tname(item.name)} as "
                                f"select * from {_reader(glob, pq)}")
                    n_view += 1
                elif is_data(item):
                    pq = item.suffix == ".parquet"
                    src = _reader(str(item), pq)
                    tname = _tname(item.name)
                    name = f"{schema}.{tname}"
                    want_table = item.stat().st_size / 1e6 <= MATERIALIZE_MAX_MB
                    existing = con.execute(
                        "select table_type from information_schema.tables where table_schema=? and table_name=?",
                        [schema, tname]).fetchone()
                    if existing and (existing[0] == "VIEW") == want_table:  # type changes -> drop first
                        con.execute(f"drop {'view' if existing[0] == 'VIEW' else 'table'} {name}")
                    if want_table:
                        con.execute(f"create or replace table {name} as select * from {src}")
                        n_tab += 1
                    else:
                        con.execute(f"create or replace view {name} as select * from {src}")
                        n_view += 1
            except Exception as e:  # noqa: BLE001
                log(f"  DuckDB: could not register {item.relative_to(DATA)} ({str(e).splitlines()[0][:100]})")
    inv = DATA / "inventory.csv"
    if inv.exists() and not sections:
        con.execute(f"create or replace table main.inventory as select * from read_csv_auto('{inv}')")
    con.close()
    log(f"  DuckDB: {n_tab} tables + {n_view} views updated in {DB_PATH.relative_to(ROOT)}")


# ================================================================== orchestration
def run_section(n: int, w: Wrds | None, skip_public: bool, nport_all: bool) -> None:
    log(f"\n##### Section {n} #####")
    if n == 1:
        if not skip_public:
            fred_block(FRED["vol"], S1 / "vol_indices_fred.csv")
            cboe_block(S1 / "cboe_vix_term_structure.csv")
        if w:
            w.section1()
    elif n == 2:
        if not skip_public:
            yahoo_block(YAHOO_INDICES + INDEX_ETFS + SINGLE_NAMES + ["BRK-B"],
                        S2 / "yahoo_underlyings_daily.csv.gz", S2 / "yahoo_underlyings_dividends.csv")
        if w:
            w.section2()
    elif n == 3:
        if not skip_public:
            fred_block(FRED["treasury"], S3 / "fred_treasury_curve.csv")
            for fn, args in [(gsw_block, (S3 / "fed_gsw_zero_curve.csv",)),
                             (treasury_block, ("daily_treasury_yield_curve", S3 / "treasury_par_yield_curve.csv", 1990)),
                             (treasury_block, ("daily_treasury_bill_rates", S3 / "treasury_bill_rates.csv", 2002))]:
                try:
                    fn(*args)
                except Exception as e:  # noqa: BLE001
                    log(f"    {fn.__name__} FAILED: {str(e)[:100]}")
        if w:
            w.section3()
    elif n == 4:
        if not skip_public:
            snap = S4 / "terms" / TODAY
            inv = ft = None
            try:
                inv = scrape_innovator(snap)
            except Exception as e:  # noqa: BLE001
                log(f"    Innovator scrape failed: {e}")
            try:
                ft = scrape_ftvest(snap)
            except Exception as e:  # noqa: BLE001
                log(f"    FT Vest scrape failed: {e}")
            combine_terms(inv, ft, snap, TODAY)
            yahoo_block(etf_universe(), S4 / "etf_prices_daily.csv.gz", S4 / "etf_dividends.csv")
            nport_block(etf_universe() if nport_all else sorted(set(NPORT_DEFAULT + buffer_nport_universe())),
                        S4 / "nport")
        if w:
            w.section4()
    elif n == 5:
        if not skip_public:
            fred_block(FRED["funding"], S5 / "fred_funding_rates.csv")
            try:
                nyfed_block(S5 / "nyfed_reference_rates.csv")
            except Exception as e:  # noqa: BLE001
                log(f"    NY Fed FAILED: {str(e)[:100]}")
        swap_legs(S4 / "nport", S5 / "nport_swap_legs.csv")
        box_rates(S1 / "optionm_option_quotes", S3 / "optionm_zero_curve.parquet",
                  S5 / "spx_box_implied_financing.parquet")
        if not skip_public:
            fred_block(FRED["regime"], SX / "regime_fred.csv")


def main(argv=None) -> int:
    global REFRESH
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    import os
    ap.add_argument("--user", default=os.environ.get("WRDS_USERNAME"),
                    help="WRDS username (or set WRDS_USERNAME). Without one, only free sources are pulled")
    ap.add_argument("--only", nargs="*", type=int, choices=[1, 2, 3, 4, 5], help="sections to run")
    ap.add_argument("--start", type=int, default=1996, help="first year for OptionMetrics (data begins 1996)")
    ap.add_argument("--end", type=int, default=date.today().year)
    ap.add_argument("--skip-wrds", action="store_true")
    ap.add_argument("--skip-public", action="store_true")
    ap.add_argument("--single-name-quotes", action="store_true",
                    help="also pull raw option quotes for single names (very large; surfaces are always pulled)")
    ap.add_argument("--nport-all", action="store_true", help="N-PORT for every ETF in the universe (slow)")
    ap.add_argument("--refresh", action="store_true", help="re-download files that already exist")
    ap.add_argument("--inventory", action="store_true", help="only rebuild data/inventory.csv (+ DuckDB)")
    ap.add_argument("--build-db", action="store_true", help="only (re)build data/sandbox.duckdb from the files")
    ap.add_argument("--no-db", action="store_true", help="do not touch the DuckDB database")
    a = ap.parse_args(argv)
    REFRESH = a.refresh

    if a.build_db:
        sync_db()
        return 0
    if not a.inventory:
        w = None
        if not a.skip_wrds and not a.user:
            log("No WRDS username given (--user or WRDS_USERNAME): pulling free sources only.\n"
                "OptionMetrics/CRSP tables and the box-spread proxy need a WRDS account.")
            a.skip_wrds = True
        if not a.skip_wrds:
            w = Wrds(a.user, a.start, a.end, a.single_name_quotes)
        for n in a.only or [1, 2, 3, 4, 5]:
            try:
                run_section(n, w, a.skip_public, a.nport_all)
            except Exception as e:  # noqa: BLE001
                log(f"  Section {n} stopped: {e}")
            if not a.no_db:  # make this section queryable right away
                sync_db([[S1, S2, S3, S4, S5][n - 1]] + ([SX] if n == 5 else []))
        if w:
            w.close()
    inventory()
    if not a.no_db:
        sync_db()  # full rebuild of the catalogue + inventory table
    return 0


if __name__ == "__main__":
    sys.exit(main())
