"""
Pricing library for buffer structures, off the listed-option volatility surface.

A standard buffer ETF holds four European FLEX options on the reference asset, all with
the outcome-period expiry (S0 = reference price at the start of the period):
    long  deep-ITM call      ~ the reference asset (price return, no dividends)
    long  put  at (1 - a) S0  buffer starts at -a (a = 0 for a standard buffer)
    short put  at (1 - a - b) S0   buffer of size b
    short call at (1 + cap) S0     upside is capped
The cap is set so the overlay costs nothing:  Put(1-a) - Put(1-a-b) = Call(1+cap).
FLEX legs are priced with Black-Scholes at the implied volatility read off the listed
surface at the same moneyness ("price off the listed surface"); the forward uses the
zero curve and dividend yield of the same feed, so prices match the market's own quotes.
Funds quote a cap net of fees: net cap = gross cap - expense ratio.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import erf, exp, log, sqrt

import numpy as np
import pandas as pd

UNDERLYING_FOR = {"SPY": "SPX", "SPX": "SPX", "QQQ": "NDX", "NDX": "NDX", "IWM": "RUT", "RUT": "RUT"}


def _N(x: float) -> float:
    return 0.5 * (1 + erf(x / sqrt(2)))


def bs(cp: str, S: float, K: float, T: float, r: float, q: float, vol: float) -> float:
    if vol <= 0 or T <= 0:
        return max(0.0, (S - K) if cp == "C" else (K - S))
    d1 = (log(S / K) + (r - q + 0.5 * vol * vol) * T) / (vol * sqrt(T))
    d2 = d1 - vol * sqrt(T)
    if cp == "C":
        return S * exp(-q * T) * _N(d1) - K * exp(-r * T) * _N(d2)
    return K * exp(-r * T) * _N(-d2) - S * exp(-q * T) * _N(-d1)


@dataclass
class Market:
    """Market state for one underlying on one date, for one maturity."""
    date: pd.Timestamp
    S: float
    r: float          # continuously compounded zero rate for T
    q: float          # continuous dividend yield
    T: float          # years
    log_m: np.ndarray  # log(K / F), sorted
    iv: np.ndarray

    @property
    def F(self) -> float:
        return self.S * exp((self.r - self.q) * self.T)

    def vol(self, K: float) -> float:
        x = log(K / self.F)
        return float(np.interp(x, self.log_m, self.iv))  # flat beyond the quoted wings

    def price(self, cp: str, k_rel: float) -> float:
        """Option price per 1 unit of S0, strike = k_rel * S."""
        K = k_rel * self.S
        return bs(cp, self.S, K, self.T, self.r, self.q, self.vol(K)) / self.S


def overlay_value(m: Market, buffer: float, cap: float, buffer_start: float = 0.0) -> float:
    """Put spread bought minus call sold, per unit notional (zero at the fair cap)."""
    put_spread = m.price("P", 1 - buffer_start) - m.price("P", 1 - buffer_start - buffer)
    return put_spread - m.price("C", 1 + cap)


def dividend_budget(m: Market) -> float:
    """Cash freed by holding the index through a deep-ITM call instead of the shares:
    the call costs S*exp(-qT) (the holder forgoes the dividends), so 1 - exp(-qT) per unit
    notional is available to fund the option overlay."""
    return 1 - exp(-m.q * m.T)


def solve_cap(m: Market, buffer: float, buffer_start: float = 0.0, budget: float | None = None,
              lo: float = 0.0, hi: float = 2.0) -> float:
    """Gross cap at which the overlay's net cost equals the budget (bisection).
    Default budget = the forgone dividends (dividend_budget), which is how listed buffer ETFs
    are built; pass budget=0 for a pure zero-cost collar on the price index."""
    if budget is None:
        budget = dividend_budget(m)
    f = lambda c: overlay_value(m, buffer, c, buffer_start) - budget  # noqa: E731  rises with c
    if f(lo) >= 0:
        return lo
    if f(hi) < 0:
        return float("inf")  # the dividend budget alone pays for the buffer: no cap needed
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def buffer_payoff(R: np.ndarray | float, cap: float, buffer: float, buffer_start: float = 0.0):
    """Gross outcome-period return of the buffer structure given the reference price return R."""
    R = np.asarray(R, dtype=float)
    a, b = buffer_start, buffer
    return np.select([R >= 0, R >= -a, R >= -(a + b)],
                     [np.minimum(R, cap), R, np.full_like(R, -a)],
                     default=R + b)


# --------------------------------------------------------------------------------------
# loading market states from the canonical layer
# --------------------------------------------------------------------------------------
def load_markets(con, underlying: str, dates: list | None = None, tenor_days: int = 365) -> dict:
    """Market objects keyed by date, built from canon.vol_surface / zero_curve / div_yield / spot.
    The smile combines OTM puts (|delta| 0.10-0.50) and OTM calls (delta 0.10-0.50)."""
    date_filter = ""
    if dates is not None:
        ds = ",".join(f"'{pd.Timestamp(d).date()}'" for d in dates)
        date_filter = f"and date in ({ds})"
    surf = con.sql(f"""select date, cp, delta, iv, strike from canon.vol_surface
                       where underlying = '{underlying}' and tenor_days = {tenor_days} {date_filter}
                         and abs(delta) between 0.10 and 0.50""").df()
    zero = con.sql(f"select date, tenor_days, rate from canon.zero_curve where 1=1 {date_filter}").df()
    divs = con.sql(f"select date, q from canon.div_yield where underlying = '{underlying}' {date_filter}").df()
    spot = con.sql(f"select date, close from canon.spot where underlying = '{underlying}' {date_filter}").df()
    for d in (surf, zero, divs, spot):
        d["date"] = pd.to_datetime(d["date"])
    T = tenor_days / 365.0
    zr = {d: np.interp(tenor_days, g.tenor_days.to_numpy(), g.rate.to_numpy())
          for d, g in zero.sort_values("tenor_days").groupby("date")}
    qd = dict(zip(divs.date, divs.q))
    sd = dict(zip(spot.date, spot.close))
    out = {}
    for d, g in surf.groupby("date"):
        if d not in zr or d not in sd:
            continue
        S, r, q = float(sd[d]), float(zr[d]), float(qd.get(d, 0.0))
        F = S * exp((r - q) * T)
        g = g.assign(x=np.log(g.strike / F)).sort_values("x")
        # OTM side only: puts below the forward, calls above
        g = g[((g.cp == "P") & (g.x <= 0.02)) | ((g.cp == "C") & (g.x >= -0.02))]
        g = g.groupby("x", as_index=False).iv.mean()
        if len(g) < 4:
            continue
        out[d] = Market(d, S, r, q, T, g.x.to_numpy(), g.iv.to_numpy())
    return out


def trading_dates(con, underlying: str = "SPX") -> pd.DatetimeIndex:
    d = con.sql(f"select distinct date from canon.vol_surface where underlying = '{underlying}' "
                f"and tenor_days = 365 order by 1").df()
    return pd.DatetimeIndex(pd.to_datetime(d.date))


def on_or_before(index: pd.DatetimeIndex, d) -> pd.Timestamp | None:
    i = index.searchsorted(pd.Timestamp(d), side="right") - 1
    return index[i] if i >= 0 else None
