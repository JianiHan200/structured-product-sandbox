"""
SVI volatility smile, fitted to listed option quotes and interpolated to any maturity.

This replaces the first version's smile (linear interpolation of the OptionMetrics delta grid,
flat beyond the 0.10-delta wings, fixed 365-day tenor) with:

  1. One raw-SVI slice per listed expiry (Gatheral 2004), fitted to end-of-day out-of-the-money
     quotes in forward log-moneyness k = ln(K / F_T), total variance w = iv^2 * T:

         w(k) = a + b * ( rho * (k - m) + sqrt((k - m)^2 + sigma^2) )

     Fit: quasi-explicit method (Zeliade 2009). For fixed (m, sigma) the problem is linear in
     (a, d = b*rho*sigma, c = b*sigma); an outer Nelder-Mead searches (m, sigma). The box
     0 <= c <= 4*sigma, |d| <= min(c, 4*sigma - c) and a >= -sqrt(c^2 - d^2) keeps total
     variance positive and the wings within Lee's moment bound. Butterfly arbitrage is checked
     on a grid (Gatheral-Jacquier g(k) >= 0) and reported.
  2. Any maturity T (for example the exact outcome-period expiry of a fund) by linear
     interpolation of total variance between the two listed expiries around T, at equal k.
     Beyond the last listed expiry the nearest slice is used at constant implied vol.
  3. The forward for T is interpolated from OptionMetrics' forward prices (ln F/S linear in T);
     the dividend yield the pricer uses is the one implied by that forward and the zero rate.

load_markets_svi() returns the same engine.pricing.Market objects as load_markets(), with the
SVI smile sampled on a fine k-grid, so every pricing function works unchanged. Dates where the
quotes cannot support a fit fall back to the delta-grid surface; fit_report() lists which.
Pure numpy: no scipy needed.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from math import exp, log
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.canonical import ROOT  # noqa: E402
from engine.pricing import Market, load_markets

CACHE_DIR = ROOT / "data" / "derived"     # fitted SVI parameters (derived, small); rebuilt when missing

K_GRID = np.linspace(-2.0, 1.2, 641)   # forward log-moneyness grid the smile is sampled on
MIN_POINTS = 6                          # OTM quotes needed to fit a slice
MIN_ABS_DELTA = 0.03                    # drop the deepest wings, where quotes are stale or one tick
WING_PAD = 0.05                         # SVI used up to 0.05 in log-moneyness past the last quote


@dataclass
class SVISlice:
    T: float
    a: float
    b: float
    rho: float
    m: float
    sigma: float
    rmse_vol: float = np.nan   # root-mean-square error in implied vol points
    n: int = 0
    kmin: float = -np.inf      # quoted moneyness range; beyond it implied vol is held flat,
    kmax: float = np.inf       # so an unconstrained SVI wing never extrapolates past the data

    def w(self, k):
        k = np.clip(np.asarray(k, dtype=float), self.kmin - WING_PAD, self.kmax + WING_PAD)
        x = k - self.m
        return self.a + self.b * (self.rho * x + np.sqrt(x * x + self.sigma ** 2))

    def butterfly_free(self) -> bool:
        k = np.linspace(max(self.kmin, -1.5), min(self.kmax, 1.0), 201)
        w = self.w(k)
        if (w <= 0).any():
            return False
        x = k - self.m
        r = np.sqrt(x * x + self.sigma ** 2)
        w1 = self.b * (self.rho + x / r)
        w2 = self.b * self.sigma ** 2 / r ** 3
        g = (1 - k * w1 / (2 * w)) ** 2 - w1 ** 2 / 4 * (1 / w + 0.25) + w2 / 2
        return bool((g >= -1e-6).all())


# ------------------------------------------------------------------------------- fitting
def _nelder_mead(f, x0, step, iters=200, tol=1e-12):
    n = len(x0)
    pts = [np.asarray(x0, float)] + [np.asarray(x0, float) + np.eye(n)[i] * step[i] for i in range(n)]
    vals = [f(p) for p in pts]
    for _ in range(iters):
        order = np.argsort(vals)
        pts, vals = [pts[i] for i in order], [vals[i] for i in order]
        if abs(vals[-1] - vals[0]) < tol:
            break
        c = np.mean(pts[:-1], axis=0)
        xr = c + (c - pts[-1]); fr = f(xr)
        if fr < vals[0]:
            xe = c + 2 * (c - pts[-1]); fe = f(xe)
            pts[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < vals[-2]:
            pts[-1], vals[-1] = xr, fr
        else:
            xc = c + 0.5 * (pts[-1] - c); fc = f(xc)
            if fc < vals[-1]:
                pts[-1], vals[-1] = xc, fc
            else:
                pts = [pts[0] + 0.5 * (p - pts[0]) for p in pts]
                vals = [f(p) for p in pts]
    i = int(np.argmin(vals))
    return pts[i], vals[i]


def _inner(k, w, m, s):
    """Best (a, d, c) for fixed (m, sigma), projected onto the no-arbitrage box."""
    y = (k - m) / s
    z = np.sqrt(y * y + 1)
    X = np.column_stack([np.ones_like(y), y, z])
    a, d, c = np.linalg.lstsq(X, w, rcond=None)[0]
    c = min(max(c, 0.0), 4 * s)
    dmax = min(c, 4 * s - c)
    d = min(max(d, -dmax), dmax)
    a = float(np.mean(w - d * y - c * z))                 # re-fit the level after projection
    a = max(a, -np.sqrt(max(c * c - d * d, 0.0)) + 1e-8)  # keeps w(k) > 0 everywhere
    return a, d, c, a + d * y + c * z


def fit_svi(k, w, T) -> SVISlice:
    k, w = np.asarray(k, float), np.asarray(w, float)
    lo, hi = k.min(), k.max()

    def obj(p):
        m, ls = p
        s = exp(ls)
        if not (lo - 0.5 <= m <= hi + 0.5) or not (0.005 <= s <= 3.0):
            return 1e9
        return float(np.mean((_inner(k, w, m, s)[3] - w) ** 2))

    best = min(((obj([m, log(s)]), m, s) for m in (-0.2, -0.05, 0.05, 0.2) for s in (0.05, 0.15, 0.4)))
    (m, ls), _ = _nelder_mead(obj, [best[1], log(best[2])], step=[0.05, 0.3])
    s = exp(ls)
    a, d, c, fit = _inner(k, w, m, s)
    b = c / s
    rho = d / c if c > 0 else 0.0
    rmse = float(np.sqrt(np.mean((np.sqrt(np.maximum(fit, 1e-12) / T) - np.sqrt(w / T)) ** 2)))
    return SVISlice(T, a, b, rho, m, s, rmse, len(k), float(lo), float(hi))


# ------------------------------------------------------------------------------- data
def _quotes(con, underlying: str, d: pd.Timestamp) -> pd.DataFrame:
    q = con.sql(f"""
        select q.expiry, q.cp, q.strike, q.iv, q.delta, q.bid, f.forward
        from canon.option_quotes q
        join canon.forward_curve f on f.date = q.date and f.underlying = q.underlying and f.expiry = q.expiry
        where q.underlying = '{underlying}' and q.date = '{d.date()}'
          and q.bid > 0 and q.iv between 0.01 and 3""").df()
    if q.empty:
        return q
    q["T"] = (pd.to_datetime(q.expiry) - d).dt.days / 365.0
    q["k"] = np.log(q.strike / q.forward)
    otm = ((q.cp == "P") & (q.k < 0)) | ((q.cp == "C") & (q.k >= 0))
    return q[otm & (q.delta.abs() >= MIN_ABS_DELTA)]


_CACHE: dict[str, pd.DataFrame] = {}
_NEW: dict[str, list] = {}
FIELDS = ["T", "a", "b", "rho", "m", "sigma", "rmse_vol", "n", "kmin", "kmax"]


def _cache(underlying: str) -> pd.DataFrame:
    if underlying not in _CACHE:
        f = CACHE_DIR / f"svi_{underlying}.parquet"
        _CACHE[underlying] = pd.read_parquet(f) if f.exists() else pd.DataFrame(columns=["date", "expiry"] + FIELDS)
        _CACHE[underlying]["date"] = pd.to_datetime(_CACHE[underlying]["date"])
    return _CACHE[underlying]


def save_cache() -> None:
    for und, rows in _NEW.items():
        if not rows:
            continue
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df = pd.concat([_cache(und), pd.DataFrame(rows)], ignore_index=True)
        df["date"] = pd.to_datetime(df["date"])
        df = df.drop_duplicates(["date", "expiry"], keep="last").sort_values(["date", "expiry"])
        df.to_parquet(CACHE_DIR / f"svi_{und}.parquet", index=False)
        _CACHE[und], _NEW[und] = df, []


def fit_date(con, underlying: str, d, T_min=0.15, T_max=3.0) -> dict[pd.Timestamp, SVISlice]:
    """SVI slice for every listed expiry between T_min and T_max years on date d (cached)."""
    d = pd.Timestamp(d)
    c = _cache(underlying)
    hit = c[c.date == d]
    if len(hit):
        return {pd.Timestamp(r.expiry): SVISlice(*[float(getattr(r, f)) for f in FIELDS[:-3]], int(r.n),
                                                  float(r.kmin), float(r.kmax))
                for r in hit.itertuples() if pd.notna(r.expiry)}
    out = _fit_date(con, underlying, d, T_min, T_max)
    rows = [{"date": d, "expiry": e, **{f: getattr(s, f) for f in FIELDS}} for e, s in out.items()]
    _NEW.setdefault(underlying, []).extend(rows or [{"date": d, "expiry": pd.NaT, **{f: np.nan for f in FIELDS}}])
    return out


def _fit_date(con, underlying: str, d, T_min, T_max) -> dict[pd.Timestamp, SVISlice]:
    q = _quotes(con, underlying, d)
    out = {}
    for exp_, g in q[(q["T"] >= T_min) & (q["T"] <= T_max)].groupby("expiry"):
        g = g.drop_duplicates("k")
        if len(g) < MIN_POINTS:
            continue
        T = float(g["T"].iloc[0])
        out[pd.Timestamp(exp_)] = fit_svi(g.k.to_numpy(), (g.iv ** 2 * T).to_numpy(), T)
    return out


def _forward(con, underlying, d, S, T):
    f = con.sql(f"""select expiry, forward from canon.forward_curve
                    where underlying = '{underlying}' and date = '{d.date()}' and expiry > date
                    order by expiry""").df()
    if f.empty:
        return None
    t = (pd.to_datetime(f.expiry) - d).dt.days.to_numpy() / 365.0
    y = np.log(f.forward.to_numpy() / S)
    if T <= t.max():
        return S * exp(float(np.interp(T, t, y)))
    slope = y[-1] / t[-1]                                  # constant carry beyond the last expiry
    return S * exp(slope * T)


def interpolate(slices: dict, T: float):
    """Total-variance function w(k) at maturity T from the fitted slices."""
    s = sorted(slices.values(), key=lambda x: x.T)
    if not s:
        return None
    below = [x for x in s if x.T <= T]
    above = [x for x in s if x.T >= T]
    if below and above:
        s1, s2 = below[-1], above[0]
        if s2.T == s1.T:
            return s1.w
        u = (T - s1.T) / (s2.T - s1.T)
        return lambda k: (1 - u) * s1.w(k) + u * s2.w(k)
    near = (below or above)[-1 if below else 0]
    return lambda k: near.w(k) * T / near.T                # constant implied vol beyond the quotes


# ------------------------------------------------------------------------------- markets
_FITS: list[dict] = []


def load_markets_svi(con, underlying: str, items, tenor_days: int = 365) -> dict:
    """Market objects with an SVI smile.
    items: dates (tenor = tenor_days) or (date, T_years) pairs for exact maturities.
    Keyed by date (or by (date, T) when pairs are given)."""
    pairs = [(pd.Timestamp(x[0]), float(x[1])) if isinstance(x, tuple) else (pd.Timestamp(x), tenor_days / 365.0)
             for x in items]
    keyed_by_pair = any(isinstance(x, tuple) for x in items)
    base = load_markets(con, underlying, sorted({d for d, _ in pairs}))   # spot, zero rate, fallback smile
    zero = con.sql(f"""select date, tenor_days, rate from canon.zero_curve
                       where date in ({','.join(f"'{d.date()}'" for d, _ in pairs) or "null"})""").df()
    zero["date"] = pd.to_datetime(zero.date)
    zc = {d: g.sort_values("tenor_days") for d, g in zero.groupby("date")}
    cache, out = {}, {}
    for d, T in pairs:
        key = (d, T) if keyed_by_pair else d
        if d not in base:
            continue
        S = base[d].S
        if d not in cache:
            try:
                cache[d] = fit_date(con, underlying, d)
            except Exception:  # noqa: BLE001
                cache[d] = {}
        slices = cache[d]
        wfun = interpolate(slices, T)
        F = _forward(con, underlying, d, S, T) if wfun else None
        if wfun is None or F is None or d not in zc:
            fb = base[d] if abs(base[d].T - T) < 1e-9 else Market(d, S, base[d].r, base[d].q, T, base[d].log_m, base[d].iv)
            out[key] = fb
            _FITS.append({"date": d, "T": T, "method": "delta-grid fallback", "slices": len(slices)})
            continue
        g = zc[d]
        r = float(np.interp(T * 365, g.tenor_days.to_numpy(), g.rate.to_numpy()))
        q = r - log(F / S) / T
        w = np.maximum(wfun(K_GRID), 1e-10)
        out[key] = Market(d, S, r, q, T, K_GRID.copy(), np.sqrt(w / T))
        near = min(slices.values(), key=lambda x: abs(x.T - T))
        _FITS.append({"date": d, "T": T, "method": "svi", "slices": len(slices),
                      "nearest_T": near.T, "rmse_vol": near.rmse_vol, "points": near.n,
                      "butterfly_free": near.butterfly_free(), "rho": near.rho, "b": near.b})
    return out


def fit_report() -> pd.DataFrame:
    return pd.DataFrame(_FITS)


def monthly_starts(con, underlying: str = "SPX") -> list[pd.Timestamp]:
    from engine.pricing import trading_dates
    idx = trading_dates(con, underlying)
    months = pd.date_range(idx.min(), idx.max(), freq="MS")
    return [idx[idx.searchsorted(m)] for m in months if idx.searchsorted(m) < len(idx)]


if __name__ == "__main__":
    # Pre-fit and cache the SVI slices for every monthly launch date (about 1.5 s per date).
    #   python engine/surface.py [budget_seconds]
    import time

    from engine.canonical import connect

    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 1e9
    con, t0 = connect(), time.time()
    todo = [d for d in monthly_starts(con) if not (_cache("SPX").date == d).any()]
    for i, d in enumerate(todo):
        fit_date(con, "SPX", d)
        if (i + 1) % 20 == 0:
            save_cache()
        if time.time() - t0 > budget:
            break
    save_cache()
    c = _cache("SPX")
    print(f"SVI cache: {c.date.nunique()} dates, {c.expiry.notna().sum()} slices; "
          f"{len(todo) - (i + 1 if todo else 0)} monthly dates still to fit")
