"""1-minute-grid pull and strip computation, sized for ~1,060 sessions.

Differences from the 5-minute pilot path, all forced by scale:

  grid      390 one-minute marks, 09:30..15:59. The last mark expires at 16:00
            so T=0 there and it carries no strip; 389 usable bars per session.
  storage   one Parquet file per session instead of 542 per-contract CSVs.
            At 1-minute resolution the CSV layout came to ~27 GB before the
            per-contract copies; Parquet with zstd holds it to a few GB.
  underlying  optional. Stocks aggregates are capped at 2 years on this plan,
            so before ~2024-08-26 there are no observed SPY bars and the
            forward comes from put-call parity on the options themselves. That
            is enough for the strip, but it makes the parity check circular,
            so the check is skipped and flagged rather than reported as passing.

The strip itself is recomputed here on numpy arrays rather than per-bar
DataFrames. massive/strip.py stays the reference implementation -- it is the
one validated against Cboe's published worked examples -- and check_fast()
asserts the two agree before this path is trusted.
"""

import os
from datetime import datetime, time as dtime, timedelta

import numpy as np
import pandas as pd

from .client import MARKET_TZ, Massive, ns
from .strip import MIN_PER_YEAR

GRID_START, GRID_END = dtime(9, 30), dtime(16, 0)
N_BARS = 390


def grid(day):
    d = pd.Timestamp(day).date()
    t0 = datetime.combine(d, GRID_START, tzinfo=MARKET_TZ)
    return [t0 + timedelta(minutes=i) for i in range(N_BARS)]


def bin_quotes(m, ticker, day, edges_ns, max_pages=400):
    """Last NBBO in each 1-minute bin. Returns (bid, ask, bsz, asz, n) arrays."""
    n = len(edges_ns)
    bid = np.full(n, np.nan, "float32"); ask = np.full(n, np.nan, "float32")
    bsz = np.full(n, np.nan, "float32"); asz = np.full(n, np.nan, "float32")
    cnt = np.zeros(n, "int32")
    i = 0
    for q in m.option_quotes(ticker, ns(day, GRID_START), edges_ns[-1], max_pages=max_pages):
        t = q["sip_timestamp"]
        while i < n and t > edges_ns[i]:
            i += 1
        if i >= n:
            break
        bid[i] = q.get("bid_price", np.nan); ask[i] = q.get("ask_price", np.nan)
        bsz[i] = q.get("bid_size", np.nan);  asz[i] = q.get("ask_size", np.nan)
        cnt[i] += 1
    return bid, ask, bsz, asz, cnt


# ---------------------------------------------------------------- strip, fast

# Cboe amended the strike-selection rule on 2025-02-10 so that a zero ASK excludes a
# strike as well as a zero bid. The amendment matters far more than its size suggests,
# because a CROSSED quote -- bid > 0 with ask = 0.00 -- yields a mid that is finite,
# small and meaningless. On 2025-04-09 at 13:20 nine call legs were quoted that way,
# and the 489 strike (bid 0.70 / ask 0.00, mid 0.35) consequently won the
# argmin|C - P| forward search: F came back 489.20 against a true forward near 516,
# which dragged ~30 points of intrinsic value into the variance integral and printed
# a strip of 148.7 vol points between neighbours of 60.7 and 42.5. That single bar
# carried essentially the whole of the paper's unexplained residual.
#
# Set False only to reproduce a pre-amendment run.
ZERO_ASK_EXCLUDES = True


def _unquoted(b, a):
    """True if this leg has no usable quote: no bid, or -- post 2025-02-10 -- no ask."""
    if not np.isfinite(b) or b <= 0:
        return True
    return ZERO_ASK_EXCLUDES and (not np.isfinite(a) or a <= 0)


def _cboe_keep(K, bid, ask, mid, k0, upward):
    """Cboe scan. An isolated unquoted strike skips; two consecutive stop and drop both."""
    idx = np.where(K > k0)[0] if upward else np.where(K < k0)[0][::-1]
    keep, zeros = [], 0
    for j in idx:
        if _unquoted(bid[j], ask[j]):
            zeros += 1
            if zeros >= 2:
                break
            continue
        zeros = 0
        if np.isfinite(mid[j]):
            keep.append(j)
    return keep


def strip_bar(K, cb, ca, cm, pb, pa, pm, T, variant, r=0.0):
    # The forward search must see only two-sided quotes. A crossed leg otherwise wins
    # the argmin outright, since its bogus mid sits far below the true call-put gap.
    ok = np.isfinite(cm) & np.isfinite(pm)
    if ZERO_ASK_EXCLUDES:
        ok = ok & np.isfinite(ca) & (ca > 0) & np.isfinite(pa) & (pa > 0)
    if T <= 0 or not ok.any():
        return None
    j = np.where(ok)[0][np.argmin(np.abs(cm[ok] - pm[ok]))]
    F = K[j] + np.exp(r * T) * (cm[j] - pm[j])
    below = K[K <= F]
    if not len(below):
        return None
    k0 = below.max()
    i0 = int(np.where(K == k0)[0][0])

    if variant == "cboe":
        js = _cboe_keep(K, pb, pa, pm, k0, False) + _cboe_keep(K, cb, ca, cm, k0, True)
        ks = K[js]
        qs = np.where(ks < k0, pm[js], cm[js])
    else:
        sel = K != k0
        isc = K > k0
        b = np.where(isc, cb, pb); a = np.where(isc, ca, pa); mv = np.where(isc, cm, pm)
        # A crossed leg is malformed rather than a wing, so it is dropped outright
        # instead of being valued: the lower/upper pair is meant to bound the
        # ZERO-BID problem, and a bogus mid inside the band defeats that.
        crossed = np.isfinite(b) & (b > 0) & (~np.isfinite(a) | (a <= 0))
        sel = sel & ~crossed
        live = np.isfinite(b) & (b > 0) & np.isfinite(mv) & ~crossed
        if variant == "lower":
            fill = 0.0
        elif variant == "ask_full":
            fill = a
        else:
            fill = np.maximum(a - 0.01, 0.0)
        val = np.where(live, mv, fill)
        use = sel & (live | np.isfinite(a))
        ks, qs = K[use], val[use]

    if np.isfinite(cm[i0]) and np.isfinite(pm[i0]):
        ks = np.append(ks, k0); qs = np.append(qs, (cm[i0] + pm[i0]) / 2)
    if len(ks) < 3:
        return None
    o = np.argsort(ks); ks, qs = ks[o], qs[o]
    dk = np.empty_like(ks)
    dk[1:-1] = (ks[2:] - ks[:-2]) / 2
    dk[0], dk[-1] = ks[1] - ks[0], ks[-1] - ks[-2]
    s2 = (2 / T) * np.sum(dk / ks**2 * np.exp(r * T) * qs) - (1 / T) * (F / k0 - 1) ** 2
    return {"sigma2": s2, "vol": 100 * np.sqrt(max(s2, 0.0)), "n_strikes": len(ks),
            "k_lo": ks.min(), "k_hi": ks.max(), "F": F, "K0": k0}


VARIANTS = ("lower", "cboe", "upper", "ask_full")


def strip_day(day, path=None):
    """All four variants for every bar of one session, from its Parquet file."""
    path = path or f"data/pilot1m/{day}/bars.parquet"
    df = pd.read_parquet(path)
    Ks = np.sort(df.strike.unique()).astype("float64")
    ki = {k: i for i, k in enumerate(Ks)}
    nb, nk = N_BARS, len(Ks)
    A = {f"{r}_{f}": np.full((nb, nk), np.nan) for r in ("c", "p") for f in ("bid", "ask", "mid")}
    for r, tag in (("C", "c"), ("P", "p")):
        s = df[df.right == r]
        if s.empty:            # a session with one leg missing entirely
            continue
        bi = s.bar.to_numpy(dtype=np.intp)
        kk = np.asarray([ki[k] for k in s.strike], dtype=np.intp)
        A[f"{tag}_bid"][bi, kk] = s.bid; A[f"{tag}_ask"][bi, kk] = s.ask
        A[f"{tag}_mid"][bi, kk] = (s.bid + s.ask) / 2

    g = grid(day)
    rows = []
    for i in range(nb):
        end = g[i] + timedelta(minutes=1)
        T = max((pd.Timestamp(f"{day} 16:00", tz=MARKET_TZ) - end).total_seconds() / 60, 0) / MIN_PER_YEAR
        if T <= 0:
            continue
        args = (Ks, A["c_bid"][i], A["c_ask"][i], A["c_mid"][i],
                A["p_bid"][i], A["p_ask"][i], A["p_mid"][i], T)
        row = {"bar_start": g[i], "bar": i, "T": T}
        bad = False
        for v in VARIANTS:
            s = strip_bar(*args, v)
            if s is None:
                bad = True
                break
            row[f"vol_{v}"] = s["vol"]; row[f"n_{v}"] = s["n_strikes"]
            if v == "cboe":
                row.update(F=s["F"], K0=s["K0"], k_lo=s["k_lo"], k_hi=s["k_hi"])
        if not bad:
            rows.append(row)
    out = pd.DataFrame(rows)
    if len(out):
        out["sig_T"] = out.vol_cboe / 100 * np.sqrt(out["T"])
        out["z_lo"] = np.log(out.k_lo / out.F) / out.sig_T
        out["z_hi"] = np.log(out.k_hi / out.F) / out.sig_T
    return out


def check_fast(day="2026-06-05"):
    """Assert the numpy strip matches the validated reference on real data."""
    from .strip import cross_section, strip as ref_strip
    q = pd.read_csv(f"data/pilot/{day}/quote_bars_all.csv", parse_dates=["bar_start"])
    ch = pd.read_csv(f"data/pilot/{day}/chain.csv")[["ticker", "strike_price", "contract_type"]]
    q = q.merge(ch, on="ticker")
    worst = {v: 0.0 for v in VARIANTS}
    n = 0
    for bs, gg in q.groupby("bar_start"):
        end = bs + timedelta(minutes=5)
        T = max((pd.Timestamp(f"{day} 16:00", tz=bs.tz) - end).total_seconds() / 60, 0) / MIN_PER_YEAR
        if T <= 0:
            continue
        xs = cross_section(gg)
        K = xs.K.to_numpy(float)
        args = (K, xs.call_bid.to_numpy(float), xs.call_ask.to_numpy(float),
                xs.call_mid.to_numpy(float), xs.put_bid.to_numpy(float),
                xs.put_ask.to_numpy(float), xs.put_mid.to_numpy(float), T)
        for v in VARIANTS:
            a, b = ref_strip(xs, T, v), strip_bar(*args, v)
            if a is None or b is None:
                assert (a is None) == (b is None), f"{v} disagrees on None at {bs}"
                continue
            worst[v] = max(worst[v], abs(a["vol"] - b["vol"]))
        n += 1
    print(f"fast vs reference on {day}: {n} bars")
    for v in VARIANTS:
        print(f"  {v:9s} max |vol diff| = {worst[v]:.3e}")
    assert max(worst.values()) < 1e-9, "fast strip diverges from reference"
    print("  OK - identical to the validated implementation")
