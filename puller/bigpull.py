"""2022-2026 SPY 0DTE pull on a 1-minute grid.

    python -m puller.bigpull --start 2022-06-08 --end 2026-08-25
    python -m puller.bigpull --days 2022-06-08 2024-06-05 2026-06-05   # smoke

Resumable per session. A session already holding bars.parquet and strip.csv is
skipped without a single request, so this can be killed and restarted freely.

Underlying bars are optional by design: stocks aggregates are capped at two
years on this plan, so before ~2024-08-26 SPY 1-minute bars 403 and the forward
comes from put-call parity instead. That is recorded per session in
`has_underlying`, and the parity sanity check is SKIPPED rather than reported
when it is false -- with F derived from the options, C-P-(F-K) is identically
zero and would look like a pass.
"""

import argparse
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, time as dtime, timedelta

import numpy as np
import pandas as pd

from .client import AuthError, MARKET_TZ, Massive, MassiveError, et, ns, occ_ticker
from .fast import GRID_START, N_BARS, bin_quotes, grid, strip_day
from .pilot import DateMismatch, assert_day, assert_tickers

ROOT = "data/pilot1m"
UNDERLYING = "SPY"


def is_plan_limit(exc):
    s = str(exc)
    return "timeframe" in s or "NOT_AUTHORIZED" in s


def spot_probe(m, day, chain, n=24, workers=8):
    """Forward from put-call parity, with no underlying feed.

    F = K + e^{rT}(C - P) holds at every strike, so any one strike with a
    two-sided market on BOTH legs pins the forward. Deep strikes drop out
    automatically -- the far leg has no bid -- so the survivors are near the
    money, which is where parity is tightest. Median over the survivors.

    Replaces a strike-spacing heuristic that read 314.5 for a session whose
    spot was ~411: listing density does not track the money reliably.
    """
    Ks = sorted({r["strike_price"] for r in chain})
    step = max(1, len(Ks) // (n + 1))
    sample = Ks[step::step][:n]
    ts, floor = ns(day, dtime(12, 0)), ns(day, GRID_START)

    def leg(k):
        out = {}
        for r in ("C", "P"):
            q = m.last_quote_at_or_before(occ_ticker(UNDERLYING, day, r, k), ts, floor_ns=floor)
            if not q:
                return None
            b, a = q.get("bid_price"), q.get("ask_price")
            if b is None or a is None or b <= 0 or a <= b:
                return None
            out[r] = (b + a) / 2
        return k + out["C"] - out["P"]

    Fs = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for f in as_completed([ex.submit(leg, k) for k in sample]):
            try:
                v = f.result()
            except Exception:
                v = None
            if v is not None:
                Fs.append(v)
    return float(np.median(Fs)) if Fs else None


def universe(chain, S, near=0.03, wing=0.25):
    """Both legs near the money; OTM only beyond it.

    The strip integrates OTM options and stops at two consecutive zero bids,
    which in the pilot never reached past ~6% of forward. Deep ITM contracts
    are never used and are the heaviest to pull, so they are skipped. The
    near-the-money band keeps both legs because the forward needs them.
    """
    keep = []
    for r in chain:
        m_ = r["strike_price"] / S - 1
        c = r["contract_type"] == "call"
        if abs(m_) <= near or (c and near < m_ <= wing) or (not c and -wing <= m_ < -near):
            keep.append(r["ticker"])
    return sorted(keep)


def underlying_bars(m, day):
    """SPY 1-minute RTH bars, or None if the plan's history does not reach."""
    try:
        rows = m.stock_bars(UNDERLYING, day, multiplier=1, timespan="minute")
    except AuthError as exc:
        if is_plan_limit(exc):
            return None
        raise
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df["ts"] = [et(ts_ms=t) for t in df["t"]]
    df = df[[(t.time() >= grid(day)[0].time()) and (t.time() <= grid(day)[-1].time())
             for t in df["ts"]]].sort_values("ts")
    if df.empty:
        return None
    assert_day(day, df.ts.dt.date, "underlying 1-min bars")
    return df[["ts", "o", "h", "l", "c", "v"]].rename(
        columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})


def pull_session(m, day, workers=8, max_pages=400):
    out = os.path.join(ROOT, str(day))
    os.makedirs(out, exist_ok=True)

    rows, _ = m.option_contracts(UNDERLYING, expiration_date=str(day), as_of=str(day))
    if not rows:
        return {"day": str(day), "status": "no_0dte"}
    ch = pd.DataFrame(rows)
    assert_day(day, ch.expiration_date, f"{day} chain")
    assert_tickers(day, ch.ticker, f"{day} chain")
    ch[["ticker", "strike_price", "contract_type"]].to_parquet(
        os.path.join(out, "chain.parquet"), index=False)

    S = spot_probe(m, str(day), rows, workers=workers)
    tickers = universe(rows, S) if S else sorted(ch.ticker)

    und = underlying_bars(m, day)
    if und is not None:
        und.to_parquet(os.path.join(out, "underlying.parquet"), index=False)

    g = grid(day)
    edges = np.array([int((t + timedelta(minutes=1)).timestamp() * 1e9) for t in g])
    meta = dict(zip(ch.ticker, zip(ch.strike_price, ch.contract_type)))

    def one(tk):
        b, a, bs, as_, c = bin_quotes(m, tk, str(day), edges, max_pages=max_pages)
        live = np.isfinite(b) | np.isfinite(a)          # drop never-quoted bins
        if not live.any():
            return None
        k, r = meta[tk]
        return pd.DataFrame({
            "ticker": tk, "strike": np.float32(k), "right": r[0].upper(),
            "bar": np.arange(N_BARS, dtype="int16")[live],
            "bid": b[live], "ask": a[live], "bid_size": bs[live],
            "ask_size": as_[live], "n_quotes": c[live]})

    frames, failed, ticks = [], {}, 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(one, t): t for t in tickers}
        for f in as_completed(futs):
            try:
                d = f.result()
            except Exception as exc:
                failed[futs[f]] = f"{type(exc).__name__}: {str(exc)[:80]}"
                continue
            if d is not None:
                ticks += int(d.n_quotes.sum())
                frames.append(d)
    if not frames:
        return {"day": str(day), "status": "no_quotes", "n_contracts": len(ch)}

    bars = pd.concat(frames, ignore_index=True)
    bars["right"] = bars.right.astype("category")
    bars["ticker"] = bars.ticker.astype("category")
    bars.to_parquet(os.path.join(out, "bars.parquet"), index=False, compression="zstd")
    return {"day": str(day), "status": "ok", "n_contracts": len(ch),
            "n_pulled": len(tickers), "spot": S, "n_failed": len(failed),
            "ticks": ticks, "rows": len(bars), "has_underlying": und is not None}


def sessions(start, end):
    d, out = pd.Timestamp(start).date(), []
    e = pd.Timestamp(end).date()
    while d <= e:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2022-06-08")
    ap.add_argument("--end", default="2026-08-25")
    ap.add_argument("--days", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-pages", type=int, default=400)
    a = ap.parse_args(argv)

    days = [pd.Timestamp(x).date() for x in a.days] if a.days else sessions(a.start, a.end)
    print(f"{len(days)} candidate weekdays, {days[0]} .. {days[-1]}", flush=True)
    try:
        m = Massive()
    except AuthError as exc:
        sys.exit(str(exc))

    log, t0 = [], time.time()
    for i, day in enumerate(days, 1):
        out = os.path.join(ROOT, str(day))
        sp, bp = os.path.join(out, "strip.csv"), os.path.join(out, "bars.parquet")
        if os.path.exists(sp):
            log.append({"day": str(day), "status": "cached"})
            continue
        try:
            r = pull_session(m, day, a.workers, a.max_pages) if not os.path.exists(bp) \
                else {"day": str(day), "status": "ok", "resumed": True}
            if r["status"] == "ok":
                s = strip_day(str(day), bp)
                s.to_csv(sp, index=False)
                r["n_bars"] = len(s)
                r["vol_cboe"] = float(s.vol_cboe.median()) if len(s) else np.nan
        except (DateMismatch, MassiveError, Exception) as exc:
            r = {"day": str(day), "status": f"FAIL {type(exc).__name__}", "err": str(exc)[:120]}
            traceback.print_exc(limit=2)
        log.append(r)
        el = time.time() - t0
        done = sum(1 for x in log if x.get("status") == "ok")
        print(f"  [{i:4d}/{len(days)}] {day} {r['status']:>10} "
              f"bars={r.get('n_bars','-'):>4} vol={r.get('vol_cboe',float('nan')):6.2f} "
              f"und={r.get('has_underlying','-')} "
              f"| ok={done} {el/60:5.1f}m {m.calls:,} calls", flush=True)
        pd.DataFrame(log).to_csv(os.path.join(ROOT, "_manifest.csv"), index=False)

    ok = sum(1 for x in log if x.get("status") in ("ok", "cached"))
    print(f"\n{ok}/{len(days)} sessions on disk. {m.calls:,} calls, "
          f"{(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
