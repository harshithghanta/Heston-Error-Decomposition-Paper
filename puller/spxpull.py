"""SPX 0DTE pull on the same 1-minute grid as the SPY chain.

    python -m puller.spxpull --days 2024-06-05 2025-04-09      # smoke test first
    python -m puller.spxpull --start 2022-06-08 --end 2026-08-25

This file is AI-assisted (original data pipeline, not part of the code rewrite).

It does not modify `bigpull.py`. Everything is reused from there; only the four
things that are genuinely SPY-specific are overridden, and the overrides are
installed on the bigpull module at run time so the SPY pull stays byte-identical.

What actually differs for SPX
-----------------------------
1. The contract query keys on `SPX`. Verified against the API 2026-09-17:
   `SPX` returns 598 contracts for 2025-04-09, while `I:SPX`, `SPXW` and `$SPX`
   all return zero. The index-prefixed form is what the aggregates endpoints
   want, not the contract index -- an easy and silent way to pull nothing.
2. Daily-expiry SPX options are listed as SPXW (the weeklys root), so tickers
   read `O:SPXW260918C06000000`. `bigpull.spot_probe` builds tickers with
   `occ_ticker(UNDERLYING, ...)`, which would produce `O:I:SPX...` -- wrong on
   both counts. The override reads the ticker off the chain row instead, which
   is what the API already returned and is correct for any root.
3. There are no `stock_bars` for an index, so the underlying is always None and
   the forward comes from put-call parity. `bigpull` already supports that path
   for pre-2024 SPY sessions and records it as `has_underlying=False`.
4. Output goes to `data/spx1m/`. Never `data/pilot1m/`, which is the SPY panel
   the paper is built on and which several scripts overwrite in place.

The 09:30-16:00 grid, the 390 one-minute bars, the near/wing strike filter and
the Cboe strip construction are all shared with SPY on purpose: the point of
this dataset is a like-for-like comparison, so anything that differs by
construction would confound it.

A note for the spring analysis: SPX is European and cash-settled, so put-call
parity holds exactly and there is no early-exercise or dividend wedge. That
removes a caveat the SPY results have to state.
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import time as dtime

import numpy as np

from . import bigpull as bp
from .client import ns
from .fast import GRID_START

UNDERLYING = "SPX"            # NOT "I:SPX" -- see the note above; verified 2026-09-17
ROOT = "data/spx1m"           # never data/pilot1m


def spot_probe(m, day, chain, n=12, workers=8):
    """Forward from put-call parity at midday, using tickers off the chain.

    Same estimator as bigpull's: sample strikes across the listed range, keep
    only strikes where BOTH legs are two-sided, take the median of K + C - P.
    The only change is that the call and put tickers are looked up in the chain
    rather than constructed, because the SPXW root cannot be derived from the
    underlying symbol.
    """
    by_strike = {}
    for r in chain:
        k = r["strike_price"]
        side = "C" if r["contract_type"] == "call" else "P"
        by_strike.setdefault(k, {})[side] = r["ticker"]

    both = sorted(k for k, v in by_strike.items() if "C" in v and "P" in v)
    if not both:
        return None
    step = max(1, len(both) // (n + 1))
    sample = both[step::step][:n]
    ts, floor = ns(day, dtime(12, 0)), ns(day, GRID_START)

    def leg(k):
        out = {}
        for side in ("C", "P"):
            q = m.last_quote_at_or_before(by_strike[k][side], ts, floor_ns=floor)
            if not q:
                return None
            b, a = q.get("bid_price"), q.get("ask_price")
            if b is None or a is None or b <= 0 or a <= b:
                return None
            out[side] = (b + a) / 2
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


def underlying_bars(m, day):
    """Always None: an index has no tradeable 1-minute stock bars.

    bigpull treats this as the parity path and records has_underlying=False,
    which is the same branch pre-2024-08-26 SPY sessions already take.
    """
    return None


def main(argv=None):
    bp.UNDERLYING = UNDERLYING
    bp.ROOT = ROOT
    bp.spot_probe = spot_probe
    bp.underlying_bars = underlying_bars
    os.makedirs(ROOT, exist_ok=True)
    print(f"SPX pull -> {ROOT}/  (underlying query {UNDERLYING}, SPXW roots)", flush=True)
    return bp.main(argv)


if __name__ == "__main__":
    main()
