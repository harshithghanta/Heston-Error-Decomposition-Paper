"""NDX and QQQ 0DTE pulls on the same 1-minute grid as SPY and SPX.

    python -m puller.altpull ndx --days 2024-06-05 2025-04-09      # smoke test first
    python -m puller.altpull ndx --start 2022-06-08 --end 2026-08-25
    python -m puller.altpull qqq --start 2022-06-08 --end 2026-08-25

Like `spxpull.py`, this file is AI-assisted (original data pipeline, not part of
the code rewrite).

Why these two, and why nothing else
-----------------------------------
The preprint compares a 2x2: {S&P 500, Nasdaq-100} x {ETF American-style, Index
European-style}. SPY and SPX give the first row, QQQ and NDX the second. XSP was
considered and dropped -- it is the same index and the same cash settlement as
SPX at a tenth the notional, so it varies no factor, only liquidity.

The two products need opposite handling, and it is the same split SPY/SPX already
has:

  QQQ  is an ETF, so it behaves exactly like SPY. It has tradeable stock bars and
       ordinary OCC tickers, so `bigpull` runs unmodified and only the symbol and
       output root change.

  NDX  is an index, so it behaves exactly like SPX. There are no stock bars, the
       forward comes from put-call parity, and daily-expiry contracts are listed
       under a weeklys root (NDXP) rather than the index symbol -- so tickers must
       be read off the chain rather than constructed. `spxpull` already solved all
       three; its overrides are reused here rather than copied.

Both write to their own `data/<sym>1m/` root. Never `data/pilot1m`, which is the
main SPY panel and which several scripts rewrite in
place.
"""

import os
import sys

from . import bigpull as bp
from . import spxpull as spx

# name -> (underlying query symbol, output root, index-style?)
PRODUCTS = {
    "ndx": ("NDX", "data/ndx1m", True),
    "qqq": ("QQQ", "data/qqq1m", False),
}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0].lower() not in PRODUCTS:
        sys.exit(f"usage: python -m puller.altpull {{{'|'.join(PRODUCTS)}}} [bigpull args]")

    name = argv.pop(0).lower()
    symbol, root, index_style = PRODUCTS[name]

    bp.UNDERLYING = symbol
    bp.ROOT = root
    if index_style:
        # No stock bars for an index; forward by parity, tickers read off the chain.
        # bigpull records these sessions as has_underlying=False, the same branch
        # pre-2024 SPY sessions already take.
        bp.spot_probe = spx.spot_probe
        bp.underlying_bars = spx.underlying_bars

    os.makedirs(root, exist_ok=True)
    kind = "index (parity forward)" if index_style else "ETF (stock bars)"
    print(f"{symbol} pull -> {root}/  ({kind})", flush=True)
    return bp.main(argv)


if __name__ == "__main__":
    main()
