"""Pull the 5-minute SPY and VIX1D series that intraday.csv is built from.

VIX1D is the 1-day expected-volatility index, so it is the right variance state
for a 0DTE option -- the 30-day VIX is measuring a different maturity entirely.

Yahoo only serves 5-minute data for the trailing 60 trading days. That window
rolls: a session that falls off the back is gone for good from this source. So
this script is an *archive builder*, not a snapshot. It unions what comes back
with whatever the target file already holds and rewrites the union, which means
the sample grows every time it runs instead of sliding forward. Run it weekly.

Two files, deliberately:

    archive.csv    grows forever; what the cron writes; the future sample
    intraday.csv   frozen at the 57 days the current paper reports

The paper is reproducible only if its input stops moving, so the archive is kept
separate and intraday.csv is promoted from it by hand when a new version is cut.
Writing to intraday.csv is possible but has to be asked for by name.

Existing rows are never overwritten. Yahoo does revise its intraday history --
11 of the 4,446 bars in the committed intraday.csv differ from what a fresh pull
returns today, one of them by 0.35 VIX1D points -- so the archived value wins and
the discrepancy is reported rather than applied.

    python fetch_data.py                         append to archive.csv
    python fetch_data.py some.csv                append to some.csv
    python fetch_data.py --fresh some.csv        ignore any existing file
"""

import os
import sys

import pandas as pd
import yfinance as yf

UNDERLYING = "SPY"
VOL_INDEX = "^VIX1D"
BARS_PER_DAY = 78          # 09:30 to 15:55 inclusive, 5-minute marks
MARKET_TZ = "America/New_York"
ARCHIVE = "archive.csv"    # the growing file; intraday.csv stays frozen
SEED = "intraday.csv"      # what a fresh archive starts from


def pull(ticker):
    df = yf.download(ticker, period="60d", interval="5m",
                     progress=False, auto_adjust=False)
    if df.empty:
        raise RuntimeError(f"no 5-minute data came back for {ticker}")
    df.columns = [c[0] for c in df.columns]      # drop the ticker level
    return df["Close"].rename(ticker)


def to_market_tz(index):
    """Normalise an index to wall-clock New York.

    The archive will eventually straddle a DST change, at which point a naive
    string parse yields mixed -04:00/-05:00 offsets and an object-dtype index.
    Going through UTC keeps it a single tz-aware dtype across the boundary.
    """
    idx = pd.DatetimeIndex(index)
    if idx.tz is None:
        return idx.tz_localize(MARKET_TZ)
    return idx.tz_convert(MARKET_TZ)


def read_existing(path):
    if not os.path.exists(path):
        return None
    # round_trip, not the default parser: the fast one is good to ~1 ULP, which
    # would nudge every archived value on every run and drift published numbers.
    old = pd.read_csv(path, index_col=0, float_precision="round_trip")
    old.index = to_market_tz(pd.to_datetime(old.index, utc=True))
    old.index.name = "timestamp"
    return old


def fetch():
    und = pull(UNDERLYING)
    vol = pull(VOL_INDEX)
    df = pd.concat([und, vol], axis=1).dropna()
    df.index = to_market_tz(df.index)
    df.index.name = "timestamp"
    return df.rename(columns={UNDERLYING: "und_close", VOL_INDEX: "vix1d_close"})


def report_revisions(old, new):
    """Count timestamps Yahoo now disagrees with us about. Archive value wins."""
    shared = old.index.intersection(new.index)
    if len(shared) == 0:
        return
    diff = (old.loc[shared] - new.loc[shared]).abs()
    changed = diff[(diff > 1e-9).any(axis=1)]
    if len(changed):
        print(f"note: Yahoo revised {len(changed)} archived bar(s); keeping the "
              f"archived values. First: {changed.index[0]}")


def main():
    argv = [a for a in sys.argv[1:] if a != "--fresh"]
    fresh = "--fresh" in sys.argv[1:]
    out = argv[0] if argv else ARCHIVE

    new = fetch()
    old = None if fresh else read_existing(out)

    # first run: start the archive from the published sample rather than from
    # the 60-day window, so the May-July days are carried forward not lost.
    if old is None and not fresh and out == ARCHIVE:
        old = read_existing(SEED)
        if old is not None:
            print(f"seeding {ARCHIVE} from {SEED} ({len(old)} rows)")

    if old is None:
        df = new
        before = 0
    else:
        report_revisions(old, new)
        before = len(old)
        # old first: duplicated(keep="first") drops the incoming copy, so the
        # archive is append-only and re-running adds nothing.
        df = pd.concat([old, new])
        df = df[~df.index.duplicated(keep="first")]

    df = df.sort_index()

    # keep only days with a complete grid; a partial day would make dt wrong.
    # this also drops early closes (half-sessions have 48 bars, not 78).
    counts = df.groupby(df.index.date).size()
    full = counts[counts == BARS_PER_DAY].index
    df = df[pd.Series(df.index.date, index=df.index).isin(full)]

    df.to_csv(out)
    days = sorted(set(df.index.date))
    print(f"wrote {out}: {len(days)} full days x {BARS_PER_DAY} bars "
          f"({days[0]} to {days[-1]})")
    print(f"  {before} rows on file before, {len(df)} after "
          f"(+{len(df) - before})")
    dropped = counts[counts != BARS_PER_DAY]
    if len(dropped):
        print("skipped incomplete days:", dict(dropped))


if __name__ == "__main__":
    main()
