"""Pull the 5-minute SPY and VIX1D series that intraday.csv is built from.

VIX1D is the 1-day expected-volatility index, so it is the right variance state
for a 0DTE option -- the 30-day VIX is measuring a different maturity entirely.
Yahoo only serves 5-minute index data for the trailing 60 trading days, so the
sample window is whatever those 60 days happen to be on the day this is run.

    python fetch_data.py [output.csv]
"""

import sys

import pandas as pd
import yfinance as yf

UNDERLYING = "SPY"
VOL_INDEX = "^VIX1D"
BARS_PER_DAY = 78          # 09:30 to 15:55 inclusive, 5-minute marks


def pull(ticker):
    df = yf.download(ticker, period="60d", interval="5m",
                     progress=False, auto_adjust=False)
    if df.empty:
        raise RuntimeError(f"no 5-minute data came back for {ticker}")
    df.columns = [c[0] for c in df.columns]      # drop the ticker level
    return df["Close"].rename(ticker)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "intraday.csv"

    und = pull(UNDERLYING)
    vol = pull(VOL_INDEX)
    df = pd.concat([und, vol], axis=1).dropna()
    df.index.name = "timestamp"
    df = df.rename(columns={UNDERLYING: "und_close", VOL_INDEX: "vix1d_close"})

    # keep only days with a complete grid; a partial day would make dt wrong
    counts = df.groupby(df.index.date).size()
    full = counts[counts == BARS_PER_DAY].index
    df = df[pd.Series(df.index.date, index=df.index).isin(full)]

    df.to_csv(out)
    days = sorted(set(df.index.date))
    print(f"wrote {out}: {len(days)} full days x {BARS_PER_DAY} bars "
          f"({days[0]} to {days[-1]})")
    dropped = counts[counts != BARS_PER_DAY]
    if len(dropped):
        print("skipped incomplete days:", dict(dropped))


if __name__ == "__main__":
    main()
