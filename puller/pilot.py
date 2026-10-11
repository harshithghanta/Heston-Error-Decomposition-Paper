"""One-session pilot pull against the Massive API, in the order the plan calls for.

    1  underlying bars      SPY 5-minute -> the session clock and the ATM strike
    2  reference chain      as_of the test date, same-day expiry -> the strike list
    3  quote bars, ATM      one contract, both construction methods, cross-checked
    4  quote bars, chain    every strike from step 2
    5  sanity pass          the five checks that gate writing a pricer

Run them one at a time and read the output between steps:

    python -m puller.pilot step1 --date 2025-10-15
    python -m puller.pilot step2
    python -m puller.pilot step3
    python -m puller.pilot step4
    python -m puller.pilot step5

Each step writes to data/pilot/<date>/ and the next step reads from there, so a
step can be re-run without repeating the one before it.

ON "QUOTE BARS"
---------------
Massive has no quote-bar endpoint. Options bars (/v2/aggs) are built from
trades, so a 0DTE wing that never prints has no bar at all -- exactly the
strikes whose mid we need. So we build quote bars from the NBBO tick feed
(/v3/quotes), two ways:

    asof    one call per bar edge asking for the last quote at or before it.
            78 calls per contract, flat cost regardless of how busy the name is.

    stream  pull every tick in the session and keep the last one in each bin.
            Exact, but an ATM 0DTE SPY contract can print millions of quotes.

They should agree bar for bar. Step 3 runs both on the ATM contract and diffs
them; that is the point of step 3. Step 4 then uses asof for the full chain.
"""

import argparse
import json
import re
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta

import pandas as pd

from .client import (
    MARKET_TZ, OPTION_QUOTE_CLOSE, SESSION_CLOSE, SESSION_OPEN,
    AuthError, Massive, MassiveError, et, ns, occ_ticker,
)

OCC = re.compile(r"^O:([A-Z]+)(\d{6})([CP])(\d{8})$")


class DateMismatch(RuntimeError):
    """An artifact carries a session date other than the one requested.

    Raised rather than warned. Two silent date bugs in one pilot -- zero-bid
    quotes counted as coverage, and resolve_date picking the lexicographically
    latest session -- both produced plausible numbers for the wrong day. Neither
    errored. Anything downstream of a date mismatch is poison, so it stops here.
    """


def assert_day(day, values, what):
    bad = sorted({str(v)[:10] for v in values if str(v)[:10] != str(day)})
    if bad:
        raise DateMismatch(f"{what}: requested session {day}, artifact carries "
                           f"{bad[:4]}{' ...' if len(bad) > 4 else ''} "
                           f"({len(bad)} distinct wrong date(s))")


def assert_tickers(day, tickers, what):
    """The expiry is encoded in the OCC symbol; check it against the session."""
    want = f"{pd.Timestamp(day):%y%m%d}"
    bad = sorted({t for t in tickers
                  if not (OCC.match(t) and OCC.match(t).group(2) == want)})
    if bad:
        raise DateMismatch(f"{what}: {len(bad)} ticker(s) not expiring {day}, "
                           f"e.g. {bad[:3]}")


UNDERLYING = "SPY"
BAR_MINUTES = 5
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "pilot")


# ---- paths ---------------------------------------------------------------

def daydir(day, *parts):
    p = os.path.join(DATA, str(day), *parts)
    os.makedirs(os.path.dirname(p) if os.path.splitext(p)[1] else p, exist_ok=True)
    return p


def state_path(day):
    return daydir(day, "state.json")


def save_state(day, **kw):
    p = state_path(day)
    cur = json.load(open(p)) if os.path.exists(p) else {}
    cur.update(kw)
    json.dump(cur, open(p, "w"), indent=2, default=str)
    return cur


def load_state(day):
    p = state_path(day)
    if not os.path.exists(p):
        sys.exit(f"no state for {day}; run step1 first")
    st = json.load(open(p))
    assert_day(day, [st.get("date")], f"state.json for {day}")
    return st


def resolve_date(args):
    """--date wins; otherwise the most recent session dir under data/pilot."""
    if args.date:
        return args.date
    if os.path.isdir(DATA):
        # the session most recently WORKED ON, not the latest calendar date --
        # pulling an earlier day after a later one must not silently resolve to
        # the later one and re-run the whole pilot on stale files.
        days = [d for d in os.listdir(DATA)
                if d[:2] == "20" and os.path.exists(os.path.join(DATA, d, "state.json"))]
        if days:
            return max(days, key=lambda d: os.path.getmtime(os.path.join(DATA, d, "state.json")))
    sys.exit("no --date given and no previous session found; pass --date YYYY-MM-DD")


def hr(title):
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


# ==========================================================================
# step 1 -- underlying bars
# ==========================================================================

def step1(m, args):
    day = args.date
    hr(f"STEP 1  {UNDERLYING} {BAR_MINUTES}-minute bars, {day}")

    rows = m.stock_bars(UNDERLYING, day, multiplier=BAR_MINUTES, timespan="minute")
    if not rows:
        sys.exit(f"no bars for {UNDERLYING} on {day}. Market holiday, or a date "
                 f"outside the plan's history window.")

    df = pd.DataFrame(rows)
    df["ts"] = [et(ts_ms=t) for t in df["t"]]
    df = df.sort_values("ts").reset_index(drop=True)

    # The regular session only. Massive returns pre/post bars for equities, and
    # a 04:00 bar would put the ATM strike on overnight liquidity.
    intraday = df[[SESSION_OPEN <= t.time() < SESSION_CLOSE for t in df["ts"]]].copy()
    if intraday.empty:
        sys.exit(f"bars came back for {day} but none inside {SESSION_OPEN}-{SESSION_CLOSE}")

    spot_open = float(intraday.iloc[0]["o"])
    spot_close = float(intraday.iloc[-1]["c"])

    out = intraday[["ts", "o", "h", "l", "c", "v", "vw", "n"]].rename(
        columns={"o": "open", "h": "high", "l": "low", "c": "close",
                 "v": "volume", "vw": "vwap", "n": "trades"})
    assert_day(day, intraday["ts"].dt.date, "step1 underlying bars")
    out.to_csv(daydir(day, "underlying_5m.csv"), index=False)

    print(f"  bars returned      {len(df)}  ({len(df) - len(intraday)} outside RTH, dropped)")
    print(f"  RTH bars           {len(intraday)}   expected 78")
    print(f"  first bar          {intraday.iloc[0]['ts']}")
    print(f"  last bar           {intraday.iloc[-1]['ts']}   <- window START, closes 5 min later")
    print(f"  open / close       {spot_open:.2f} / {spot_close:.2f}")
    print(f"  session range      {intraday['l'].min():.2f} - {intraday['h'].max():.2f}")

    gaps = intraday["ts"].diff().dropna()
    odd = gaps[gaps != timedelta(minutes=BAR_MINUTES)]
    print(f"  missing bars       {'none' if odd.empty else list(odd.index)}")

    # ATM off the open: the strike a hedger would be sitting on at 09:30.
    atm = round(spot_open)
    print(f"\n  ATM strike (round of open {spot_open:.2f})  ->  {atm}")
    print(f"  wrote {os.path.relpath(daydir(day, 'underlying_5m.csv'), ROOT)}")

    save_state(day, date=day, n_bars=len(intraday), spot_open=spot_open,
               spot_close=spot_close, atm_strike=atm,
               first_bar=str(intraday.iloc[0]["ts"]), last_bar=str(intraday.iloc[-1]["ts"]))
    return out


# ==========================================================================
# step 2 -- reference chain
# ==========================================================================

def step2(m, args):
    day = args.date
    st = load_state(day)
    hr(f"STEP 2  {UNDERLYING} contract index, as_of={day}, expiring {day}")

    rows, expired_flag = m.option_contracts(UNDERLYING, expiration_date=day, as_of=day)
    if not rows:
        sys.exit(f"no {UNDERLYING} contracts expiring {day}. Either {day} is not an "
                 f"expiry (SPY has daily expiries Mon-Fri, so check it is a session) "
                 f"or the plan's contract history does not reach back that far.")

    df = pd.DataFrame(rows)
    assert_day(day, df.expiration_date, "step2 chain expiration_date")
    assert_tickers(day, df.ticker, "step2 chain tickers")
    print(f"  expired={expired_flag} is the flag that returned data")
    print(f"  contracts          {len(df)}")
    print(f"  contract_type      {df['contract_type'].value_counts().to_dict()}")
    print(f"  exercise_style     {df['exercise_style'].value_counts().to_dict()}")
    if "shares_per_contract" in df:
        print(f"  shares/contract    {sorted(df['shares_per_contract'].unique().tolist())}")

    dupes = df[df.duplicated(["contract_type", "strike_price"], keep=False)]
    if len(dupes):
        print(f"  !! {len(dupes)} duplicate type/strike rows (adjusted contracts?)")

    calls = sorted(df[df.contract_type == "call"].strike_price.unique())
    puts = sorted(df[df.contract_type == "put"].strike_price.unique())
    both = sorted(set(calls) & set(puts))

    print(f"\n  call strikes       {len(calls)}   {min(calls):g} .. {max(calls):g}")
    print(f"  put strikes        {len(puts)}   {min(puts):g} .. {max(puts):g}")
    print(f"  paired strikes     {len(both)}   (both a call and a put -- parity needs these)")
    only_c = sorted(set(calls) - set(puts))
    only_p = sorted(set(puts) - set(calls))
    if only_c:
        print(f"  calls with no put  {only_c}")
    if only_p:
        print(f"  puts with no call  {only_p}")

    spacing = pd.Series(both).diff().dropna().round(4)
    print(f"\n  strike spacing     {spacing.value_counts().sort_index().to_dict()}")

    spot = st["spot_open"]
    print(f"  spot at open       {spot:.2f}")
    print(f"  moneyness span     {min(both) / spot - 1:+.1%} .. {max(both) / spot - 1:+.1%}")

    df.sort_values(["strike_price", "contract_type"]).to_csv(
        daydir(day, "chain.csv"), index=False)
    print(f"\n  wrote {os.path.relpath(daydir(day, 'chain.csv'), ROOT)}")

    save_state(day, n_contracts=len(df), n_paired_strikes=len(both),
               strike_min=float(min(both)), strike_max=float(max(both)),
               modal_spacing=float(spacing.mode().iloc[0]) if len(spacing) else None,
               expired_flag=bool(expired_flag))
    return df


# ==========================================================================
# quote bars -- the thing the API does not give us
# ==========================================================================

def bar_grid(day):
    """Bar STARTS for the RTH session, taken from step 1's file when it exists."""
    p = daydir(day, "underlying_5m.csv")
    if os.path.exists(p):
        ts = pd.read_csv(p, parse_dates=["ts"])["ts"]
        assert_day(day, ts.dt.date, "bar grid from underlying_5m.csv")
        return [t.to_pydatetime() for t in ts]
    d = date.fromisoformat(day)
    start = datetime.combine(d, SESSION_OPEN, tzinfo=MARKET_TZ)
    n = int(((SESSION_CLOSE.hour * 60 + SESSION_CLOSE.minute)
             - (SESSION_OPEN.hour * 60 + SESSION_OPEN.minute)) / BAR_MINUTES)
    return [start + timedelta(minutes=BAR_MINUTES * i) for i in range(n)]


def _row(bar_start, q, n_quotes=None):
    if q is None:
        return {"bar_start": bar_start, "bid": None, "ask": None, "bid_size": None,
                "ask_size": None, "mid": None, "spread": None, "n_quotes": n_quotes,
                "quote_ts": None, "lag_s": None}
    bid, ask = q.get("bid_price"), q.get("ask_price")
    qt = et(ts_ns=q["sip_timestamp"])
    edge = bar_start + timedelta(minutes=BAR_MINUTES)
    mid = (bid + ask) / 2 if bid is not None and ask is not None else None
    return {
        "bar_start": bar_start, "bid": bid, "ask": ask,
        "bid_size": q.get("bid_size"), "ask_size": q.get("ask_size"),
        "mid": mid, "spread": (ask - bid) if mid is not None else None,
        "n_quotes": n_quotes, "quote_ts": qt,
        "lag_s": round((edge - qt).total_seconds(), 3),
    }


def quote_bars_asof(m, ticker, day, grid):
    """One request per bar edge: the last NBBO at or before the edge."""
    floor = ns(day, SESSION_OPEN)
    rows = []
    for bs in grid:
        edge = bs + timedelta(minutes=BAR_MINUTES)
        q = m.last_quote_at_or_before(ticker, int(edge.timestamp() * 1e9), floor_ns=floor)
        rows.append(_row(bs, q))
    return pd.DataFrame(rows)


def quote_bars_stream(m, ticker, day, grid, max_pages=400):
    """Pull every tick in the session and keep the last one in each bin."""
    start = ns(day, SESSION_OPEN)
    end = int((grid[-1] + timedelta(minutes=BAR_MINUTES)).timestamp() * 1e9)
    edges = [int((bs + timedelta(minutes=BAR_MINUTES)).timestamp() * 1e9) for bs in grid]

    last = [None] * len(grid)
    count = [0] * len(grid)
    i, total = 0, 0
    for q in m.option_quotes(ticker, start, end, max_pages=max_pages):
        total += 1
        t = q["sip_timestamp"]
        while i < len(edges) and t > edges[i]:
            i += 1
        if i >= len(edges):
            break
        last[i], count[i] = q, count[i] + 1
    return pd.DataFrame([_row(bs, q, n) for bs, q, n in zip(grid, last, count)]), total


# ==========================================================================
# step 3 -- one contract, both ways
# ==========================================================================

def step3(m, args):
    day = args.date
    st = load_state(day)
    k = args.strike or st["atm_strike"]
    right = args.right
    ticker = occ_ticker(UNDERLYING, day, right, k)
    grid = bar_grid(day)

    hr(f"STEP 3  quote bars for ONE contract: {ticker}")
    print(f"  grid {len(grid)} bars, {grid[0].time()} .. {grid[-1].time()} (starts)")
    print(f"  options quote until {OPTION_QUOTE_CLOSE}; the grid stops at the equity close\n")

    t0 = time.time(); c0 = m.calls
    a = quote_bars_asof(m, ticker, day, grid)
    print(f"  asof    {m.calls - c0:4d} calls  {time.time() - t0:6.1f}s  "
          f"{a['mid'].notna().sum()}/{len(a)} bars priced")

    t0 = time.time(); c0 = m.calls
    try:
        s, nticks = quote_bars_stream(m, ticker, day, grid, max_pages=args.max_pages)
        print(f"  stream  {m.calls - c0:4d} calls  {time.time() - t0:6.1f}s  "
              f"{s['mid'].notna().sum()}/{len(s)} bars priced  {nticks:,} ticks")
    except MassiveError as exc:
        print(f"  stream  FAILED: {exc}")
        s, nticks = None, 0

    if s is not None:
        j = a.merge(s, on="bar_start", suffixes=("_asof", "_stream"))
        for col in ("bid", "ask"):
            d = (j[f"{col}_asof"] - j[f"{col}_stream"]).abs()
            bad = int((d > 1e-9).sum())
            print(f"  {col:>4} agreement   {len(j) - bad}/{len(j)} bars identical"
                  + (f"   max diff {d.max():.4f}" if bad else ""))
        both_na = int((j["mid_asof"].isna() != j["mid_stream"].isna()).sum())
        print(f"  coverage mismatch  {both_na} bars")
        print(f"\n  ticks per bar      median {s['n_quotes'].median():.0f}  "
              f"max {s['n_quotes'].max():,}  total {nticks:,}")
        print(f"  -> a full-chain stream pull is roughly "
              f"{nticks * st.get('n_contracts', 0):,.0f} ticks; asof is "
              f"{len(grid) * st.get('n_contracts', 0):,} calls flat")

    print(f"\n  quote lag behind bar edge (s): median {a['lag_s'].median():.2f}  "
          f"p95 {a['lag_s'].quantile(0.95):.2f}  max {a['lag_s'].max():.2f}")
    print(f"  first bar  {a.iloc[0]['bar_start'].time()}  bid {a.iloc[0]['bid']}  ask {a.iloc[0]['ask']}")
    print(f"  last  bar  {a.iloc[-1]['bar_start'].time()}  bid {a.iloc[-1]['bid']}  ask {a.iloc[-1]['ask']}")

    a.to_csv(daydir(day, "quotes", f"{ticker.replace(':', '_')}.csv"), index=False)
    if s is not None:
        s.to_csv(daydir(day, f"step3_stream_{ticker.replace(':', '_')}.csv"), index=False)
    print(f"\n  wrote data/pilot/{day}/quotes/{ticker.replace(':', '_')}.csv")
    save_state(day, step3_ticker=ticker, step3_ticks=nticks)
    return a


# ==========================================================================
# step 4 -- full chain
# ==========================================================================

def _one(m, ticker, day, grid, path, refresh, mode, max_pages):
    if os.path.exists(path) and not refresh:
        return ticker, "cached", 0
    if mode == "stream":
        df, nticks = quote_bars_stream(m, ticker, day, grid, max_pages=max_pages)
    else:
        df, nticks = quote_bars_asof(m, ticker, day, grid), 0
    df.to_csv(path, index=False)
    return ticker, "pulled", nticks


def step4(m, args):
    day = args.date
    st = load_state(day)
    chain = pd.read_csv(daydir(day, "chain.csv"))
    grid = bar_grid(day)

    if args.max_moneyness:
        lo = st["spot_open"] * (1 - args.max_moneyness)
        hi = st["spot_open"] * (1 + args.max_moneyness)
        chain = chain[chain.strike_price.between(lo, hi)]

    assert_day(day, chain.expiration_date, "step4 chain")
    tickers = sorted(chain.ticker.unique())
    assert_tickers(day, tickers, "step4 tickers")
    hr(f"STEP 4  quote bars for the full chain: {len(tickers)} contracts x {len(grid)} bars")
    if args.mode == "stream":
        print(f"  mode=stream: one paged tick pull per contract, {args.workers} workers")
        print(f"  (step 3 showed stream costs fewer calls than asof even at the ATM)\n")
    else:
        print(f"  mode=asof: {len(tickers) * len(grid):,} requests at {args.workers} workers\n")

    qdir = daydir(day, "quotes")
    todo = [(t, os.path.join(qdir, f"{t.replace(':', '_')}.csv")) for t in tickers]
    done, failed, ticks, t0 = 0, {}, 0, time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_one, m, t, day, grid, p, args.refresh,
                          args.mode, args.max_pages): t for t, p in todo}
        for fut in as_completed(futs):
            t = futs[fut]
            try:
                ticks += fut.result()[2]
            except Exception as exc:
                failed[t] = str(exc)[:120]
            done += 1
            if done % 10 == 0 or done == len(todo):
                el = time.time() - t0
                print(f"    {done:4d}/{len(todo)}  {el:6.1f}s  {m.calls:6d} calls  "
                      f"{ticks:12,d} ticks  {len(failed)} failed")

    if failed:
        print(f"\n  FAILED {len(failed)}:")
        for t, e in list(failed.items())[:10]:
            print(f"    {t}: {e}")
    save_state(day, n_pulled=len(todo) - len(failed), n_failed=len(failed),
               step4_mode=args.mode, step4_ticks=ticks, step4_calls=m.calls)
    if args.mode == "stream":
        print(f"\n  {ticks:,} ticks in {m.calls:,} calls")
    print(f"\n  {len(todo) - len(failed)}/{len(todo)} contracts on disk in data/pilot/{day}/quotes/")


# ==========================================================================
# step 5 -- sanity pass
# ==========================================================================

def step5(m, args):
    day = args.date
    st = load_state(day)
    hr(f"STEP 5  sanity pass, {day}")

    und = pd.read_csv(daydir(day, "underlying_5m.csv"), parse_dates=["ts"])
    chain = pd.read_csv(daydir(day, "chain.csv")).set_index("ticker")
    qdir = daydir(day, "quotes")
    files = sorted(f for f in os.listdir(qdir) if f.endswith(".csv"))
    if not files:
        sys.exit("no quote files; run step4 first")

    assert_day(day, [st.get("date")], "step5 state.json")
    assert_day(day, und.ts.dt.date, "step5 underlying_5m.csv")
    assert_day(day, chain.expiration_date, "step5 chain.csv")
    assert_tickers(day, chain.index, "step5 chain.csv tickers")

    frames = {}
    for f in files:
        t = f[:-4].replace("O_", "O:")
        d = pd.read_csv(os.path.join(qdir, f), parse_dates=["bar_start", "quote_ts"])
        assert_day(day, d.bar_start.dt.date, f"step5 quotes/{f}")
        if t in chain.index:
            assert_tickers(day, [t], f"step5 quotes/{f}")
            frames[t] = d

    n_bars = len(und)

    # --- 1. last bar timestamp
    print("\n[1] LAST BAR TIMESTAMP")
    print(f"    underlying last bar start   {und.ts.iloc[-1]}")
    print(f"    underlying last bar closes  {und.ts.iloc[-1] + timedelta(minutes=BAR_MINUTES)}")
    lasts = pd.Series([d.dropna(subset=["mid"]).bar_start.max() for d in frames.values()
                       if d.mid.notna().any()])
    print(f"    option last priced bar      mode {lasts.mode().iloc[0]}  "
          f"({(lasts == lasts.mode().iloc[0]).sum()}/{len(lasts)} contracts)")
    early = lasts[lasts < lasts.mode().iloc[0]]
    print(f"    contracts ending early      {len(early)}")

    # --- 2. bar count per contract
    print("\n[2] BAR COUNT PER CONTRACT")
    cov = pd.Series({t: int(d.mid.notna().sum()) for t, d in frames.items()})
    two = pd.Series({t: int(((d.bid > 0) & (d.ask > d.bid)).sum()) for t, d in frames.items()})
    print(f"    grid                        {n_bars} bars")
    print(f"    full coverage (any quote)   {(cov == n_bars).sum()}/{len(cov)} contracts")
    print(f"    full coverage (bid > 0)     {(two == n_bars).sum()}/{len(two)} contracts")
    print(f"    median / min  any quote     {cov.median():.0f} / {cov.min()}")
    print(f"    median / min  bid > 0       {two.median():.0f} / {two.min()}")
    for lo, hi in [(0, 0), (1, 19), (20, 59), (60, n_bars - 1), (n_bars, n_bars)]:
        c = int(cov.between(lo, hi).sum())
        if c:
            print(f"      {lo:3d}-{hi:3d} bars  {c:4d} contracts")

    # --- 3. zero or crossed spreads
    print("\n[3] ZERO OR CROSSED SPREADS")
    allq = pd.concat([d.assign(ticker=t) for t, d in frames.items()], ignore_index=True)
    q = allq.dropna(subset=["bid", "ask"])
    crossed = q[q.ask < q.bid]
    zero = q[q.ask == q.bid]
    zbid = q[q.bid == 0]
    print(f"    priced bars                 {len(q):,} of {len(allq):,}")
    print(f"    crossed (ask < bid)         {len(crossed):,}  ({len(crossed) / max(len(q), 1):.3%})")
    print(f"    locked   (ask == bid)       {len(zero):,}  ({len(zero) / max(len(q), 1):.3%})")
    print(f"    bid == 0                    {len(zbid):,}  ({len(zbid) / max(len(q), 1):.3%})"
          f"   <- no two-sided mid, wings")
    wide = q[q.spread > 0.5 * ((q.bid + q.ask) / 2)]
    print(f"    spread > 50% of mid         {len(wide):,}  ({len(wide) / max(len(q), 1):.3%})")
    print(f"    FILTER RULE: keep bid > 0 and ask > bid  -> "
          f"{len(q) - len(crossed) - len(zero) - len(zbid):,} bars survive")

    # --- 4. strikes with no quotes at all
    #
    # "Has a quote" and "has a mid" are different questions on 0DTE. A deep wing
    # quotes 0.00 bid / 0.01 ask all session: present, two-sided-looking, and
    # useless. The strike filter has to key off bid > 0, not off the quote count.
    print("\n[4] STRIKES WITH NO QUOTES AT ALL")
    s0 = st["spot_open"]
    a = allq.merge(chain[["strike_price", "contract_type"]], left_on="ticker",
                   right_index=True, how="inner")
    a["ok"] = (a.bid > 0) & (a.ask > a.bid)

    per = a.groupby("ticker").agg(quoted=("bid", "count"), twosided=("ok", "sum"))
    print(f"    never quoted at all         {int((per.quoted == 0).sum())}/{len(per)}")
    print(f"    quoted but never bid > 0   {int(((per.quoted > 0) & (per.twosided == 0)).sum())}"
          f"/{len(per)}   <- present, unusable")

    # the OTM side is what a variance strip integrates over
    otm = a[((a.contract_type == "call") & (a.strike_price >= s0))
            | ((a.contract_type == "put") & (a.strike_price < s0))]
    g = otm.groupby("strike_price").ok.mean()
    good = g[g >= 0.95]
    print(f"    OTM strikes with >=95% two-sided coverage: {len(good)}/{len(g)}")
    if len(good):
        print(f"      band {good.index.min():g} .. {good.index.max():g}"
              f"   ({good.index.min() / s0 - 1:+.1%} .. {good.index.max() / s0 - 1:+.1%})")

    w = otm[otm.ok].groupby("bar_start").strike_price.agg(["min", "max", "count"])
    if len(w):
        worst = w["count"].idxmin()
        print(f"    OTM strikes per bar        median {w['count'].median():.0f}  "
              f"min {w['count'].min()} (at {worst.time()})  max {w['count'].max()}")
        print(f"    strip bounds               median {w['min'].median():g}..{w['max'].median():g}"
              f"   ({w['min'].median() / s0 - 1:+.1%} .. {w['max'].median() / s0 - 1:+.1%})")
        print(f"    worst bar                  {w.loc[worst, 'min']:g}..{w.loc[worst, 'max']:g}"
              f"   ({w.loc[worst, 'min'] / s0 - 1:+.1%} .. {w.loc[worst, 'max'] / s0 - 1:+.1%})")
        print(f"    STRIKE FILTER: the strip is truncated at roughly "
              f"{w['min'].median() / s0 - 1:+.0%}/{w['max'].median() / s0 - 1:+.0%}, "
              f"not at the chain's edge")

    # --- 5. put-call parity at the ATM
    print("\n[5] PUT-CALL PARITY AT THE ATM")
    k = st["atm_strike"]
    ct = occ_ticker(UNDERLYING, day, "C", k)
    pt = occ_ticker(UNDERLYING, day, "P", k)
    if ct in frames and pt in frames:
        c = frames[ct][["bar_start", "mid"]].rename(columns={"mid": "call"})
        p = frames[pt][["bar_start", "mid"]].rename(columns={"mid": "put"})
        u = und[["ts", "close"]].rename(columns={"ts": "bar_start", "close": "S"})
        j = c.merge(p, on="bar_start").merge(u, on="bar_start").dropna()
        # 0DTE: discounting K over <6.5h moves parity by well under a tick, so
        # any residual here is a sign or scaling error, not a rates effect.
        j["resid"] = j.call - j.put - (j.S - k)
        print(f"    K = {k}, {len(j)} bars with both legs")
        print(f"    C - P - (S - K):  median {j.resid.median():+.4f}  "
              f"mean {j.resid.mean():+.4f}  sd {j.resid.std():.4f}")
        print(f"                      p5 {j.resid.quantile(.05):+.4f}  "
              f"p95 {j.resid.quantile(.95):+.4f}  max|.| {j.resid.abs().max():.4f}")
        tick = 0.01
        print(f"    within 1 tick               {(j.resid.abs() <= tick).mean():.1%}")
        print(f"    within 5 ticks              {(j.resid.abs() <= 5 * tick).mean():.1%}")
        if j.resid.abs().median() > 0.10:
            print("    !! median residual over a dime -- check sign, scaling, or an ex-div date")
        else:
            print("    OK: no sign or scaling error visible at the ATM")
    else:
        print(f"    missing a leg at K={k}: call={ct in frames} put={pt in frames}")

    assert_day(day, allq.bar_start.dt.date, "step5 quote_bars_all.csv")
    assert_tickers(day, allq.ticker.unique(), "step5 quote_bars_all.csv")
    allq.to_csv(daydir(day, "quote_bars_all.csv"), index=False)
    print(f"\n  wrote data/pilot/{day}/quote_bars_all.csv  ({len(allq):,} rows)")


# ==========================================================================

STEPS = {"step1": step1, "step2": step2, "step3": step3, "step4": step4, "step5": step5}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=list(STEPS) + ["all"])
    ap.add_argument("--date", help="session to pull, YYYY-MM-DD")
    ap.add_argument("--strike", type=float, help="step3: override the ATM strike")
    ap.add_argument("--right", default="C", choices=["C", "P"], help="step3 leg")
    ap.add_argument("--max-pages", type=int, default=400,
                    help="stream page cap per contract (50k quotes per page)")
    ap.add_argument("--mode", default="stream", choices=["stream", "asof"],
                    help="step4 quote-bar construction; step 3 compares them")
    ap.add_argument("--workers", type=int, default=8, help="step4 concurrency")
    ap.add_argument("--refresh", action="store_true", help="step4: re-pull cached contracts")
    ap.add_argument("--max-moneyness", type=float, default=None,
                    help="step4: keep strikes within +/- this fraction of spot")
    ap.add_argument("--rpm", type=float, default=None, help="client-side requests/min cap")
    args = ap.parse_args(argv)
    args.date = resolve_date(args)

    try:
        m = Massive(rpm=args.rpm)
    except AuthError as exc:
        sys.exit(f"\n{exc}\n")

    steps = list(STEPS) if args.step == "all" else [args.step]
    for s in steps:
        STEPS[s](m, args)
    print(f"\n[{m.calls} API calls this run]")


if __name__ == "__main__":
    main()
