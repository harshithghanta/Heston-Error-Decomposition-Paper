"""Implied-variance strip, computed three ways to bound the zero-bid problem.

The strip is the usual Cboe construction for a single term,

    sigma^2 = (2/T) * sum_i (dK_i / K_i^2) * e^{rT} * Q(K_i)  -  (1/T) * (F/K0 - 1)^2

with Q the OTM option price at K_i and both legs averaged at K0. What is not
usual is that on 0DTE the wings quote 0.00 bid / 0.01 ask for most of the
session, so Q is undefined exactly where the integrand still has weight. Rather
than pick a convention and report a point estimate, we compute three:

    lower     every OTM strike in the chain; zero-bid strikes valued at 0.00
    upper     every OTM strike in the chain; zero-bid strikes at ask - $0.01
    ask_full  as upper but at the full ask; declared diagnostic, strictly looser
    cboe      Cboe's own strike selection, mids only  <- the headline estimator

The convention was fixed in data/pilot/PRECOMMIT_convention.md before any
57-day number existed. `upper` uses ask - $0.01 because at the offer there is a
seller and no buyer, so fair value is at most one tick below it; for the modal
0.00 x 0.01 wing quote that correctly collapses the band to zero width.

lower and upper are not estimates. A zero-bid option is worth somewhere in
[0, ask] by definition of a quote, so the true strip lies between them and the
gap is a hard bound on how much the truncation can matter. `cboe` is the
convention a replication would actually use, and it should sit inside.

CBOE STRIKE SELECTION -- Volatility Index Methodology: Cboe 1-Day Volatility
Index (VIX1D), section "Strike Selection", p. 12. Verified against the current
paper 2026-08-26:

    "First, for each term, select out-of-the-money put options with strike
     prices less than K0. Start with the put strike immediately lower than K0
     and move to successively lower strike prices. Exclude any put option that
     has a bid price equal to zero (i.e., no bid). ... once two puts with
     consecutive strike prices are found to have zero bid prices, exclude the
     observed put(s) and consider no puts with lower strikes for inclusion."

and symmetrically upward for calls. The subtlety worth stating, because it is
easy to implement wrongly: an ISOLATED zero bid is skipped but does NOT stop
the scan. Cboe's own worked example excludes the 3880 put for a zero bid and
still includes the 3885 put below it. Only two CONSECUTIVE zero bids terminate,
and the two strikes that trigger it are themselves excluded.
"""

import numpy as np

# Cboe counts time in business minutes, not calendar time.
MIN_PER_YEAR = 252 * 6.75 * 60          # 102,060
CBOE_DOC = ("Volatility Index Methodology: Cboe 1-Day Volatility Index, "
            "'Strike Selection', p.12 (retrieved 2026-08-26)")


# Cboe's 2025-02-10 amendment: a zero ASK excludes a strike as well as a zero bid.
# Kept in step with massive/fast.py, whose ZERO_ASK_EXCLUDES carries the full note and
# the 2025-04-09 13:20 case that motivated it. check_fast() compares the two, so the
# two implementations must always be changed together.
ZERO_ASK_EXCLUDES = True


def forward(df, T, r=0.0):
    """F = K + e^{rT}(C - P) at the strike where |C - P| is smallest.

    Only two-sided strikes are eligible. A crossed quote (bid > 0, ask = 0.00) has a
    finite but meaningless mid that sits far below the true call-put gap, so it wins
    the argmin outright and returns a forward tens of points away from spot.
    """
    ok = df.call_mid.notna() & df.put_mid.notna()
    if ZERO_ASK_EXCLUDES:
        ok = ok & (df.call_ask > 0) & (df.put_ask > 0)
    if not ok.any():
        return None, None
    d = df[ok]
    k = d.K.iloc[(d.call_mid - d.put_mid).abs().to_numpy().argmin()]
    row = d[d.K == k].iloc[0]
    F = k + np.exp(r * T) * (row.call_mid - row.put_mid)
    below = df.K[df.K <= F]
    return (F, below.max()) if len(below) else (F, None)


def _cboe_side(df, K0, side):
    """Cboe scan outward from K0. Returns the strikes to include."""
    if side == "put":
        d = df[df.K < K0].sort_values("K", ascending=False)
        bid, ask, mid = "put_bid", "put_ask", "put_mid"
    else:
        d = df[df.K > K0].sort_values("K", ascending=True)
        bid, ask, mid = "call_bid", "call_ask", "call_mid"

    keep, zeros = [], 0
    for _, row in d.iterrows():
        b, a = row[bid], row[ask]
        no_bid = b is None or np.isnan(b) or b <= 0
        no_ask = ZERO_ASK_EXCLUDES and (a is None or np.isnan(a) or a <= 0)
        if no_bid or no_ask:
            zeros += 1
            if zeros >= 2:          # two consecutive: stop, and drop both
                break
            continue                # isolated zero: skip, keep scanning
        zeros = 0
        if not np.isnan(row[mid]):
            keep.append(row.K)
    return keep


def _dk(ks):
    """Cboe dK: half the gap to the neighbours, one-sided at the ends."""
    ks = np.asarray(sorted(ks), float)
    if len(ks) == 1:
        return ks, np.array([1.0])
    d = np.empty_like(ks)
    d[1:-1] = (ks[2:] - ks[:-2]) / 2
    d[0], d[-1] = ks[1] - ks[0], ks[-1] - ks[-2]
    return ks, d


def strip(df, T, variant, r=0.0):
    """One variance number for one bar.

    df: one row per strike, columns K, call_bid/ask/mid, put_bid/ask/mid.
    Returns dict with sigma2, the strike range used, and the count.
    """
    if T <= 0:
        return None
    F, K0 = forward(df, T, r)
    if K0 is None:
        return None

    if variant == "cboe":
        put_ks = _cboe_side(df, K0, "put")
        call_ks = _cboe_side(df, K0, "call")
        q = {k: df.loc[df.K == k, "put_mid"].iloc[0] for k in put_ks}
        q.update({k: df.loc[df.K == k, "call_mid"].iloc[0] for k in call_ks})
    else:
        q = {}
        for _, row in df.iterrows():
            if row.K == K0:
                continue
            leg = "put" if row.K < K0 else "call"
            b, a, m = row[f"{leg}_bid"], row[f"{leg}_ask"], row[f"{leg}_mid"]
            if b is not None and not np.isnan(b) and b > 0:
                v = m                                   # live two-sided quote
            elif a is not None and not np.isnan(a):
                if variant == "lower":
                    v = 0.0
                elif variant == "ask_full":
                    v = a
                else:                                    # "upper": one tick under the offer
                    v = max(a - 0.01, 0.0)
            else:
                continue                                 # no quote at all
            if not np.isnan(v):
                q[row.K] = v

    # K0 itself: both legs, averaged. Present in every variant.
    z = df[df.K == K0]
    if len(z) and not np.isnan(z.call_mid.iloc[0]) and not np.isnan(z.put_mid.iloc[0]):
        q[K0] = (z.call_mid.iloc[0] + z.put_mid.iloc[0]) / 2
    if len(q) < 3:
        return None

    ks, dk = _dk(list(q.keys()))
    qs = np.array([q[k] for k in ks], float)
    s2 = (2 / T) * np.sum(dk / ks**2 * np.exp(r * T) * qs) - (1 / T) * (F / K0 - 1) ** 2
    return {"sigma2": s2, "vol": 100 * np.sqrt(max(s2, 0)), "n_strikes": len(ks),
            "k_lo": ks.min(), "k_hi": ks.max(), "F": F, "K0": K0}


def cross_section(bar_df):
    """Stack a bar's contracts into one row per strike with both legs."""
    out = bar_df.pivot_table(index="strike_price", columns="contract_type",
                             values=["bid", "ask", "mid"], aggfunc="first")
    out.columns = [f"{t}_{v}" for v, t in out.columns]
    out = out.reset_index().rename(columns={"strike_price": "K"})
    for c in ("call_bid", "call_ask", "call_mid", "put_bid", "put_ask", "put_mid"):
        if c not in out:
            out[c] = np.nan
    return out.sort_values("K")
