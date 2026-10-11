import os
import sys
import numpy as np
import pandas as pd

from settings import HALF_DAYS

ROOTS = {'spy': 'pilot1m', 'spx': 'spx1m', 'qqq': 'qqq1m', 'ndx': 'ndx1m'}

def load_day(path):
    df = pd.read_parquet(path)
    df = df[df.bar < 390]
    Ks = np.sort(df.strike.unique()).astype(np.float64)
    col = {k: j for j, k in enumerate(Ks)}
    grids = {}
    for name in ['C_bid', 'C_ask', 'C_mid', 'P_bid', 'P_ask', 'P_mid']:
        grids[name] = np.full((390, len(Ks)), np.nan, dtype = np.float64)
    for right in ['C', 'P']:
        s = df[df.right == right]
        rows = s.bar.to_numpy()
        cols = np.array([col[float(k)] for k in s.strike], dtype = int)
        grids[right + '_bid'][rows, cols] = s.bid.to_numpy()
        grids[right + '_ask'][rows, cols] = s.ask.to_numpy()
        grids[right + '_mid'][rows, cols] = (s.bid.to_numpy() + s.ask.to_numpy()) / np.float32(2)
    return Ks, grids


def carry(bid, ask, mid, max_age = 2):
    bid, ask, mid = bid.copy(), ask.copy(), mid.copy()
    present = np.isfinite(bid) & np.isfinite(ask)
    last = np.full(bid.shape[1], -999)
    for i in range(bid.shape[0]):
        has = present[i]
        last[has] = i
        fill = ~has & (i - last <= max_age) & (last >= 0)
        src = last[fill]
        cols = np.where(fill)[0]
        bid[i, cols] = bid[src, cols]
        ask[i, cols] = ask[src, cols]
        mid[i, cols] = mid[src, cols]
    return bid, ask, mid


def forward(Ks, cm, pm, ca, pa):
    ok = np.isfinite(cm) & np.isfinite(pm) & (ca > 0) & (pa > 0)
    if not ok.any():
        return None, None
    gap = np.where(ok, np.abs(cm - pm), np.inf)
    j = np.argmin(gap)
    F = Ks[j] + (cm[j] - pm[j])
    below = Ks[Ks <= F]
    if len(below) == 0:
        return None, None
    return F, below.max()


def bad(b, a):
    return not (np.isfinite(b) and b > 0 and np.isfinite(a) and a > 0)

def pick(Ks, k0, bid, ask, mid, direction):
    keep = []
    j = k0 + direction
    misses = 0
    while 0 <= j < len(Ks):
        if bad(bid[j], ask[j]):
            misses += 1
            if misses == 2:
                break
        else:
            misses = 0
            if np.isfinite(mid[j]):
                keep.append((Ks[j], mid[j]))
        j += direction
    return keep


def strip(Ks, cb, ca, cm, pb, pa, pm, T):
    F, K0 = forward(Ks, cm, pm, ca, pa)
    if F is None:
        return None
    k0 = int(np.where(Ks == K0)[0][0])
    puts = pick(Ks, k0, pb, pa, pm, -1)
    calls = pick(Ks, k0, cb, ca, cm, 1)
    rows = puts + calls
    if np.isfinite(cm[k0]) and np.isfinite(pm[k0]):
        rows.append((K0, (cm[k0] + pm[k0]) / 2))
    if len(rows) < 3:
        return None
    rows.sort()
    K = np.array([r[0] for r in rows])
    Q = np.array([r[1] for r in rows], dtype = float)
    dK = np.empty(len(K))
    dK[1:-1] = (K[2:] - K[:-2]) / 2
    dK[0] = K[1] - K[0]
    dK[-1] = K[-1] - K[-2]
    var = (2 / T) * np.sum(dK / K ** 2 * Q) - (1 / T) * (F / K0 - 1) ** 2
    return F, 100 * np.sqrt(max(var, 0)), len(K)


def parity_price(path):
    df = pd.read_parquet(path)
    df = df[(df.bid > 0) & (df.ask > 0)]
    df = df.assign(mid = (df.bid + df.ask) * np.float32(0.5))
    C = df[df.right == 'C'][['bar', 'strike', 'mid']]
    P = df[df.right == 'P'][['bar', 'strike', 'mid']]
    both = C.merge(P, on = ['bar', 'strike'], suffixes = ('_c', '_p'))
    out = {}
    for bar, g in both.groupby('bar'):
        if bar >= 389 or len(g) < 2:
            continue
        T = (389 - bar) / 102060
        diff = g.mid_c - g.mid_p
        atm = g.strike.to_numpy()[np.argmin(np.abs(diff.to_numpy()))]
        near = np.argsort(np.abs(g.strike.to_numpy() - atm))[:11]
        out[bar] = float(np.median(g.strike.iloc[near] + np.exp(0.045 * T) * diff.iloc[near]))
    return out


def screen(strip_rows, parity):
    bars = sorted(strip_rows)
    F = pd.Series({b: strip_rows[b][0] for b in bars}, dtype = float)
    vol = pd.Series({b: strip_rows[b][1] for b in bars}, dtype = float)
    flagged = []
    for b in bars:
        if b <= 380 and b in parity:
            gap_bp = 1e4 * (F[b] / parity[b] - 1)
            if abs(gap_bp) > 10:
                flagged.append(b)
    F[flagged] = np.nan
    vol[flagged] = np.nan
    F = F.interpolate(method = 'index', limit_area = 'inside')
    vol = vol.interpolate(method = 'index', limit_area = 'inside')
    return F, vol, flagged


def day_path(sym, day):
    base = 'data_repull_wide' if day in ['2025-04-08', '2025-04-09'] else 'data'
    return f'{base}/{ROOTS[sym]}/{day}/bars.parquet'

def day_panel(sym, day, step):
    path = day_path(sym, day)
    Ks, g = load_day(path)
    cb, ca, cm = carry(g['C_bid'], g['C_ask'], g['C_mid'])
    pb, pa, pm = carry(g['P_bid'], g['P_ask'], g['P_mid'])
    rows = {}
    for i in range(389):
        r = strip(Ks, cb[i], ca[i], cm[i], pb[i], pa[i], pm[i], (389 - i) / 102060)
        if r:
            rows[i] = r
    if 0 not in rows:
        return None
    raw = np.sort(pd.read_parquet(path).strike.unique())
    strike = float(raw[np.argmin(np.abs(raw - float(rows[0][0])))])
    F, vol, flagged = screen(rows, parity_price(path))
    half = day in HALF_DAYS
    last = 200 if half else 380
    out = []
    for b in range(0, last + 1, step):
        if b not in F.index or not (F[b] > 0) or not (vol[b] > 0):
            return None
        v = vol[b] * np.sqrt((389 - b) / (209 - b)) if half else vol[b]
        stamp = pd.Timestamp(day) + pd.Timedelta(hours = 9, minutes = 31 + int(b))
        out.append((stamp.strftime('%Y-%m-%d %H:%M:%S'), F[b], v, strike))
    return out


def build_panel(sym, step, out_path):
    root = f'data/{ROOTS[sym]}'
    days = sorted(d for d in os.listdir(root) if d.startswith('20'))
    rows = []
    kept = 0
    for day in days:
        if not os.path.exists(day_path(sym, day)):
            continue
        if sym == 'spy' and step == 5:
            r = spy_half_day(day) if day in HALF_DAYS else spy_main_day(day)
        else:
            r = day_panel(sym, day, step)
        if r is None:
            continue
        rows += r
        kept += 1
    df = pd.DataFrame(rows, columns = ['timestamp', 'und_close', 'vix1d_close', 'strike'])
    df.to_csv(out_path, index = False)
    print(sym, step, 'days kept:', kept)

MARKS = list(range(0, 381, 5))


def longest_gap(have):
    longest = run = 0
    prev = None
    for b in range(381):
        if b in have:
            continue
        run = run + 1 if prev == b - 1 else 1
        longest = max(longest, run)
        prev = b
    return longest

def spy_vols(vol, have):
    v = np.full(len(MARKS), np.nan)
    for i, m in enumerate(MARKS):
        if m in have:
            v[i] = (vol[m] / 100) ** 2
            continue
        for k in (-1, 1, -2, 2):
            if m + k in have:
                v[i] = (vol[m + k] / 100) ** 2
                break
    missing = [i for i in range(len(MARKS)) if np.isnan(v[i])]
    if len(missing) > 5:
        return None
    if missing:
        good = [i for i in range(len(MARKS)) if not np.isnan(v[i])]
        v[missing] = np.interp(missing, good, v[good])
    return 100 * np.sqrt(v)


def spy_prices(par, day):
    s = np.array([par.get(m, np.nan) for m in MARKS], dtype = float)
    for i, m in enumerate(MARKS):
        if not (s[i] > 0):
            for k in (-1, 1, -2, 2):
                x = par.get(m + k, np.nan)
                if x > 0:
                    s[i] = x
                    break
    still = [i for i in range(len(MARKS)) if not (s[i] > 0)]
    obs_path = day_path('spy', day).replace('bars.parquet', 'underlying.parquet')
    if still and os.path.exists(obs_path):
        ob = pd.read_parquet(obs_path)
        ts = pd.to_datetime(ob.ts)
        if ts.dt.tz is not None:
            ts = ts.dt.tz_convert('America/New_York')
        minute = ts.dt.hour * 60 + ts.dt.minute - (9 * 60 + 31)
        obs = pd.Series(ob.close.to_numpy(float), index = minute.to_numpy())
        obs = obs[obs > 0]
        good = [i for i in range(len(MARKS)) if i not in still and np.isfinite(obs.get(MARKS[i], np.nan))]
        for i in still:
            lo = max([j for j in good if j < i], default = None)
            hi = min([j for j in good if j > i], default = None)
            ratios = [s[j] / obs[MARKS[j]] for j in (lo, hi) if j is not None]
            if ratios and np.isfinite(obs.get(MARKS[i], np.nan)):
                s[i] = obs[MARKS[i]] * np.mean(ratios)
    still = [i for i in range(len(MARKS)) if not (s[i] > 0)]
    if still:
        if len(still) > 5:
            return None
        good = [i for i in range(len(MARKS)) if i not in still]
        s[still] = np.exp(np.interp(still, good, np.log(s[good])))
    return s

def spy_main_day(day):
    path = day_path('spy', day)
    if not os.path.exists(f'data/pilot1m/{day}/strip.csv'):
        return None
    Ks, g = load_day(path)
    cb, ca, cm = carry(g['C_bid'], g['C_ask'], g['C_mid'])
    pb, pa, pm = carry(g['P_bid'], g['P_ask'], g['P_mid'])
    rows = {}
    for i in range(389):
        r = strip(Ks, cb[i], ca[i], cm[i], pb[i], pa[i], pm[i], (389 - i) / 102060)
        if r:
            rows[i] = r
    have = set(rows)
    if not rows or longest_gap(have) > 30:
        return None
    par = parity_price(path)
    F, vol, flagged = screen(rows, par)
    v = spy_vols(vol, have)
    s = spy_prices(par, day)
    if v is None or s is None:
        return None
    b0 = next((b for b in (0, 1, 2) if b in rows), None)
    if b0 is None:
        return None
    raw = pd.read_parquet(path).strike.unique()
    strike = float(raw[np.argmin(np.abs(raw - float(rows[b0][0])))])
    out = []
    for i, m in enumerate(MARKS):
        stamp = pd.Timestamp(day) + pd.Timedelta(hours = 9, minutes = 31 + m)
        out.append((stamp.strftime('%Y-%m-%d %H:%M:%S'), s[i], v[i], strike))
    return out


def spy_half_day(day):
    path = day_path('spy', day)
    Ks, g = load_day(path)
    cb, ca, cm = carry(g['C_bid'], g['C_ask'], g['C_mid'])
    pb, pa, pm = carry(g['P_bid'], g['P_ask'], g['P_mid'])
    rows = {}
    for i in range(389):
        r = strip(Ks, cb[i], ca[i], cm[i], pb[i], pa[i], pm[i], (389 - i) / 102060)
        if r:
            rows[i] = r
    par = parity_price(path)
    chain = np.sort(pd.read_parquet(f'data/pilot1m/{day}/chain.parquet').strike_price.unique())
    strike = float(chain[np.argmin(np.abs(chain - par[0]))])
    out = []
    last_vol = last_px = None
    for b in range(0, 201, 5):
        vol = rows[b][1] if b in rows else np.nan
        px = par.get(b, np.nan)
        vol = vol if np.isfinite(vol) else last_vol
        px = px if np.isfinite(px) else last_px
        last_vol, last_px = vol, px
        v13 = vol * np.sqrt((389 - b) / (209 - b))
        stamp = pd.Timestamp(day) + pd.Timedelta(hours = 9, minutes = 31 + b)
        out.append((stamp.strftime('%Y-%m-%d %H:%M:%S'), px, v13, strike))
    return out


if __name__ == '__main__':
    sym, step = sys.argv[1], int(sys.argv[2])
    build_panel(sym, step, f'results_v2/panels/{sym}{step}m_main1015.csv')
