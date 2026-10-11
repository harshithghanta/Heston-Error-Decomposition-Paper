import settings
import numpy as np
import pandas as pd
import heston_pricer as H
import statistical_methods as SM
import load_data as L
import os

dates, prices, variances = L.load_days('results_v2/panels/spy5m_main1015.csv')
f = pd.read_csv('calibration/fits_spy.csv').set_index('date')
fits = []
for d in dates:
    fits.append((f.loc[d, 'eta'], f.loc[d, 'eps'], f.loc[d, 'rho']))

if os.path.exists('results_v2/runs/his_main_run.npz'):
    z = np.load('results_v2/runs/his_main_run.npz')
    pnls = z['pnls']
    totals = {}
    for g in settings.GREEKS:
        totals[g] = z[g]
else:
    totals, pnls = H.run_all(prices, variances, 1, fits = fits)
    np.savez('results_v2/runs/his_main_run.npz', pnls = pnls, **totals)




def table_freq():
    strides = [2, 3, 4, 6, 12]
    for s in strides:
        t, pn = H.run_all(prices, variances, s, fits = fits)
        shares = SM.variance_shares(t, pn)
        res = SM.residual_pct(t, pn)
        print(5 * s, shares['gamma'], shares['vega'], res)


def table_drifts():
    m, t, neg = SM.mean_stats(np.array(pnls))
    print('P&L', round(m, 3), round(t, 2), neg)
    order = ['gamma', 'vega', 'vanna', 'volga', 'charm', 'tconv', 'veta']
    for g in order:
        m, t, neg = SM.mean_stats(np.array(totals[g]))
        print(g, round(m, 3), round(t, 2), neg)

    vs = SM.variance_shares(totals, pnls)
    gs = SM.gross_shares(totals)
    ci, gci = SM.bootstrap_int(totals, pnls, 10000, 12345)
    for g in order:
        print(g, round(vs[g], 1), ci[g].round(1), round(gs[g], 1), gci[g].round(1))

    print('unexplained', round(vs['residual'], 1), ci['residual'].round(1))


def table_close_early(ns = (0, 1, 2, 3, 4, 6, 8, 12, 18, 24, 36)):
    for n in ns:
        t, pn = H.run_all(prices, variances, 1, close_early = n, fits = fits)
        pn = np.array(pn)
        left = pn - sum(np.array(t[g]) for g in settings.GREEKS)
        vs = SM.variance_shares(t, pn)
        print(n, round(1e4 * np.mean(np.abs(pn)), 3), round(1e4 * np.mean(pn), 2),
              round(1e4 * np.mean(np.abs(left)), 3), round(SM.residual_pct(t, pn), 2),
              round(vs['residual'], 2), round(vs['gamma'], 1), round(vs['vega'], 1))


def table_pnlsweep():
    def block(path, strides, steps):
        d, pr, v = L.load_days(path)
        f = pd.read_csv('calibration/fits_spy.csv').set_index('date')
        ft = [(f.loc[x, 'eta'], f.loc[x, 'eps'], f.loc[x, 'rho']) for x in d]
        print(path, len(d), 'days, mean v', np.mean(np.concatenate(v)))
        for s in strides:
            bars = list(range(0, steps + 1, s))
            if bars[-1] != steps:
                bars.append(steps)
            gaps = np.diff(bars)
            uneven = gaps.min() != gaps.max()
            t, pn = H.run_all(pr, v, s, steps_daily = steps, fits = ft)
            pn = np.array(pn)
            print(s, len(gaps), uneven, round(1e4 * np.mean(np.abs(pn)), 3), round(1e4 * np.mean(pn), 2))
    block('results_v2/panels/spy5m_main1015.csv', [1, 2, 3, 4, 6, 12], 76)
    block('results_v2/panels/spy1m452_main1015.csv', [1, 5, 10, 15, 20, 30, 60], 380)


def table_thirdorder():
    saved = (settings.KAPPA, settings.ETA, settings.EPSILON, settings.RHO)
    for stride in [1, 2, 4, 6, 12]:
        sp = np.zeros(len(prices))
        for day in range(len(prices)):
            settings.ETA, settings.EPSILON, settings.RHO = fits[day]
            S, v = prices[day], variances[day]
            last = len(S) - 1
            bars = list(range(0, last + 1, stride))
            if bars[-1] != last:
                bars.append(last)
            for a, b in zip(bars[:-1], bars[1:]):
                tau = settings.ONE_DAY * (last - a) / 76 + settings.TAU_OFF
                dt = settings.ONE_DAY * (b - a) / 76
                t, pn, s = H.one_step(S[a], v[a], S[b], v[b], tau, dt, speed_out = True)
                sp[day] += s
        t7, pn = H.run_all(prices, variances, stride, fits = fits)
        pn = np.array(pn)
        r7 = SM.residual_pct(t7, pn)
        left7 = pn - sum(np.array(t7[g]) for g in settings.GREEKS)
        left8 = left7 - sp
        u7 = 100 * np.cov(left7, pn, bias = True)[0, 1] / np.var(pn)
        u8 = 100 * np.cov(left8, pn, bias = True)[0, 1] / np.var(pn)
        r8 = 100 * np.mean(np.abs(left8)) / np.mean(np.abs(pn))
        print(5 * stride, round(r7, 2), round(r8, 2), round(r8 - r7, 2), round(u7, 2), round(u8, 2))
        if stride == 1:
            size = {g: np.mean(np.abs(t7[g])) for g in settings.GREEKS}
            size['speed'] = np.mean(np.abs(sp))
            gross = 100 * size['speed'] / sum(size.values())
            share = 100 * np.cov(sp, pn, bias = True)[0, 1] / np.var(pn)
            rank = sorted(size, key = size.get, reverse = True).index('speed') + 1
            print('speed at 5 m: gross share', round(gross, 2), 'variance share', round(share, 2), 'rank', rank, 'of 8')
    settings.KAPPA, settings.ETA, settings.EPSILON, settings.RHO = saved


def table_clock():
    def row(name, t, pn, v_used):
        pn = np.array(pn)
        m, tt, neg = SM.mean_stats(pn)
        vs = SM.variance_shares(t, pn)
        gs = SM.gross_shares(t)
        dt = settings.ONE_DAY / 76
        model = sum(np.sum(np.array(v[:-1]) * np.array(S[:-1]) ** 2 * dt) for S, v in zip(prices, v_used))
        real = sum(np.sum(np.diff(S) ** 2) for S in prices)
        print(name, round(m, 2), round(tt, 2), round(vs['gamma'], 2), round(vs['vega'], 2),
              round(gs['gamma'], 2), round(gs['vega'], 2), round(SM.residual_pct(t, pn), 2), round(model / real, 3))
    row('model close (main)', totals, pnls, variances)
    old_day, old_off = settings.ONE_DAY, settings.TAU_OFF
    settings.ONE_DAY, settings.TAU_OFF = 1 / 252, 0.0
    t, pn = H.run_all(prices, variances, 1, fits = fits)
    row('settle at payoff', t, pn, variances)
    scaled = [np.array(v) * 380 / 405 for v in variances]
    t, pn = H.run_all(prices, scaled, 1, fits = fits)
    row('variance rescaled', t, pn, scaled)
    settings.ONE_DAY, settings.TAU_OFF = old_day, old_off


def table_params():
    saved = (settings.KAPPA, settings.ETA, settings.EPSILON, settings.RHO)
    def row(name, t, pn):
        vs = SM.variance_shares(t, pn)
        print(name, round(vs['gamma'], 2), round(vs['vega'], 2),
              round(SM.residual_pct(t, pn), 2), round(1e4 * np.mean(pn), 2))
    row('main', totals, pnls)
    eps = [f[1] for f in fits]
    rho = [f[2] for f in fits]
    keep, trail = [], []
    for i in range(len(fits)):
        lo = max(0, i - 60)
        if i - lo < 20:
            continue
        keep.append(i)
        trail.append((fits[i][0], np.median(eps[lo:i]), np.median(rho[lo:i])))
    t, pn = H.run_all([prices[i] for i in keep], [variances[i] for i in keep], 1, fits = trail)
    row('trailing', t, pn)
    settings.KAPPA = 40
    t, pn = H.run_all(prices, variances, 1, fits = fits)
    settings.KAPPA = 3.0
    row('kappa40', t, pn)
    settings.ETA = 0.022364
    for name, k, e, r in [('round', 3, 1.0, -0.7), ('guided', 40, 1.5, -0.25), ('path', 3, 2.33, -0.26)]:
        settings.KAPPA, settings.EPSILON, settings.RHO = k, e, r
        t, pn = H.run_all(prices, variances, 1)
        row(name, t, pn)
    settings.KAPPA = 3.0
    settings.KAPPA, settings.ETA, settings.EPSILON, settings.RHO = saved


def table_summary4(sym):
    d, pr, v = L.load_days(f'results_v2/panels/{sym}5m_main1015.csv')
    f = pd.read_csv(f'calibration/fits_{sym}.csv').set_index('date')
    ft = [(f.loc[x, 'eta'], f.loc[x, 'eps'], f.loc[x, 'rho']) for x in d]
    t, pn = H.run_all(pr, v, 1, fits = ft)
    pn = np.array(pn)
    m, tt, neg = SM.mean_stats(pn)
    vs = SM.variance_shares(t, pn)
    gs = SM.gross_shares(t)
    ci, gci = SM.bootstrap_int(t, pn, 10000, 12345)
    print(sym, len(d), round(m, 2), round(tt, 2), round(100 * neg / len(d), 1),
          round(vs['gamma'], 1), ci['gamma'].round(1), round(vs['vega'], 1), ci['vega'].round(1),
          round(gs['gamma'], 1), round(gs['vega'], 1), round(vs['residual'], 1), round(SM.residual_pct(t, pn), 2))


def table_xsec4():
    Z = {s: np.load(f'results_v2/runs/his_1m_{s}.npz') for s in ['spy', 'spx', 'qqq', 'ndx']}
    shared = set.intersection(*[set(Z[s]['dates']) for s in Z])
    for label, use_shared in [('900 shared', True), ('own days', False)]:
        print(label)
        for s in Z:
            if use_shared:
                m = np.array([x in shared for x in Z[s]['dates']])
            else:
                m = np.ones(len(Z[s]['dates']), bool)
            t = {g: Z[s][g][m] for g in settings.GREEKS}
            pn = Z[s]['pnl_5'][m]
            vs = SM.variance_shares(t, pn)
            gs = SM.gross_shares(t)
            print(s, m.sum(), round(float(Z[s]['avg_v']), 4), round(vs['gamma'], 2), round(vs['vega'], 2),
                  round(gs['gamma'], 2), round(gs['vega'], 2), round(vs['residual'], 2),
                  round(SM.residual_pct(t, pn), 2), round(1e4 * np.mean(pn), 2))


def run_1m(sym):
    d, pr, v = L.load_days(f'results_v2/panels/{sym}1m_main1015.csv')
    f = pd.read_csv(f'calibration/fits_{sym}.csv').set_index('date')
    ft = [(f.loc[x, 'eta'], f.loc[x, 'eps'], f.loc[x, 'rho']) for x in d]
    out = {'dates': np.array(d), 'avg_v': np.mean(np.concatenate(v))}
    for s in [1, 5, 10, 15, 20, 30, 60]:
        t, pn = H.run_all(pr, v, s, steps_daily = 380, fits = ft)
        out[f'pnl_{s}'] = np.array(pn)
        if s == 5:
            for g in settings.GREEKS:
                out[g] = np.array(t[g])
    np.savez(f'results_v2/runs/his_1m_{sym}.npz', **out)


def table_floor4():
    Z = {s: np.load(f'results_v2/runs/his_1m_{s}.npz') for s in ['spy', 'spx', 'qqq', 'ndx']}
    shared = set.intersection(*[set(Z[s]['dates']) for s in Z])
    mins = [1, 5, 10, 15, 20, 30, 60]
    for label, use_shared in [('900 shared', True), ('own days', False)]:
        print(label)
        for s in Z:
            if use_shared:
                m = np.array([x in shared for x in Z[s]['dates']])
            else:
                m = np.ones(len(Z[s]['dates']), bool)
            err = [1e4 * np.mean(np.abs(Z[s][f'pnl_{k}'][m])) for k in mins]
            a, b, r2 = SM.fit_floor(np.array(mins), np.array(err))
            print(s, m.sum(), np.round(err, 2), round(a, 2), round(b, 3), round(r2, 4), round(100 * a / err[0]))


def run_ticks(sym, strides = (1, 3, 6, 30, 360)):
    d, pr, v = L.load_days(f'results_v2/runs/ticks360/panels/{sym}_tick10.csv')
    f = pd.read_csv(f'calibration/fits_{sym}.csv').set_index('date')
    ft = [(f.loc[x, 'eta'], f.loc[x, 'eps'], f.loc[x, 'rho']) for x in d]
    for s in strides:
        t, pn = H.run_all(pr, v, s, steps_daily = 2280, fits = ft)
        np.save(f'results_v2/runs/his_ticks_{sym}_{s}.npy', np.array(pn))


def table_ticks4():
    syms = ['spy', 'spx', 'qqq', 'ndx']
    P = {s: {k: np.load(f'results_v2/runs/his_ticks_{s}_{k}.npy') for k in [1, 3, 6, 30, 360]} for s in syms}
    for s in syms:
        Z = np.load(f'results_v2/runs/his_1m_{s}.npz')
        mins = [1, 5, 10, 15, 20, 30, 60]
        err = [1e4 * np.mean(np.abs(Z[f'pnl_{k}'])) for k in mins]
        a, b, r2 = SM.fit_floor(np.array(mins), np.array(err))
        pred = (a + b * np.sqrt(1 / 6)) / (a + b)
        ratio, lo, hi = SM.tick_ratio(P[s][1], P[s][6], 5000, 20260927)
        E = [round(1e4 * np.mean(np.abs(P[s][k])), 2) for k in [1, 3, 6, 30, 360]]
        print(s, E, round(ratio, 2), [round(lo, 2), round(hi, 2)], round(pred, 2))
    print('pooled', SM.pooled_ratio([(P[s][1], P[s][6]) for s in syms], 5000, 20260927))


def table_vegasplit(sym, com = None):
    saved = (settings.KAPPA, settings.ETA, settings.EPSILON, settings.RHO)
    if com is None:
        com = set.intersection(*[set(L.load_days(f'results_v2/panels/{s}1m_main1015.csv')[0]) for s in ['spy', 'spx', 'qqq', 'ndx']])
    d, pr, v = L.load_days(f'results_v2/panels/{sym}5m_main1015.csv')
    f = pd.read_csv(f'calibration/fits_{sym}.csv').set_index('date')
    sel = [i for i, x in enumerate(d) if x in com]
    full = np.array([len(v[i]) == 77 for i in sel])
    dV = np.zeros((len(sel), 76))
    for j, i in enumerate(sel):
        if full[j]:
            dV[j] = np.diff(v[i])
    csum = np.cumsum(dV, axis = 0)
    ccnt = np.cumsum(full)
    T = {g: [] for g in settings.GREEKS}
    pnl, drift, surprise = [], [], []
    for j in range(60, len(sel)):
        i = sel[j]
        curve = csum[j - 1] / ccnt[j - 1]
        settings.ETA, settings.EPSILON, settings.RHO = f.loc[d[i], 'eta'], f.loc[d[i], 'eps'], f.loc[d[i], 'rho']
        S, vv = pr[i], v[i]
        last = len(S) - 1
        off = 0 if last == 76 else 36
        day = {g: 0.0 for g in settings.GREEKS}
        p = dr = su = 0.0
        for b in range(last):
            tau = settings.ONE_DAY * (last - b) / 76 + settings.TAU_OFF
            dt = settings.ONE_DAY / 76
            t, pn = H.one_step(S[b], vv[b], S[b+1], vv[b+1], tau, dt)
            vg = H.sv_greeks(S[b], vv[b], tau)['vega']
            E = curve[b + off]
            nudt = settings.KAPPA * (settings.ETA - vv[b]) * dt
            su += vg * (vv[b+1] - vv[b] - E)
            dr += vg * (E - nudt)
            p += pn
            for g in settings.GREEKS:
                day[g] += t[g]
        for g in settings.GREEKS:
            T[g].append(day[g])
        pnl.append(p)
        drift.append(dr)
        surprise.append(su)
    pnl, drift, surprise = np.array(pnl), np.array(drift), np.array(surprise)
    share = lambda x, y: 100 * np.cov(x, y, bias = True)[0, 1] / np.var(y)
    rng = np.random.default_rng(12345)
    n = len(pnl)
    bs = {'surprise': [], 'drift': []}
    for r in range(10000):
        idx = rng.integers(0, n, n)
        bs['surprise'].append(share(surprise[idx], pnl[idx]))
        bs['drift'].append(share(drift[idx], pnl[idx]))
    for name, x in [('vega', np.array(T['vega'])), ('drift', drift), ('surprise', surprise)]:
        m, tt, neg = SM.mean_stats(x)
        print(sym, name, n, round(m, 2), round(tt, 1), round(share(x, pnl), 1),
              np.percentile(bs[name], [2.5, 97.5]).round(1) if name in bs else '')
    settings.KAPPA, settings.ETA, settings.EPSILON, settings.RHO = saved


def table_volhedge():
    saved = (settings.KAPPA, settings.ETA, settings.EPSILON, settings.RHO)
    rows = [('delta only', None), ('0.5% OTM', 1.005), ('1% OTM', 1.01)]
    base = None
    for name, kh in rows:
        book = np.zeros(len(prices))
        capped = 0
        bars_total = 0
        for day in range(len(prices)):
            settings.ETA, settings.EPSILON, settings.RHO = fits[day]
            S, v = prices[day], variances[day]
            last = len(S) - 1
            for b in range(last):
                tau = settings.ONE_DAY * (last - b) / 76 + settings.TAU_OFF
                dt = settings.ONE_DAY / 76
                t, p = H.one_step(S[b], v[b], S[b+1], v[b+1], tau, dt)
                bars_total += 1
                if kh is None:
                    book[day] += p
                    continue
                vT = H.sv_greeks(S[b], v[b], tau)['vega']
                th, ph = H.one_step(S[b] / kh, v[b], S[b+1] / kh, v[b+1], tau, dt)
                vH = kh * H.sv_greeks(S[b] / kh, v[b], tau)['vega']
                if vH <= 1e-12:
                    n = -25
                    capped += 1
                else:
                    n = -vT / vH
                    if abs(n) >= 25:
                        n = np.sign(n) * 25
                        capped += 1
                book[day] += p + n * kh * ph
        if base is None:
            base = book
        m, tt, neg = SM.mean_stats(book)
        print(name, round(1e4 * np.mean(np.abs(book)), 2), round(1e4 * np.std(book, ddof = 1), 2), round(m, 2), round(tt, 2),
              round(100 * (np.var(book) / np.var(base) - 1), 1), round(100 * capped / bars_total, 1))
    settings.KAPPA, settings.ETA, settings.EPSILON, settings.RHO = saved


def table_staleness():
    for s in ['spy', 'qqq', 'spx', 'ndx']:
        df = pd.read_csv(f'results_v2/runs/ticks360/stale_{s}.csv')
        out = [s]
        for h in [10, 30, 60]:
            r = df[df.h == h]
            out.append(round(100 * (1 - r.used_fresh.sum() / r.used_cb.sum()), 1))
        r = df[df.h == 60]
        out.append(round((r.ticks_band_per_cb * r.band_cb).sum() / r.band_cb.sum()))
        print(*out)


def table_stale(sym):
    rules = ['clock0', 'clock1', 'clock2', 'clock5', 'clock15', 'asof', 'arrival95', 'move1', 'move2', 'move3']
    f = pd.read_csv(f'calibration/fits_{sym}.csv').set_index('date')
    for rule in rules:
        d, pr, v = L.load_days(f'results_v2/panels/stale_main1015/panel_{sym}_{rule}.csv')
        ft = [(f.loc[x, 'eta'], f.loc[x, 'eps'], f.loc[x, 'rho']) for x in d]
        t, pn = H.run_all(pr, v, 1, fits = ft)
        vs = SM.variance_shares(t, pn)
        print(sym, rule, len(d), round(vs['gamma'], 2), round(vs['vega'], 2), round(SM.residual_pct(t, pn), 2))


def table_fill():
    p0 = np.array(pnls)
    T0 = {g: np.array(totals[g]) for g in settings.GREEKS}
    def row(name, T, p):
        vs = SM.variance_shares(T, p)
        m, t, neg = SM.mean_stats(p)
        print(name, len(p), round(m, 2), round(t, 2), round(vs['gamma'], 2), round(vs['vega'], 2),
              round(vs['residual'], 2), round(SM.residual_pct(T, p), 2))
    def swap(day, S, v):
        i = dates.index(day)
        t, pn = H.run_all([S], [v], 1, fits = [fits[i]])
        T = {g: T0[g].copy() for g in settings.GREEKS}
        p = p0.copy()
        for g in settings.GREEKS:
            T[g][i] = t[g][0]
        p[i] = pn[0]
        return T, p
    row('main', T0, p0)
    i = dates.index('2025-04-07')
    S = pd.Series(prices[i]).copy()
    v = pd.Series(variances[i]).copy()
    S[1:4] = np.nan
    v[1:4] = np.nan
    row('0407 interp', *swap('2025-04-07', S.interpolate().to_numpy(), v.interpolate().to_numpy()))
    i = dates.index('2025-04-09')
    for name, val in [('unscreened', (63.84996 / 100) ** 2), ('prev', variances[i][45]), ('next', variances[i][47])]:
        v = np.array(variances[i]).copy()
        v[46] = val
        row('0409 ' + name, *swap('2025-04-09', prices[i], v))
    for name, days in [('drop 0407', ['2025-04-07']), ('drop 0409', ['2025-04-09']), ('drop both', ['2025-04-07', '2025-04-09'])]:
        keep = np.array([d not in days for d in dates])
        row(name, {g: T0[g][keep] for g in settings.GREEKS}, p0[keep])


def text_numbers():
    years = np.array([d[:4] for d in dates])
    p = np.array(pnls)
    groups = {}
    for y in ['2022', '2023', '2024', '2025', '2026']:
        groups[y] = years == y
    half = np.arange(len(p)) < len(p) // 2
    groups['first half'] = half
    groups['second half'] = ~half
    for name, m in groups.items():
        sub = {g: np.array(totals[g])[m] for g in settings.GREEKS}
        unexpl = SM.variance_shares(sub, p[m])['residual']
        gross = SM.residual_pct(sub, p[m])
        print(name, m.sum(), round(unexpl, 2), round(gross, 2))
    T = {g: np.array(totals[g]) for g in settings.GREEKS}
    move_S = np.array([abs(S[-1] - S[0]) for S in prices])
    move_v = np.array([abs(100 * np.sqrt(v[-1]) - 100 * np.sqrt(v[0])) for v in variances])
    for name, x in [('control', move_S), ('proxy', move_v)]:
        for k in SM.terciles_by(x):
            sub = {g: T[g][k] for g in settings.GREEKS}
            vs = SM.variance_shares(sub, p[k])
            gs = SM.gross_shares(sub)
            print(name, k.sum(), round(vs['gamma'], 1), round(vs['vega'], 1), round(gs['vega'], 1),
                  round(1e4 * np.std(p[k]), 2), round(1e4 * np.mean(p[k]), 2))
    print('ljung-box', SM.ljung_box(p, 10))
    shares, jumps = [], 0
    for S in prices:
        rv, bv, z = SM.bipower_test(np.array(S))
        shares.append(max(rv - bv, 0) / rv)
        jumps += z > 1.6449
    print('jumps', round(100 * np.mean(shares), 1), jumps)
    mk = pd.read_csv('results_v2/runs/v2_calib/market_pnl.csv')['market_pnl'].to_numpy()
    tot = sum(T[g] for g in settings.GREEKS)
    print('corr', round(np.corrcoef(mk, p)[0, 1], 3), 'market mean', round(1e4 * np.mean(mk), 2))
    print('market unexpl', round(SM.variance_shares(T, mk)['residual'], 1), 'gross', round(100 * np.mean(np.abs(mk - tot)) / np.mean(np.abs(mk)), 1))


def validation_sim():
    saved = (settings.KAPPA, settings.ETA, settings.EPSILON, settings.RHO)
    settings.KAPPA, settings.ETA, settings.EPSILON, settings.RHO = 3.0, 0.04, 1.0, -0.7
    D = settings.ONE_DAY
    S_path, v_path = H.euler_maruyamaFT(41, 0.04, 200, 0, 8, 40 * D)
    pnl_sum = np.zeros(200)
    explained = np.zeros(200)
    for d in range(200):
        S = S_path[d] / S_path[d][0]
        v = np.clip(v_path[d], 1e-8, None)
        for i in range(40):
            t, pn = H.one_step(S[i], v[i], S[i+1], v[i+1], 60 * D - i * D, D)
            explained[d] += sum(t.values())
            pnl_sum[d] += pn
    print('algebra control resid %', 100 * np.mean(np.abs(pnl_sum - explained)) / np.mean(np.abs(pnl_sum)))
    S_path, v_path = H.euler_maruyamaFT(79, 0.04, 200, 0, 8, D)
    prices_sim = [S_path[i] for i in range(200)]
    variances_sim = [np.clip(v_path[i], 1e-8, None) for i in range(200)]
    dts = []
    sizes = []
    for s in [1, 2, 3, 6, 13, 26, 39]:
        t, pn = H.run_all(prices_sim, variances_sim, s, steps_daily = 78)
        dts.append(D * s / 78)
        sizes.append(np.mean(np.abs(t["gamma"])))
    print('gamma slope', np.polyfit(np.log(dts), np.log(sizes), 1)[0])
    settings.KAPPA, settings.ETA, settings.EPSILON, settings.RHO = saved


if __name__ == '__main__':
    import sys
    globals()[sys.argv[1]](*sys.argv[2:])
