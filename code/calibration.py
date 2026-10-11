import numpy as np
import load_data as L
import math
import settings
import pandas as pd
import heston_pricer as H

def trailing_eta(variances):
    etas = []
    for d in range(len(variances)):
        if d == 0:
            etas.append(variances[0][0])
        else:
            past = variances[max(0, d - 60):d]
            etas.append(np.mean(np.concatenate(past)))
    return etas


def bs_vega(S, K, tau, sig):
    d1 = (math.log(S / K) + (settings.R + 0.5 * sig**2) * tau) / (sig * math.sqrt(tau))
    return S * math.exp(-0.5 * d1**2) / math.sqrt(2 * math.pi) * math.sqrt(tau)


def load_smile(day, spot, tau, root):
    df = pd.read_parquet(f'data/{root}/{day}/bars.parquet')
    b = df[(df.bar == 0) & (df.bid > 0) & (df.ask > df.bid)].copy()
    b['mid'] = 0.5 * (b.bid + b.ask)
    C = b[b.right == 'C'].set_index('strike').mid
    P = b[b.right == 'P'].set_index('strike').mid
    both = C.index.intersection(P.index)
    if len(both) == 0:
        return None
    k0 = both[np.argmin(np.abs(both - spot))]
    F = k0 + math.exp(settings.R * tau) * (C[k0] - P[k0])
    S = F * math.exp(-settings.R * tau)
    rows = []
    for K in sorted(set(C.index) | set(P.index)):
        if abs(math.log(K / F)) > 0.03:
            continue
        if K >= F and K in C.index:
            c = C[K]
        elif K < F and K in P.index:
            c = P[K] + S - K * math.exp(-settings.R * tau)
        else:
            continue
        otm = C[K] if K >= F else P[K]
        if otm < 0.05:
            continue
        rows.append((K, c))
    if len(rows) < 5:
        return None
    return S, F, np.array(rows)



def lm(f, z0, iters=60):
    z = np.array(z0, float)
    r = f(z)
    cost = 0.5 * r @ r
    lam = 1e-2
    for _ in range(iters):
        J = np.empty((len(r), len(z)))
        for k in range(len(z)):
            h = 1e-5 * max(1.0, abs(z[k]))
            zz = z.copy()
            zz[k] += h
            J[:, k] = (f(zz) - r) / h
        g = J.T @ r
        A = J.T @ J
        improved = False
        for _ in range(10):
            step = np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-12), -g)
            zn = z + step
            rn = f(zn)
            cn = 0.5 * rn @ rn
            if np.isfinite(cn) and cn < cost:
                improved = cost - cn > 1e-12 * max(cost, 1e-30)
                z, r, cost = zn, rn, cn
                lam = max(lam / 3, 1e-9)
                break
            lam *= 4
        if not improved:
            break
    return z, r, cost


def to_p(z):
    return np.array([np.clip(math.exp(z[0]), 0.05, 30.0), 0.999 * math.tanh(z[1])])

def to_z(p):
    return np.array([math.log(p[0]), math.atanh(p[1] / 0.999)])


def fit_day(day, spot, v0, last, eta, root):
    tau = settings.ONE_DAY * (last) / 76 + settings.TAU_OFF
    smile = load_smile(day, spot, tau, root)
    if smile is None:
        return {'date': day, 'ok': False}
    S, F, rows = smile
    K = rows[:, 0]
    C = rows[:, 1]
    sig0 = math.sqrt(v0)
    vegas = np.array([max(bs_vega(S, Kk, tau, sig0), 1e-6) for Kk in K])
    keep = vegas >= 0.10 * bs_vega(S, F, tau, sig0)
    if keep.sum() < 5:
        return {'date': day, 'ok': False}
    K, C, vegas = K[keep], C[keep], vegas[keep]
    def f(z):
        p = to_p(z)
        settings.KAPPA, settings.ETA, settings.EPSILON, settings.RHO = 3.0, eta, p[0], p[1]
        model = np.array([Kk * H.price_of(S / Kk, v0, tau) for Kk in K])
        return (model - C) / vegas
    best = None
    for start in ([1.0, -0.7], [2.5, -0.3], [5.0, -0.9], [0.5, 0.0]):
        try:
            z, r, cost = lm(f, to_z(start))
        except Exception:
            continue
        if np.all(np.isfinite(r)) and (best is None or cost < best[2]):
            best = (z, r, cost)
    if best is None:
        return {'date': day, 'ok': False}
    p = to_p(best[0])
    return {'date': day, 'ok': True, 'eta': eta, 'eps': p[0], 'rho': p[1], 'v0': v0, 'n': len(K), 'F': F, 'rmse': float(np.sqrt(np.mean(best[1] ** 2)))}

if __name__ == '__main__':
    import sys
    sym = sys.argv[1]
    root = {'spy': 'pilot1m', 'spx': 'spx1m', 'qqq': 'qqq1m', 'ndx': 'ndx1m'}[sym]
    dates, prices, variances = L.load_days(f'results_v2/panels/{sym}5m_main1015.csv')
    etas = trailing_eta(variances)
    p = pd.read_csv(f'results_v2/panels/{sym}5m_main1015.csv')
    p['d'] = p.timestamp.str[:10]
    s0 = p.groupby('d').und_close.first()
    fits = []
    for i, d in enumerate(dates):
        fits.append(fit_day(d, float(s0[d]), variances[i][0], len(prices[i]) - 1, etas[i], root))
    pd.DataFrame(fits).to_csv(f'calibration/fits_{sym}.csv', index = False)
