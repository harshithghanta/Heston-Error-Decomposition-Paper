import numpy as np
import settings
import math
from scipy.special import gammaincc

def variance_shares(totals, pnls):
    p = np.array(pnls)
    var = np.var(p)
    shares = {}
    explained = np.zeros(len(p))
    for i in settings.GREEKS:
        a = np.array(totals[i])
        cov = np.cov(a, p, bias = True)[0,1]
        shares[i]=100 * cov/var
        explained +=a
    left = p - explained
    shares['residual'] = 100 * np.cov(left, p, bias = True)[0,1]/var
    return shares


def gross_shares(totals):
    avg = {}
    for i in settings.GREEKS:
        avg[i] = np.mean(np.abs(np.array(totals[i])))
    total = sum(avg.values())
    shares = {}

    for i in settings.GREEKS:
        shares[i] = 100 * avg[i]/total
    return shares


def residual_pct(totals, pnls):
    p = np.array(pnls)
    explained = np.zeros(len(p))
    for i in settings.GREEKS:
            a = np.array(totals[i])
            explained +=a
    left = p - explained
    top = np.mean(np.abs(left))
    bottom = np.mean(np.abs(p))
    return 100 * top/bottom


def newey_west(x, lag):
    x = np.array(x)
    n = len(x)
    e = x - x.mean()
    s = np.dot(e,e)/n
    for j in range(1, lag+1):
        gamma = np.dot(e[j:], e[:-j])/n
        s += 2 * (1-j / (lag+1)) * gamma
    se = np.sqrt(s/n)
    return x.mean() / se

def bootstrap_int(totals, pnls, reps, seed):
    p = np.array(pnls)
    A = {}
    rng = np.random.default_rng(seed)
    n = len(pnls)
    draws = {}
    gdraws = {}
    for g in settings.GREEKS:
        A[g] = np.array(totals[g])
        draws[g] = []
        gdraws[g] = []
    draws['residual'] = []

    for r in range(reps):
        idx = rng.integers(0,n,n)
        pb = p[idx]
        var = np.var(pb)
        explained = np.zeros(n)
        avg = {}
        for g in settings.GREEKS:
            share = 100 * np.cov(A[g][idx], pb, bias = True)[0,1] / var
            draws[g].append(share)
            explained += A[g][idx]
            avg[g] = np.mean(np.abs(A[g][idx]))
        left = pb - explained
        draws['residual'].append(100 * np.cov(left, pb, bias = True)[0,1] / var)
        total = sum(avg.values())
        for g in settings.GREEKS:
            gdraws[g].append(100 * avg[g] / total)

    ci = {}
    for k in draws:
        ci[k] = np.percentile(draws[k], [2.5, 97.5])
    gci = {}
    for g in gdraws:
        gci[g] = np.percentile(gdraws[g], [2.5, 97.5])

    return ci, gci

def block_bootstrap_int(totals, pnls, reps, block, seed):
    p = np.array(pnls)
    A = {}
    rng = np.random.default_rng(seed)
    n = len(pnls)
    draws = {}
    gdraws = {}
    for g in settings.GREEKS:
        A[g] = np.array(totals[g])
        draws[g] = []
        gdraws[g] = []
    draws['residual'] = []

    for r in range(reps):
        nb = math.ceil(n / block)
        starts = rng.integers(0,n,nb)
        idx = (starts[:, None] + np.arange(block)) % n
        idx = idx.ravel()[:n]
        pb = p[idx]
        var = np.var(pb)
        explained = np.zeros(n)
        avg = {}
        for g in settings.GREEKS:
            share = 100 * np.cov(A[g][idx], pb, bias = True)[0,1] / var
            draws[g].append(share)
            explained += A[g][idx]
            avg[g] = np.mean(np.abs(A[g][idx]))
        left = pb - explained
        draws['residual'].append(100 * np.cov(left, pb, bias = True)[0,1] / var)
        total = sum(avg.values())
        for g in settings.GREEKS:
            gdraws[g].append(100 * avg[g] / total)

    ci = {}
    for k in draws:
        ci[k] = np.percentile(draws[k], [2.5, 97.5])
    gci = {}
    for g in gdraws:
        gci[g] = np.percentile(gdraws[g], [2.5, 97.5])

    return ci, gci


def mean_stats(x):
    x = np.array(x)
    return np.mean(x) * 1e4, newey_west(x,6), np.sum(x<0)


def terciles_by(values):
    low, high = np.quantile(values, [1/3, 2/3])
    calm = values <= low
    medium = (values > low) & (values <= high)
    turbulent = values > high
    return calm, medium, turbulent


def ljung_box(x, lags):
    x = np.array(x)
    e = x - x.mean()
    n = len(x)
    c0 = np.dot(e,e)/n
    Q = 0
    for j in range(1, lags+1):
        rj = (np.dot(e[j:], e[:-j]) / n) / c0
        Q += rj**2 / (n-j)
    Q = n * (n+2) * Q
    p = gammaincc(lags/2, Q/2)
    return Q, p

def bipower_test(S):
    r = np.diff(np.log(S))
    n = len(r)
    a = np.abs(r)
    rv = np.sum(r**2)
    bv = settings.ABS_NORMAL_MEAN**-2 * (n/(n-1)) * np.sum(a[1:] * a[:-1])
    p = a **(4/3)
    tp = n * settings.ABS_NORMAL_MOMENT**-3 * (n/(n-2)) * np.sum(p[2:] * p[1:-1] * p[:-2])
    theta = math.pi**2/4 + math.pi - 5
    z = (1 - bv/rv) / math.sqrt((theta/n) * max(1, tp/bv**2))
    return rv, bv, z


def fit_floor(dt, err):
    err = np.array(err)
    X = np.column_stack([np.ones(len(dt)), np.sqrt(dt)])
    fit = np.linalg.lstsq(X, err, rcond = None)
    coef = fit[0]
    a = coef[0]
    b = coef[1]
    r2 = 1 - sum((err - X @ [a, b])**2) / sum((err - err.mean())**2)
    return a, b, r2


def block_bootstrap_floor(dt, pnls, reps = 10000, block = 10, seed = 7):
    errors = np.abs(np.asarray(pnls)) * 1e4
    rng = np.random.default_rng(seed)
    n = len(errors)
    draws = []
    for r in range(reps):
        nb = math.ceil(n / block)
        starts = rng.integers(0, n - block + 1, nb)
        idx = (starts[:, None] + np.arange(block)).ravel()[:n]
        err = errors[idx].mean(axis = 0)
        a, _, _ = fit_floor(dt, err)
        draws.append(100 * a / err[0])
    return np.percentile(draws, [2.5, 97.5])


def tick_ratio(pnl_fine, pnl_coarse, reps, seed):
    e1 = np.abs(pnl_fine) * 1e4
    e2 = np.abs(pnl_coarse) * 1e4
    rng = np.random.default_rng(seed)
    n = len(e1)
    draws = []
    for r in range(reps):
        idx = rng.integers(0, n, n)
        draws.append(e1[idx].mean() / e2[idx].mean())
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return e1.mean() / e2.mean(), lo, hi


def pooled_ratio(pairs, reps, seed):
    rng = np.random.default_rng(seed)
    n = len(pairs[0][0])
    draws = []
    for r in range(reps):
        idx = rng.integers(0, n, n)
        ratios = []
        for fine, coarse in pairs:
            e1 = np.abs(fine) * 1e4
            e2 = np.abs(coarse) * 1e4
            ratios.append(e1[idx].mean() / e2[idx].mean())
        draws.append(np.mean(ratios))
    real = np.mean([np.abs(fine).mean() / np.abs(coarse).mean() for fine, coarse in pairs])
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return real, lo, hi
