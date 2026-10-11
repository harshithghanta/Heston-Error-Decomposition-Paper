import settings
import numpy as np


def trapezoid_rule(y, dx):
    return dx * (np.sum((y[:-1] + y[1:]))*0.5)


def heston_call(S, v, tau):
    w = np.linspace(1e-8, settings.UMAX, settings.NPOINTS)
    dw = w[1]-w[0]

    probs = []
    for j in [1,2]:
        if j ==1:
            u = 0.5
            b= settings.KAPPA - settings.RHO*settings.EPSILON
        else:
            u= -0.5
            b= settings.KAPPA
        a = settings.KAPPA * settings.ETA
        rspi = settings.RHO * settings.EPSILON * 1j * w
        d = np.sqrt((rspi -b)**2 - settings.EPSILON**2 * (2*u*1j*w-w**2))
        g=(b-rspi-d)/(b-rspi+d)
        edt = np.exp(-d*tau)
        C = settings.R*1j*w*tau + (a/settings.EPSILON**2)*((b-rspi -d)*tau - 2*np.log((1-g*edt)/(1-g)))
        D= ((b-rspi-d)/settings.EPSILON**2) * ((1-edt)/(1-g*edt))
        f = np.exp(C + D * v + 1j* w * np.log(S))
        y = np.real(f / (1j * w))
        p = 0.5 + trapezoid_rule(y, dw) / np.pi
        probs.append(p)
    price = S * probs[0] - np.exp(-settings.R * tau) * probs[1]
    delta = probs[0]
    return price, delta


def price_of(S,v,tau):
    price,delta = heston_call(S,v,tau)
    return price


def delta_of(S,v,tau):
    price, delta = heston_call(S, v, tau)
    return delta


def sv_greeks(S, v, tau):
    if v <=0:
        raise ValueError("VARIANCE MUST BE POSITIVE!")
    else:
        w = np.linspace(1e-8, settings.UMAX, settings.NPOINTS)
        dw = w[1]-w[0]

        pieces = []
        for j in [1,2]:
            if j ==1:
                u = 0.5
                b= settings.KAPPA - settings.RHO*settings.EPSILON
            else:
                u= -0.5
                b= settings.KAPPA
            a = settings.KAPPA * settings.ETA
            rspi = settings.RHO * settings.EPSILON * 1j * w
            d = np.sqrt((rspi -b)**2 - settings.EPSILON**2 * (2*u*1j*w-w**2))
            g=(b-rspi-d)/(b-rspi+d)
            edt = np.exp(-d*tau)
            C = settings.R*1j*w*tau + (a/settings.EPSILON**2)*((b-rspi -d)*tau - 2*np.log((1-g*edt)/(1-g)))
            D= ((b-rspi-d)/settings.EPSILON**2) * ((1-edt)/(1-g*edt))
            f = np.exp(C + D * v + 1j* w * np.log(S))
            pieces.append((f,D))
        f1, D1 = pieces[0]
        f2, D2 = pieces[1]

        def Q(z):
            return trapezoid_rule(np.real(z), dw)/np.pi
        iw = 1j*w
        disc = np.exp(-settings.R*tau)
        P1 = 0.5 + Q(f1 / (1j*w))
        P2 = 0.5 + Q(f2/(1j*w))
        term = {}
        term['price'] = S * P1 - disc * P2
        term['delta'] = P1
        term['gamma'] = Q(f1) / S
        term['speed'] = Q(f1 * (iw - 1)) / S ** 2
        term['vega'] = S * Q(D1 * f1 / iw) - disc * Q(D2 * f2 / iw)
        term['volga'] = S * Q(D1 ** 2 * f1 / iw) - disc * Q(D2 ** 2 * f2/ iw)
        term['vanna'] = Q(D1 * f1 / iw)
        return term


def time_greeks(S, v, tau):
    ht = 0.005 * tau
    g= sv_greeks(S,v,tau)
    up = sv_greeks(S,v,tau + ht)
    dn = sv_greeks(S,v,tau-ht)
    g['charm'] = - (up['delta'] - dn['delta']) / (2 * ht)
    g['veta'] = - (up['vega'] - dn['vega']) / (2 * ht)
    g['tconv'] = (up['price'] - 2 * g['price'] + dn['price']) / ht ** 2
    return g


def one_step(S0, v0, S1, v1, tau, dt, speed_out = False):

    g = time_greeks(S0, v0, tau)
    dS = S1 - S0
    dv = v1 - v0
    nu = settings.KAPPA * (settings.ETA - v0)
    terms = {}
    terms['gamma'] = 0.5 * g['gamma'] * (dS ** 2 - v0 * S0 ** 2 * dt)
    terms['volga'] = 0.5 * g['volga'] * (dv ** 2 - settings.EPSILON ** 2 * v0 * dt)
    terms['vanna'] = g['vanna'] * (dS * dv - settings.RHO * settings.EPSILON * v0 * S0 * dt)
    terms['vega'] = g['vega'] * (dv - nu * dt)
    terms['charm'] = g['charm'] * dS * dt
    terms['veta'] = g['veta'] * dv * dt
    terms['tconv'] = 0.5 * g['tconv'] * dt ** 2
    V0 = g['price']
    delta = g['delta']
    if tau - dt <= 1e-12:
        V1 = max(S1 - 1, 0.0)
    else:
        V1 = price_of(S1, v1, tau - dt)
    pnl = V1 - V0 - delta * dS - settings.R * (V0 - delta * S0) * dt
    if speed_out:
        expected = 3 * S0 ** 3 * dt ** 2 * (settings.R * v0 + v0 ** 2 + 0.5 * settings.RHO * settings.EPSILON * v0)
        speed = (1 / 6) * g['speed'] * (dS ** 3 - expected)
        return terms, pnl, speed
    return terms, pnl


def run_all(prices, variances, stride, close_early = 0, steps_daily = 76, fits = None):
    totals = {'gamma': [], 'volga': [], 'vanna': [], 'vega': [], 'charm': [], 'veta': [], 'tconv': []}
    PNL = []
    saved_eta = settings.ETA
    saved_eps = settings.EPSILON
    saved_rho = settings.RHO
    try:
        for day in range(len(prices)):
            if fits is not None:
                eta, eps, rho = fits[day]
                settings.ETA = eta
                settings.EPSILON = eps
                settings.RHO = rho

            S = np.array(prices[day])
            v = np.array(variances[day])
            last = len(S) - 1
            stop = last - close_early
            bars = list(range(0, stop+1, stride))
            if bars[-1] != stop:
                bars.append(stop)
            day_total = {'gamma': 0.0, 'volga': 0.0, 'vanna': 0.0, 'vega': 0.0, 'charm': 0.0, 'veta': 0.0, 'tconv': 0.0}
            day_pnl = 0.0
            for i in range(len(bars) - 1):
                start = bars[i]
                end = bars[i + 1]
                tau = settings.ONE_DAY * (last - start) / steps_daily + settings.TAU_OFF
                dt = settings.ONE_DAY * (end - start) / steps_daily
                terms, pnl = one_step(S[start], v[start], S[end], v[end], tau, dt)
                day_pnl += pnl
                for g in settings.GREEKS:
                    day_total[g] += terms[g]

            for l in settings.GREEKS:
                totals[l].append(day_total[l])
            PNL.append(day_pnl)
    finally:
        settings.ETA = saved_eta
        settings.EPSILON = saved_eps
        settings.RHO = saved_rho
    return totals, PNL


def euler_maruyamaFT(nbars, v0, npaths, seed, substeps, horizon):
    rng = np.random.default_rng(seed)
    last = nbars - 1
    fine = last * substeps
    h = horizon / fine
    rho_c = np.sqrt(1 - settings.RHO**2)
    S = np.ones(npaths)
    v = np.full(npaths, float(v0))
    S_path = np.empty((npaths, nbars))
    v_path = np.empty((npaths, nbars))
    S_path[:, 0] = S
    v_path[:, 0] = v
    keep = 0
    for step in range(1, fine + 1):
        z1 = rng.standard_normal(npaths)
        z2 = rng.standard_normal(npaths)
        dX = np.sqrt(h) * z1
        dW = np.sqrt(h) * (settings.RHO * z1 + rho_c * z2)
        vpos = np.maximum(v, 0)
        S = S + settings.R * S * h + np.sqrt(vpos) * S * dX
        v = v + settings.KAPPA * (settings.ETA - vpos) * h + settings.EPSILON * np.sqrt(vpos) * dW
        if step % substeps == 0:
            keep += 1
            S_path[:, keep] = S
            v_path[:, keep] = v
    return S_path, v_path
