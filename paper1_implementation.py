import math
import os
import sys
import numpy as np
import pandas as pd


# model settings 
ONE_DAY = 1.0 / 252.0
K = 1.0        # strike (each day is scaled so it opens at 1)
R = 0.045      # risk-free rate
KAPPA = 3.0    # how fast variance pulls back to its long-run level
THETA = 0.04   # long-run variance 
XI = 1.0       # volatility of volatility (paper baseline; 0.5 and 2.0 are the sweep)
RHO = -0.7     # correlation between price moves and variance moves
LAM = 0.0      # market price of volatility risk

NPOINTS = 2048   # number of points for the integral in the pricer
UMAX = 2500.0    # upper end of the integral (0DTE needs a wide range)

TERM_NAMES = ['gamma', 'volga', 'vanna', 'vega', 'charm', 'veta', 'tconv']


def trapezoid_rule(y, dx):
    # basic trapezoid rule for equally spaced points
    return np.sum((y[:-1] + y[1:]) / 2.0) * dx


# Heston price and delta of a European call, for one S and one v.
def heston_call(S, v, tau):
    if tau <= 1e-12:
        # at expiry the option is just its payoff
        price = max(S - K, 0.0)
        if S > K:
            delta = 1.0
        else:
            delta = 0.0
        return price, delta

    w = np.linspace(1e-8, UMAX, NPOINTS)
    dw = w[1] - w[0]

    # the two probabilities P1 and P2 from the Heston formula
    probs = []
    for j in [1, 2]:
        if j == 1:
            u = 0.5
            b = KAPPA - RHO * XI
        else:
            u = -0.5
            b = KAPPA
        a = KAPPA * THETA
        rspi = RHO * XI * 1j * w
        d = np.sqrt((rspi - b) ** 2 - XI ** 2 * (2 * u * 1j * w - w ** 2))
        g = (b - rspi - d) / (b - rspi + d)
        edt = np.exp(-d * tau)
        C = R * 1j * w * tau + (a / XI ** 2) * ((b - rspi - d) * tau - 2 * np.log((1 - g * edt) / (1 - g)))
        D = ((b - rspi - d) / XI ** 2) * ((1 - edt) / (1 - g * edt))
        f = np.exp(C + D * v + 1j * w * np.log(S))
        y = np.real(np.exp(-1j * w * np.log(K)) * f / (1j * w))
        p = 0.5 + trapezoid_rule(y, dw) / np.pi
        probs.append(p)

    price = S * probs[0] - K * np.exp(-R * tau) * probs[1]
    delta = probs[0]
    return price, delta


def price_of(S, v, tau):
    price, delta = heston_call(S, v, tau)
    return price


def delta_of(S, v, tau):
    price, delta = heston_call(S, v, tau)
    return delta


def get_greeks(S, v, tau):
    # all the Greeks we need, found by nudging the inputs a little
    # (note: d/dt = -d/dtau, so the time ones get a minus sign)
    hS = 0.005 * S
    hv = 0.001
    ht = 0.25 * tau

    g = {}
    g['price'] = price_of(S, v, tau)
    g['delta'] = delta_of(S, v, tau)
    g['gamma'] = (delta_of(S + hS, v, tau) - delta_of(S - hS, v, tau)) / (2 * hS)

    up = price_of(S, v + hv, tau)
    down = price_of(S, v - hv, tau)
    g['vega'] = (up - down) / (2 * hv)
    g['volga'] = (up - 2 * g['price'] + down) / hv ** 2
    g['vanna'] = (delta_of(S, v + hv, tau) - delta_of(S, v - hv, tau)) / (2 * hv)
    g['charm'] = -(delta_of(S, v, tau + ht) - delta_of(S, v, tau - ht)) / (2 * ht)

    vega_later = (price_of(S, v + hv, tau + ht) - price_of(S, v - hv, tau + ht)) / (2 * hv)
    vega_earlier = (price_of(S, v + hv, tau - ht) - price_of(S, v - hv, tau - ht)) / (2 * hv)
    g['veta'] = -(vega_later - vega_earlier) / (2 * ht)
    g['tconv'] = (price_of(S, v, tau + ht) - 2 * g['price'] + price_of(S, v, tau - ht)) / ht ** 2
    return g


def one_step(S0, v0, S1, v1, tau, dt):
    # the seven terms for one hedge step, plus the actual hedged P&L
    g = get_greeks(S0, v0, tau)
    dS = S1 - S0
    dv = v1 - v0
    nu = KAPPA * (THETA - v0) - LAM * v0

    terms = {}
    terms['gamma'] = 0.5 * g['gamma'] * (dS ** 2 - v0 * S0 ** 2 * dt)
    terms['volga'] = 0.5 * g['volga'] * (dv ** 2 - XI ** 2 * v0 * dt)
    terms['vanna'] = g['vanna'] * (dS * dv - RHO * XI * v0 * S0 * dt)
    terms['vega'] = g['vega'] * (dv - nu * dt)
    terms['charm'] = g['charm'] * dS * dt
    terms['veta'] = g['veta'] * dv * dt
    terms['tconv'] = 0.5 * g['tconv'] * dt ** 2

    if tau - dt <= 1e-12:
        V1 = max(S1 - K, 0.0)   # last step, the option becomes its payoff
    else:
        V1 = price_of(S1, v1, tau - dt)
    pnl = (V1 - g['price']) - g['delta'] * dS - R * (g['price'] - g['delta'] * S0) * dt
    return terms, pnl


def find_data_file(name="intraday.csv"):
    # locate the data file relative to THIS script, not the working directory,
    # so the script runs the same from an IDE, a cron job, or any folder.
    # an explicit path on the command line (argv[1]) always wins.
    if len(sys.argv) > 1 and sys.argv[1] not in ("mc", "validate", "full"):
        candidates = [sys.argv[1]]
    else:
        here = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(here, name),                        # next to the script
            os.path.join(here, "..", name),                  # repo root (original layout)
            os.path.join(here, "..", "Strategies-Code", name),
        ]
    for path in candidates:
        if os.path.isfile(path):
            return os.path.abspath(path)
    searched = "\n  ".join(os.path.abspath(c) for c in candidates)
    raise FileNotFoundError(
        f"Could not find {name}. Looked in:\n  {searched}\n"
        f"Pass the path explicitly: python {os.path.basename(__file__)} /path/to/{name}")


def load_days(path):
    # read the csv and return the date, price path and variance path of each day.
    # the variance state comes from VIX1D, the 1-day index, because that is the
    # maturity the 0DTE option actually lives on -- the 30-day VIX would be
    # marking a different part of the term structure.
    df = pd.read_csv(path, parse_dates=['timestamp'])
    if 'vix1d_close' not in df.columns:
        raise KeyError(
            f"{path} has no 'vix1d_close' column (found: {list(df.columns)}). "
            f"Rebuild it with: python fetch_data.py {path}")
    df['date'] = df['timestamp'].dt.date
    full = int(df.groupby('date').size().max())
    dates = []
    prices = []
    variances = []
    for day, rows in df.groupby('date'):
        rows = rows.sort_values('timestamp')
        if len(rows) != full:
            continue   # skip days with missing bars
        S = rows['und_close'].to_numpy(float)
        vix1d = rows['vix1d_close'].to_numpy(float)
        dates.append(str(day))
        prices.append(S / S[0])                 # scale so the day opens at 1
        variances.append((vix1d / 100.0) ** 2)  # VIX1D is in percent, v is a variance
    return dates, prices, variances


def run_all(prices, variances, stride, close_early=0):
    # add up the seven terms and the P&L for every day, hedging every `stride` bars.
    # close_early > 0 closes the position that many bars before the bell and marks
    # out at the pricer value; expiry itself stays at the bell, so the only thing
    # that changes is how close to the kink the last hedge step gets.
    nbars = len(prices[0])
    last = nbars - 1
    stop = last - close_early
    bars = list(range(0, stop + 1, stride))
    if bars[-1] != stop:
        bars.append(stop)

    totals = {}
    for name in TERM_NAMES:
        totals[name] = []
    pnls = []

    for day in range(len(prices)):
        S = prices[day]
        v = variances[day]
        day_total = {}
        for name in TERM_NAMES:
            day_total[name] = 0.0
        day_pnl = 0.0
        for i in range(len(bars) - 1):
            b0 = bars[i]
            b1 = bars[i + 1]
            tau = ONE_DAY * (last - b0) / last
            dt = ONE_DAY * (b1 - b0) / last
            terms, pnl = one_step(S[b0], v[b0], S[b1], v[b1], tau, dt)
            for name in TERM_NAMES:
                day_total[name] = day_total[name] + terms[name]
            day_pnl = day_pnl + pnl
        for name in TERM_NAMES:
            totals[name].append(day_total[name])
        pnls.append(day_pnl)

    for name in TERM_NAMES:
        totals[name] = np.array(totals[name])
    return totals, np.array(pnls)


# ---------------------------------------------------------------------------
# the two attribution measures
# ---------------------------------------------------------------------------

def gross_shares(totals, keep=None):
    # each term's average absolute daily contribution, as a percentage of the sum
    # across all seven. says how much a term MOVES, not how much of the realized
    # error it explains -- see variance_shares for that.
    avg = {}
    for name in TERM_NAMES:
        col = totals[name] if keep is None else totals[name][keep]
        avg[name] = float(np.mean(np.abs(col)))
    total = sum(avg.values())
    return {name: 100.0 * avg[name] / total for name in TERM_NAMES}


def variance_shares(totals, pnls, keep=None):
    # signed decomposition of the variance of the REALIZED hedged P&L:
    #   share_i = cov(term_i, pnl) / var(pnl)
    # the seven terms plus the unexplained remainder sum to 100 by construction,
    # and a term earns share only if it moves with the error actually realized.
    sl = slice(None) if keep is None else keep
    y = pnls[sl]
    var = float(np.var(y))
    shares = {}
    explained = np.zeros(len(y))
    for name in TERM_NAMES:
        col = totals[name][sl]
        explained = explained + col
        shares[name] = 100.0 * float(np.cov(col, y, bias=True)[0, 1]) / var
    shares['residual'] = 100.0 * float(np.cov(y - explained, y, bias=True)[0, 1]) / var
    return shares


def residual_pct(totals, pnls, keep=None):
    # the harsher gross metric: mean absolute unexplained P&L over mean absolute P&L
    sl = slice(None) if keep is None else keep
    y = pnls[sl]
    explained = np.zeros(len(y))
    for name in TERM_NAMES:
        explained = explained + totals[name][sl]
    return 100.0 * float(np.mean(np.abs(y - explained)) / np.mean(np.abs(y)))


def bootstrap_shares(totals, pnls, nboot=10000, seed=12345):
    # percentile bootstrap over days for both measures
    rng = np.random.default_rng(seed)
    ndays = len(pnls)
    stack = np.array([totals[name] for name in TERM_NAMES])
    gross = {name: [] for name in TERM_NAMES}
    varsh = {name: [] for name in TERM_NAMES}
    for _ in range(nboot):
        idx = rng.integers(0, ndays, ndays)
        sub = stack[:, idx]
        y = pnls[idx]
        avg = np.abs(sub).mean(axis=1)
        total = avg.sum()
        vy = np.var(y)
        for i, name in enumerate(TERM_NAMES):
            gross[name].append(100.0 * avg[i] / total)
            varsh[name].append(100.0 * np.cov(sub[i], y, bias=True)[0, 1] / vy)
    pick = lambda d: {name: (float(np.percentile(d[name], 2.5)),
                             float(np.percentile(d[name], 97.5))) for name in TERM_NAMES}
    return pick(gross), pick(varsh)


# ---------------------------------------------------------------------------
# inference: daily P&L is autocorrelated, so iid standard errors are wrong
# ---------------------------------------------------------------------------

def newey_west_se(x, lag=None):
    # HAC standard error of the mean. default lag is the usual 4*(T/100)^(2/9).
    x = np.asarray(x, float)
    T = len(x)
    if lag is None:
        lag = int(np.floor(4 * (T / 100.0) ** (2.0 / 9.0)))
    e = x - x.mean()
    s = float(np.dot(e, e) / T)
    for j in range(1, lag + 1):
        gamma_j = float(np.dot(e[j:], e[:-j]) / T)
        s = s + 2.0 * (1.0 - j / (lag + 1.0)) * gamma_j   # Bartlett weights
    return math.sqrt(max(s, 1e-30) / T), lag


def ljung_box(x, lags=10):
    x = np.asarray(x, float)
    T = len(x)
    e = x - x.mean()
    c0 = float(np.dot(e, e) / T)
    Q = 0.0
    acf = []
    for j in range(1, lags + 1):
        rj = float(np.dot(e[j:], e[:-j]) / T) / c0
        acf.append(rj)
        Q = Q + rj ** 2 / (T - j)
    return T * (T + 2) * Q, acf


def chi2_sf(x, k):
    # upper tail of a chi-squared, via the regularized incomplete gamma function
    a, xx = k / 2.0, x / 2.0
    if xx < a + 1:                      # series for P(a,x); return 1 - P
        term = 1.0 / a
        total = term
        n = 0
        while abs(term) > 1e-14 * abs(total) and n < 10000:
            n += 1
            term = term * xx / (a + n)
            total = total + term
        return 1.0 - total * math.exp(-xx + a * math.log(xx) - math.lgamma(a))
    tiny = 1e-300                       # continued fraction for Q(a,x)
    b, c = xx + 1 - a, 1 / tiny
    d = 1 / b
    h = d
    for i in range(1, 10000):
        an = -i * (i - a)
        b = b + 2
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1 / d
        delta = d * c
        h = h * delta
        if abs(delta - 1) < 1e-14:
            break
    return math.exp(-xx + a * math.log(xx) - math.lgamma(a)) * h


def mean_stats(x):
    # mean of a daily series with both iid and HAC standard errors
    x = np.asarray(x, float)
    T = len(x)
    mean = float(x.mean())
    se_iid = float(x.std(ddof=1) / math.sqrt(T))
    se_nw, lag = newey_west_se(x)
    return {'n': T, 'mean': mean, 'se_iid': se_iid, 't_iid': mean / se_iid,
            'se_nw': se_nw, 't_nw': mean / se_nw, 'lag': lag,
            'n_neg': int(np.sum(x < 0))}


# ---------------------------------------------------------------------------
# Barndorff-Nielsen & Shephard bipower jump test
# ---------------------------------------------------------------------------

MU1 = math.sqrt(2.0 / math.pi)
MU43 = 2.0 ** (2.0 / 3.0) * math.gamma(7.0 / 6.0) / math.gamma(0.5)


def bipower_test(S):
    # realized variance picks up jumps, bipower variation does not, so their gap
    # isolates the jump contribution. the ratio statistic below is standardized by
    # tri-power quarticity (Huang & Tauchen) and is ~N(0,1) under "no jumps today";
    # a large positive z means a jump was detected.
    r = np.diff(np.log(S))
    n = len(r)
    a = np.abs(r)
    rv = float(np.sum(r ** 2))
    bv = float(MU1 ** -2 * (n / (n - 1.0)) * np.sum(a[1:] * a[:-1]))
    p = a ** (4.0 / 3.0)
    tp = float(n * MU43 ** -3 * (n / (n - 2.0)) * np.sum(p[2:] * p[1:-1] * p[:-2]))
    theta = math.pi ** 2 / 4.0 + math.pi - 5.0     # = mu1^-4 + 2 mu1^-2 - 5
    z = (1.0 - bv / rv) / math.sqrt((theta / n) * max(1.0, tp / bv ** 2))
    return rv, bv, z


def simulate_heston(nbars, v0, npaths, seed=0, substeps=8, horizon=ONE_DAY):
    rng = np.random.default_rng(seed)
    last = nbars - 1
    fine = last * substeps
    h = horizon / fine
    sqrt_h = np.sqrt(h)
    rho_c = np.sqrt(1.0 - RHO ** 2)

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
        dX = sqrt_h * z1
        dW = sqrt_h * (RHO * z1 + rho_c * z2)
        vpos = np.maximum(v, 0.0)          # full truncation: variance can't go negative
        sqrt_v = np.sqrt(vpos)
        S = S + R * S * h + sqrt_v * S * dX
        v = v + KAPPA * (THETA - vpos) * h + XI * sqrt_v * dW
        if step % substeps == 0:
            keep += 1
            S_path[:, keep] = S
            v_path[:, keep] = v

    return S_path, v_path


def algebra_control(npaths, seed, t0_days=60, hedge_days=40):
    # is the decomposition algebra exact away from the expiry kink?
    # hedge a longer option (t0_days to expiry) once a day for hedge_days, then
    # close out at the pricer value -- so no step straddles the payoff kink and
    # every Greek is smooth. any residual here is pure second-order Taylor
    # truncation, which isolates "is the algebra right" from "is 0DTE just the
    # hard, kinked corner of it".
    t0 = t0_days * ONE_DAY
    S_path, v_path = simulate_heston(hedge_days + 1, THETA, npaths, seed=seed,
                                     horizon=hedge_days * ONE_DAY)
    pnls = np.zeros(npaths)
    explained = np.zeros(npaths)
    for d in range(npaths):
        S = S_path[d] / S_path[d][0]
        v = np.clip(v_path[d], 1e-8, None)
        for i in range(hedge_days):
            tau = t0 - i * ONE_DAY          # from t0 down to t0-hedge_days, never 0
            terms, pnl = one_step(S[i], v[i], S[i + 1], v[i + 1], tau, ONE_DAY)
            explained[d] += sum(terms[name] for name in TERM_NAMES)
            pnls[d] += pnl
    resid = float(np.mean(np.abs(pnls - explained)) / np.mean(np.abs(pnls)) * 100)
    corr = float(np.corrcoef(explained, pnls)[0, 1])
    return resid, corr


def monte_carlo_validation(npaths=200, seed=0):
    global THETA
    THETA = 0.04                # fixed, known long-run variance for the test
    v0 = THETA                  # start at the long-run level, ATM
    nbars = 79                  # 79 marks = 78 synthetic intervals; one finer than the data's 77
                                #   (the data has 78 bars = 77 intervals), chosen so the strides below divide evenly

    print()
    print("=== MONTE CARLO VALIDATION (self-consistent Heston) ===")
    print(f"paths={npaths}  theta={THETA:.4f}  v0={v0:.4f}  "
          f"kappa={KAPPA}  xi={XI}  rho={RHO}  grid={nbars - 1} intervals/day")

    # --- control: the algebra is exact away from the expiry kink ---
    cr, cc = algebra_control(npaths, seed)
    print(f"algebra control (60-day option hedged daily, closed 20 days before "
          f"expiry): residual {cr:.2f}%  corr {cc:.4f}")

    S_path, v_path = simulate_heston(nbars, v0, npaths, seed=seed)
    prices = [S_path[i] for i in range(npaths)]
    variances = [np.clip(v_path[i], 1e-8, None) for i in range(npaths)]

    last = nbars - 1
    strides = [1, 2, 3, 6, 13, 26, 39]      # even divisors of 78 -> clean dt
    dts, per_term_abs = [], {name: [] for name in TERM_NAMES}
    totals_by_stride = {}

    print()
    print("reproduction (sum of 7 terms vs realized hedged P&L):")
    print("  steps   dt(yr)     residual%    corr")
    for s in strides:
        totals, pnls = run_all(prices, variances, s)
        totals_by_stride[s] = (totals, pnls)
        explained = np.zeros(npaths)
        for name in TERM_NAMES:
            explained = explained + totals[name]
        resid = float(np.mean(np.abs(pnls - explained)) / np.mean(np.abs(pnls)) * 100)
        corr = float(np.corrcoef(explained, pnls)[0, 1])
        dt = ONE_DAY * s / last
        dts.append(dt)
        for name in TERM_NAMES:
            per_term_abs[name].append(float(np.mean(np.abs(totals[name]))))
        print(f"  {last // s:5d}   {dt:.2e}    {resid:6.2f}%    {corr:.4f}")

    # --- mean-zero check, on the finest grid (stride 1) ---
    totals, _ = totals_by_stride[1]
    print()
    print("mean-zero check at finest grid:")
    print("  term     mean         se         t-stat")
    for name in TERM_NAMES:
        col = totals[name]
        mean = float(np.mean(col))
        se = float(np.std(col) / np.sqrt(npaths))
        t = mean / se if se > 0 else 0.0
        print(f"  {name:>6}  {mean:+.3e}  {se:.3e}   {t:+6.2f}")

    # dt-scaling: slope of log mean|daily term| vs log dt
    logdt = np.log(np.array(dts))
    expected = {'gamma': 0.5, 'volga': 0.5, 'vanna': 0.5, 'vega': 0.0,
                'charm': 1.0, 'veta': 1.0, 'tconv': 1.0}
    note = {'gamma': 'discrete-hedging root-dt law',
            'volga': 'compensated, vanishes ~sqrt(dt)',
            'vanna': 'compensated, vanishes ~sqrt(dt)',
            'vega':  'UNHEDGED risk -- persists as dt->0',
            'charm': 'higher-order time Greek, vanishes fast',
            'veta':  'higher-order time Greek, vanishes fast',
            'tconv': 'higher-order time Greek, vanishes fast'}
    print()
    print("dt-scaling (slope of log mean|daily term| vs log dt; bigger = vanishes faster):")
    print("  term    slope   expected   interpretation")
    for name in TERM_NAMES:
        slope = float(np.polyfit(logdt, np.log(np.array(per_term_abs[name])), 1)[0])
        print(f"  {name:>6}  {slope:+.2f}    ~{expected[name]:.1f}      {note[name]}")


def bs_call(S, strike, r, sigma, tau):
    # Black--Scholes closed form, used for the xi->0 limit test
    from math import log, sqrt, exp
    from statistics import NormalDist
    N = NormalDist().cdf
    d1 = (log(S / strike) + (r + 0.5 * sigma ** 2) * tau) / (sigma * sqrt(tau))
    d2 = d1 - sigma * sqrt(tau)
    return S * N(d1) - strike * exp(-r * tau) * N(d2)


def pricer_validation():
    # turn "2048 points ensures convergence" and pricer correctness from
    # assertions into demonstrated facts: BS limit, put-call parity,
    # grid-doubling convergence, and FD bump-size stability.
    import math
    global XI, RHO, NPOINTS
    S = 1.0
    v = THETA
    tau_long, tau_short = 30 * ONE_DAY, ONE_DAY
    print()
    print("=== PRICER VALIDATION BATTERY ===")
    print(f"  ATM S={S:.2f} K={K:.2f} r={R} v={v:.4f}")

    # 1. Black-Scholes limit: xi->0, rho=0 => deterministic variance, so the
    #    Heston call must collapse onto BS with sigma = sqrt(v).
    x0, r0 = XI, RHO
    XI, RHO = 1e-8, 0.0
    for tau, tag in [(tau_long, "30-day"), (tau_short, "1-day ")]:
        c_h, _ = heston_call(S, v, tau)
        c_bs = bs_call(S, K, R, math.sqrt(v), tau)
        print(f"  BS limit {tag}: heston={c_h:.6f}  bs={c_bs:.6f}  |diff|={abs(c_h - c_bs):.2e}")
    XI, RHO = x0, r0

    # 2. Put-call parity: C - P should equal S - K e^{-r tau}.
    c_h, _ = heston_call(S, v, tau_long)
    p_h = c_h - S + K * math.exp(-R * tau_long)
    resid = (c_h - p_h) - (S - K * math.exp(-R * tau_long))
    print(f"  put-call parity: residual={resid:.2e}  (call={c_h:.6f} put={p_h:.6f})")

    # 3. Grid-doubling convergence at 0DTE scale -- the paper's actual claim.
    n0 = NPOINTS
    print("  grid-doubling convergence (1-day ATM call):")
    prev = None
    for n in [512, 1024, 2048, 4096, 8192]:
        NPOINTS = n
        p, _ = heston_call(S, v, tau_short)
        tail = "" if prev is None else f"   change={abs(p - prev):.2e}"
        print(f"    N={n:5d}  price={p:.8f}{tail}")
        prev = p
    NPOINTS = n0

    # 4. FD bump-size sensitivity for the finite-difference Greeks (1-day ATM).
    print("  FD bump-size sensitivity (1-day ATM):")
    for hv in [0.0005, 0.001, 0.002, 0.004]:
        vega = (price_of(S, v + hv, tau_short) - price_of(S, v - hv, tau_short)) / (2 * hv)
        print(f"    vega  hv={hv:.4f}  ->  {vega:.5f}")
    for hs in [0.0025, 0.005, 0.01, 0.02]:
        h = hs * S
        gamma = (delta_of(S + h, v, tau_short) - delta_of(S - h, v, tau_short)) / (2 * h)
        print(f"    gamma hS={hs * 100:.2f}%  ->  {gamma:.4f}")


def terciles_by(values, labels=("calm     ", "mid      ", "turbulent")):
    # split days into three buckets by a per-day sorting variable
    values = np.asarray(values, float)
    cut1 = np.quantile(values, 1 / 3)
    cut2 = np.quantile(values, 2 / 3)
    return [(labels[0], values <= cut1),
            (labels[1], (values > cut1) & (values <= cut2)),
            (labels[2], values > cut2)]


def main(full=False):
    global THETA, KAPPA, XI
    dates, prices, variances = load_days(find_data_file())
    THETA = float(np.mean(np.concatenate(variances)))   # long-run variance = data average
    ndays = len(prices)
    nbars = len(prices[0])
    print()
    print(f"Loaded {ndays} full days of {nbars} bars each ({dates[0]} to {dates[-1]}).")
    print(f"theta = {THETA:.6f}   kappa = {KAPPA}   xi = {XI}   rho = {RHO}   r = {R}")

    # ---- overall attribution, on both measures ----
    totals, pnls = run_all(prices, variances, 1)
    gross = gross_shares(totals)
    varsh = variance_shares(totals, pnls)
    leftover = residual_pct(totals, pnls)

    print()
    print("=== OVERALL ATTRIBUTION (5-min hedging) ===")
    print(f"  hedged P&L: mean={np.mean(pnls):+.5f}  sd={np.std(pnls):.5f}  "
          f"gross residual={leftover:.1f}%")
    print("  the variance share is the signed decomposition cov(term,pnl)/var(pnl);")
    print("  the gross share is mean|term| over the sum of mean|term| across terms.")
    print(f"  {'term':>6}  {'variance %':>10}  {'gross %':>8}")
    order = sorted(TERM_NAMES, key=lambda n: -gross[n])
    for name in order:
        print(f"  {name:>6}  {varsh[name]:+10.1f}  {gross[name]:8.1f}")
    print(f"  {'unexpl':>6}  {varsh['residual']:+10.1f}  {'--':>8}")

    # ---- bootstrap intervals over days ----
    ci_gross, ci_var = bootstrap_shares(totals, pnls)
    print()
    print("=== BOOTSTRAP 95% INTERVALS (B=10,000 resamples of days) ===")
    for name in order:
        print(f"  {name:>6}  variance [{ci_var[name][0]:+6.1f},{ci_var[name][1]:+6.1f}]"
              f"   gross [{ci_gross[name][0]:5.1f},{ci_gross[name][1]:5.1f}]")
    print(f"  corr(gamma, vega) = {np.corrcoef(totals['gamma'], totals['vega'])[0, 1]:+.3f}")

    # ---- inference. daily P&L is autocorrelated, so use HAC errors ----
    Q, acf = ljung_box(pnls, 10)
    print()
    print("=== MEANS WITH HAC (NEWEY-WEST) STANDARD ERRORS ===")
    print(f"  Ljung-Box(10) on daily P&L: Q={Q:.2f}  p={chi2_sf(Q, 10):.4f}   "
          f"acf(1)={acf[0]:+.2f}  acf(2)={acf[1]:+.2f}")
    print(f"  {'series':>10}  {'mean':>11}  {'t_iid':>6}  {'t_NW':>6}  {'lag':>3}  {'days<0':>7}")
    for label, series in [('pnl', pnls)] + [(n, totals[n]) for n in order]:
        s = mean_stats(series)
        print(f"  {label:>10}  {s['mean']:+.3e}  {s['t_iid']:+6.2f}  {s['t_nw']:+6.2f}"
              f"  {s['lag']:>3}  {s['n_neg']:>3}/{s['n']}")

    # ---- is the attribution an artifact of jump days? ----
    jump = [bipower_test(S) for S in prices]
    rv = np.array([j[0] for j in jump])
    bv = np.array([j[1] for j in jump])
    z = np.array([j[2] for j in jump])
    print()
    print("=== BIPOWER JUMP TEST AND JUMP-FREE SUBSAMPLE ===")
    print(f"  mean relative jump contribution max(RV-BV,0)/RV = "
          f"{100 * np.mean(np.maximum(rv - bv, 0.0) / rv):.2f}%")
    for alpha, zcrit in [(0.05, 1.6449), (0.01, 2.3263)]:
        keep = z <= zcrit
        drop = ~keep
        g = gross_shares(totals, keep)
        v = variance_shares(totals, pnls, keep)
        print(f"  alpha={alpha:<5} {int(drop.sum()):2d} jump days, {int(keep.sum()):2d} clean: "
              f"gamma {g['gamma']:.1f}% gross / {v['gamma']:+.1f}% variance, "
              f"vega {g['vega']:.1f}% / {v['vega']:+.1f}%, "
              f"residual {residual_pct(totals, pnls, keep):.1f}%")
        if drop.sum() >= 2:
            gj = gross_shares(totals, drop)
            print(f"{'':>16}jump days alone: gamma {gj['gamma']:.1f}%  vega {gj['vega']:.1f}%")
    flagged = [dates[i] for i in range(ndays) if z[i] > 2.3263]
    print(f"  days flagged at 1%: {', '.join(flagged) if flagged else '(none)'}")

    # ---- split days by how much VIX1D moved, then by how much the underlying moved ----
    vix_moves = np.array([abs(100 * np.sqrt(v[-1]) - 100 * np.sqrt(v[0])) for v in variances])
    abs_dS = np.array([abs(S[-1] - S[0]) for S in prices])
    for title, sortvar, unit, prec in [
            ("VIX1D-MOVE", vix_moves, "|dVIX1D| (pts)", 2),
            ("UNDERLYING-MOVE (control: does not enter the vega term)",
             abs_dS, "|dS| (day opens at 1)", 4)]:
        print()
        print(f"=== BY {title} REGIME ===")
        for label, keep in terciles_by(sortvar):
            g = gross_shares(totals, keep)
            v = variance_shares(totals, pnls, keep)
            vega = mean_stats(totals['vega'][keep])
            print(f"  {label} (n={int(np.sum(keep)):>2}, avg {unit}={np.mean(sortvar[keep]):.{prec}f}): "
                  f"gross gamma {g['gamma']:.1f}% vega {g['vega']:.1f}%  |  "
                  f"variance gamma {v['gamma']:+.1f}% vega {v['vega']:+.1f}%  |  "
                  f"vega t_NW {vega['t_nw']:+.2f}")
    print(f"  the two sorts correlate {np.corrcoef(vix_moves, abs_dS)[0, 1]:+.2f} (Pearson)")

    # ---- how good is the variance proxy, really ----
    dt = ONE_DAY / (nbars - 1)
    realized = np.array([float(np.sum(np.diff(S) ** 2)) for S in prices])
    modelled = np.array([float(np.sum(v[:-1] * S[:-1] ** 2 * dt))
                         for S, v in zip(prices, variances)])
    levels = np.array([100 * np.sqrt(v) for v in variances])
    roll = mean_stats(levels[:, -1] - levels[:, 0])
    print()
    print("=== VARIANCE-PROXY DIAGNOSTICS ===")
    print(f"  realized QV {realized.mean():.3e} vs model QV {modelled.mean():.3e}"
          f"  ({modelled.mean() / realized.mean():.2f}x, model exceeds on "
          f"{int(np.sum(modelled > realized))}/{ndays} days)")
    print(f"  VIX1D intraday roll: {levels[:, 0].mean():.2f} at the open -> "
          f"{levels[:, -1].mean():.2f} at the close, up on "
          f"{ndays - roll['n_neg']}/{ndays} days, t_NW {roll['t_nw']:+.2f}")
    print("  that roll is why the vega term carries a drift; it is a property of the")
    print("  proxy's rolling 24-hour window, not of the market.")

    if not full:
        print()
        print("Run `python paper1_implementation.py full` for the terminal-interval")
        print("ladder and the kappa/epsilon sweeps (several minutes).")
        return

    # ---- how much of the residual lives in the last few bars ----
    print()
    print("=== TERMINAL-INTERVAL LADDER ===")
    print("  closing the position early, marked out at the pricer value:")
    for k in [0, 1, 2, 3, 6, 12, 24]:
        if k == 0:
            t_k, p_k = totals, pnls
        else:
            t_k, p_k = run_all(prices, variances, 1, close_early=k)
        print(f"    {k:2d} bars early: gross residual {residual_pct(t_k, p_k):5.2f}%   "
              f"unexplained variance {variance_shares(t_k, p_k)['residual']:+5.2f}%", flush=True)

    # ---- the two uncalibrated parameters ----
    k0, x0 = KAPPA, XI
    for name, values, setter in [("kappa", [1.0, 3.0, 5.0], "KAPPA"),
                                 ("xi   ", [0.5, 1.0, 2.0], "XI")]:
        print()
        print(f"=== SWEEP OVER {name.strip().upper()} (all else held fixed) ===")
        for val in values:
            globals()[setter] = val
            t_s, p_s = run_all(prices, variances, 1)
            g, v = gross_shares(t_s), variance_shares(t_s, p_s)
            turb = terciles_by(vix_moves)[2][1]
            gt = gross_shares(t_s, turb)
            print(f"  {name}={val:<4}: gross gamma {g['gamma']:5.1f}% vega {g['vega']:5.1f}% "
                  f"vanna {g['vanna']:4.1f}% volga {g['volga']:4.1f}%  |  "
                  f"variance gamma {v['gamma']:+5.1f}% vega {v['vega']:+5.1f}%  |  "
                  f"turbulent gross gamma {gt['gamma']:.1f}% vega {gt['vega']:.1f}%  |  "
                  f"residual {residual_pct(t_s, p_s):.1f}%", flush=True)
        KAPPA, XI = k0, x0


if __name__ == "__main__":
    # `python paper1_implementation.py mc [npaths]` runs the Monte Carlo
    # validation, `validate` runs the pricer battery, `full` runs the empirical
    # study plus the ladder and sweeps; anything else runs the empirical study.
    if len(sys.argv) > 1 and sys.argv[1] == "mc":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 200
        monte_carlo_validation(npaths=n)
    elif len(sys.argv) > 1 and sys.argv[1] == "validate":
        pricer_validation()
    elif len(sys.argv) > 1 and sys.argv[1] == "full":
        main(full=True)
    else:
        main()
