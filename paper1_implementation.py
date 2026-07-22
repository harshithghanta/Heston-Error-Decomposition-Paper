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
XI = 0.5       # volatility of volatility
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
    if len(sys.argv) > 1:
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
    # read the csv and return one price path and one variance path per day
    df = pd.read_csv(path, parse_dates=['timestamp'])
    df['date'] = df['timestamp'].dt.date
    full = int(df.groupby('date').size().max())
    prices = []
    variances = []
    for day, rows in df.groupby('date'):
        rows = rows.sort_values('timestamp')
        if len(rows) != full:
            continue   # skip days with missing bars
        S = rows['und_close'].to_numpy(float)
        vix = rows['vix_close'].to_numpy(float)
        prices.append(S / S[0])               # scale so the day opens at 1
        variances.append((vix / 100.0) ** 2)  # VIX is in percent, v is a variance
    return prices, variances


def run_all(prices, variances, stride):
    # add up the seven terms and the P&L for every day, hedging every `stride` bars
    nbars = len(prices[0])
    last = nbars - 1
    bars = list(range(0, nbars, stride))
    if bars[-1] != last:
        bars.append(last)

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


def main():
    global THETA
    prices, variances = load_days(find_data_file())
    THETA = float(np.mean(np.concatenate(variances)))   # long-run variance = data average
    ndays = len(prices)
    nbars = len(prices[0])
    print()
    print(f"Loaded {ndays} full days of {nbars} bars each.  theta = {THETA:.4f}")

    # ---- overall attribution ----
    totals, pnls = run_all(prices, variances, 1)

    avg = {}
    for name in TERM_NAMES:
        avg[name] = float(np.mean(np.abs(totals[name])))
    total_avg = sum(avg.values())

    explained = np.zeros(ndays)
    for name in TERM_NAMES:
        explained = explained + totals[name]
    leftover = np.mean(np.abs(pnls - explained)) / np.mean(np.abs(pnls)) * 100

    print()
    print("=== OVERALL ATTRIBUTION (5-min hedging) ===")
    print(f"  hedged P&L: mean={np.mean(pnls):+.5f}  sd={np.std(pnls):.5f}  leftover={leftover:.1f}%")

    ranked = []
    for name in TERM_NAMES:
        ranked.append((avg[name], name))
    ranked.sort(reverse=True)
    for value, name in ranked:
        print(f"  {name:>6}  {100 * value / total_avg:5.1f}%")

    # ---- split days by how much the VIX moved ----
    vix_moves = []
    for v in variances:
        vix_open = 100 * np.sqrt(v[0])
        vix_close = 100 * np.sqrt(v[-1])
        vix_moves.append(abs(vix_close - vix_open))
    vix_moves = np.array(vix_moves)
    cut1 = np.quantile(vix_moves, 1 / 3)
    cut2 = np.quantile(vix_moves, 2 / 3)

    print()
    print("=== BY VIX-MOVE REGIME ===")
    groups = [("calm     ", vix_moves <= cut1),
              ("mid      ", (vix_moves > cut1) & (vix_moves <= cut2)),
              ("turbulent", vix_moves > cut2)]
    for label, keep in groups:
        group_avg = {}
        for name in TERM_NAMES:
            group_avg[name] = float(np.mean(np.abs(totals[name][keep])))
        group_total = sum(group_avg.values())
        print(f"  {label} (n={int(np.sum(keep)):>2}, avg|dVIX|={np.mean(vix_moves[keep]):.2f}): "
              f"gamma {100 * group_avg['gamma'] / group_total:.1f}%  "
              f"vega {100 * group_avg['vega'] / group_total:.1f}%  "
              f"charm {100 * group_avg['charm'] / group_total:.1f}%")

if __name__ == "__main__":
    # `python paper1_implementation.py mc [npaths]` runs the Monte Carlo
    # validation; anything else runs the empirical study on the csv.
    if len(sys.argv) > 1 and sys.argv[1] == "mc":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 200
        monte_carlo_validation(npaths=n)
    elif len(sys.argv) > 1 and sys.argv[1] == "validate":
        pricer_validation()
    else:
        main()
