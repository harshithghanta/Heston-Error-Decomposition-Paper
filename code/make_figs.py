import settings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap, LightSource, Normalize
import heston_pricer as H
import load_data as L

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['STIX Two Text', 'Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['legend.frameon'] = False
BLUE = '#2F6FBF'
AMBER = '#D97A1E'
GREY = '#BDBDB8'
INK = '#1a1a1a'
CMAP = LinearSegmentedColormap.from_list('persp', ['#58c95c', '#62d18a', '#5cc9c4', '#4a9be6', '#3d63e0'])


dates, prices, variances = L.load_days('results_v2/panels/spy5m_main1015.csv')
f = pd.read_csv('calibration/fits_spy.csv').set_index('date')
fits = []
for d in dates:
    fits.append((f.loc[d, 'eta'], f.loc[d, 'eps'], f.loc[d, 'rho']))


def day_bars(S, v, fit):
    settings.ETA, settings.EPSILON, settings.RHO = fit
    last = len(S) - 1
    all_terms = []
    pnls = []
    for b in range(last):
        tau = settings.ONE_DAY * (last - b) / 76 + settings.TAU_OFF
        dt = settings.ONE_DAY / 76
        terms, pnl = H.one_step(S[b], v[b], S[b+1], v[b+1], tau, dt)
        all_terms.append(terms)
        pnls.append(pnl)
    return all_terms, pnls


def fig_example_day():
    i = dates.index('2024-08-28')
    all_terms, pnls = day_bars(prices[i], variances[i], fits[i])
    print(i, len(pnls))
    running_pnl = np.cumsum(pnls) * 1e4
    seven = [sum(t[g] for g in settings.GREEKS) for t in all_terms]
    running_seven = np.cumsum(seven) * 1e4
    running_vega = np.cumsum([t['vega'] for t in all_terms]) * 1e4
    running_gamma = np.cumsum([t['gamma'] for t in all_terms]) * 1e4
    hours = 9 + 31 / 60 + np.arange(1, len(pnls) + 1) * 5 / 60
    print(np.corrcoef(running_pnl, running_seven)[0, 1], running_vega[-1], running_gamma[-1])
    fig, (top, bottom) = plt.subplots(2, 1, sharex = True, figsize = (6, 4.4))
    top.plot(hours, running_pnl, label = 'Realized hedged P&L', color = INK)
    top.plot(hours, running_seven, label = 'Sum of the seven terms', linestyle = '--', color = BLUE)
    bottom.plot(hours, running_gamma, label = 'Gamma term', color = BLUE)
    bottom.plot(hours, running_vega, label = 'Vega term', color = AMBER)
    top.legend(loc = 'lower center', bbox_to_anchor = (0.5, 1.0), ncol = 2)
    bottom.legend()
    top.set_ylabel('Cumulative (bp)')
    bottom.set_ylabel('Cumulative (bp)')
    bottom.set_xlabel('Time of day (ET)')
    bottom.set_xticks([10, 12, 14, 16])
    bottom.set_xticklabels(['10:00', '12:00', '14:00', '16:00'])
    plt.savefig('fig_example_day.pdf', bbox_inches = 'tight')
    plt.close()


def fig_shares_tod():
    size = np.zeros((76, 7))
    for i in range(len(dates)):
        if len(prices[i]) != 77:
            continue
        all_terms, pnls = day_bars(prices[i], variances[i], fits[i])
        for b in range(76):
            for k, g in enumerate(settings.GREEKS):
                size[b, k] += abs(all_terms[b][g])
    total = size.sum(axis = 1)
    gamma = 100 * size[:, 0] / total
    vega = 100 * size[:, 3] / total
    other = 100 - gamma - vega
    print(gamma[0], vega[0])
    hours = 9 + 31 / 60 + np.arange(1, 77) * 5 / 60
    plt.figure(figsize = (6, 4.4))
    plt.stackplot(hours, gamma, vega, other, labels = ['Gamma', 'Vega', 'Other five terms'],
                  colors = [BLUE, AMBER, GREY], edgecolor = 'white', linewidth = 0.6)
    plt.xlabel('Time of day (ET)'), plt.ylabel('Gross share of the error (%)')
    plt.xticks([10, 12, 14, 16], ['10:00', '12:00', '14:00', '16:00'])
    plt.xlim(hours[0], hours[-1]), plt.ylim(0, 100)
    plt.legend(loc = 'upper center', bbox_to_anchor = (0.5, -0.16), ncol = 3)
    plt.savefig('fig_shares_tod.pdf', bbox_inches = 'tight')
    plt.close()


SYMS = ['spy', 'spx', 'qqq', 'ndx']


def shared_dates():
    sets = []
    for s in SYMS:
        d, p, v = L.load_days(f'results_v2/panels/{s}1m_main1015.csv')
        sets.append(set(d))
    return set.intersection(*sets)


def past_only_curves(sym, com):
    d5, p5, v5 = L.load_days(f'results_v2/panels/{sym}5m_main1015.csv')
    sel = [i for i, d in enumerate(d5) if d in com]
    full = np.array([len(v5[i]) == 77 for i in sel])
    dV = np.zeros((len(sel), 76))
    for j, i in enumerate(sel):
        if full[j]:
            dV[j] = np.diff(v5[i])
    csum = np.cumsum(dV, axis = 0)
    ccnt = np.cumsum(full)
    curves = np.array([csum[j - 1] / ccnt[j - 1] for j in range(60, len(sel))]) * 1e4
    when = [d5[sel[j]] for j in range(60, len(sel))]
    return curves, when


def fig_vega_curve_3d():
    com = shared_dates()
    hours = 9 + 31/60 + (np.arange(76) + 0.5) * 5 / 60
    fig = plt.figure(figsize = (7.4, 7.6))
    for k, s in enumerate(SYMS):
        curves, when = past_only_curves(s, com)
        ax = fig.add_subplot(2, 2, k + 1, projection = '3d', computed_zorder = False)
        Y = mdates.date2num(pd.to_datetime(when))
        hf = np.linspace(13.5, hours[-1], 240)
        Z = np.array([np.interp(hf, hours, c) for c in curves])
        X, Yg = np.meshgrid(hf, Y)
        top = np.percentile(Z, 99.8)
        zlo = min(-0.15 * top, Z.min())
        zhi = 1.05 * top
        Zc = np.clip(Z, zlo, zhi)
        shade = LightSource(azdeg = 290, altdeg = 50).hillshade(Zc, vert_exag = 0.01, dx = 1, dy = 1)
        rgb = CMAP(Normalize(zlo, zhi)(Zc))
        rgb[..., :3] *= (0.88 + 0.12 * shade)[..., None]
        ax.plot_surface(X, Yg, Zc, facecolors = rgb, rstride = 1, cstride = 1,
                        linewidth = 0, antialiased = False, shade = False, rasterized = True, zorder = 2)
        MESH = (0.05, 0.2, 0.1)
        for r in np.linspace(0, len(Y) - 1, 15).round().astype(int):
            ax.plot(hf, np.full_like(hf, Y[r]), Zc[r], color = MESH, lw = 0.2, alpha = 0.35, zorder = 3)
        for b in np.searchsorted(hf, np.arange(13.75, 16, 0.25)):
            ax.plot(np.full(len(Y), hf[b]), Y, Zc[:, b], color = MESH, lw = 0.2, alpha = 0.35, zorder = 3)
        for a in (ax.xaxis, ax.yaxis, ax.zaxis):
            a.pane.fill = False
            a.pane.set_edgecolor((1, 1, 1, 0))
            a._axinfo['grid'].update(linewidth = 0)
        x0, x1, y0, y1 = 13.5, hours[-1], Y.min(), Y.max()
        corners = {(i, j, k): (x0 if i == 0 else x1, y0 if j == 0 else y1, zlo if k == 0 else zhi)
                   for i in (0, 1) for j in (0, 1) for k in (0, 1)}
        for a in corners:
            for b in corners:
                if sum(abs(np.array(a) - np.array(b))) == 1 and a < b:
                    hidden = (1, 1, 0) in (a, b)
                    ax.plot(*zip(corners[a], corners[b]), color = '#555555', lw = 0.5,
                            ls = ':' if hidden else '-')
        ax.zaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
        ax.tick_params(labelsize = 7, pad = 0)
        ax.text2D(-0.085, 0.55, r'Mean $\Delta v$ per bar ($\times10^{-4}$)', transform = ax.transAxes,
                  rotation = 90, fontsize = 8, ha = 'center', va = 'center')
        ax.set_box_aspect((1.25, 1.25, 0.9))
        ax.grid(False)
        ax.set_xlim(13.5, hours[-1])
        ax.set_ylim(Y.min(), Y.max())
        ax.set_zlim(zlo, zhi)
        kind = 'ETF' if s in ('spy', 'qqq') else 'Index'
        ax.text2D(0.5, 0.97, rf'$\mathbf{{{s.upper()}}}$  ·  {kind}', transform = ax.transAxes, fontsize = 11, ha = 'center', va = 'bottom')
        ax.set_xticks([14, 14.5, 15, 15.5])
        ax.set_xticklabels(['14:00', '14:30', '15:00', '15:30'])
        ax.yaxis.set_major_locator(mdates.YearLocator())
        ax.yaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.set_xlabel('Time of day (ET)', fontsize = 8, labelpad = 0)
        ax.set_ylabel('Session', fontsize = 8, labelpad = 0)
        ax.view_init(elev = 28, azim = -125)
    fig.subplots_adjust(left = 0.07, right = 0.99, bottom = 0.05, top = 0.99, wspace = 0.18, hspace = 0.16)
    plt.savefig('fig_vega_curve_3d.pdf', bbox_inches = 'tight', dpi = 300)
    plt.close()


if __name__ == '__main__':
    fig_example_day()
    fig_shares_tod()
    fig_vega_curve_3d()
