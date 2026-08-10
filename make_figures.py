"""Regenerate the two figures in paper1.tex from the committed pipeline.

    python make_figures.py

Writes fig_shares.pdf, fig_example_day.pdf and fig_stats.json. Everything is
computed by importing paper1_implementation, so the figures cannot drift from
the tables: same pricer settings, same data file, same run_all.

Colours are one CVD-safe categorical pair, checked for protan/deutan/tritan
separation before use. Text stays in ink; the marks carry identity.
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import paper1_implementation as P

BLUE = "#2F6FBF"    # first categorical hue
AMBER = "#D97A1E"   # second categorical hue
INK = "#1a1a1a"
MUTED = "#6b6b6b"
GRID = "#d8d8d6"

PRETTY = {
    "gamma": "Gamma", "vega": "Vega", "vanna": "Vanna", "volga": "Volga",
    "charm": "Charm", "tconv": "Time conv.", "veta": "Veta",
}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
})


def load():
    dates, prices, variances = P.load_days(P.find_data_file())
    P.THETA = float(np.mean(np.concatenate(variances)))
    totals, pnls = P.run_all(prices, variances, 1)
    return dates, prices, variances, totals, pnls


def fig_shares(totals, pnls, path="fig_shares.pdf"):
    """Both attribution measures per term, with bootstrap intervals.

    Grouped horizontal bars: the job is comparing magnitudes across a short
    named list, and the variance measure is signed, so a zero line has to be
    readable. Two series, so a legend is always present.
    """
    gross = P.gross_shares(totals)
    varsh = P.variance_shares(totals, pnls)
    ci_gross, ci_var = P.bootstrap_shares(totals, pnls)

    order = sorted(P.TERM_NAMES, key=lambda n: gross[n])   # smallest at the bottom
    y = np.arange(len(order))
    h = 0.38

    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    for offset, (label, share, ci, colour) in enumerate([
            ("Variance share", varsh, ci_var, BLUE),
            ("Gross share", gross, ci_gross, AMBER)]):
        vals = np.array([share[n] for n in order])
        lo = np.array([ci[n][0] for n in order])
        hi = np.array([ci[n][1] for n in order])
        pos = y + (h / 2 if offset == 0 else -h / 2)
        ax.barh(pos, vals, height=h, color=colour, label=label,
                edgecolor="white", linewidth=0.8, zorder=3)
        ax.errorbar(vals, pos, xerr=[vals - lo, hi - vals], fmt="none",
                    ecolor=INK, elinewidth=0.9, capsize=2.0, zorder=4)

    ax.axvline(0, color=MUTED, linewidth=0.9, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels([PRETTY[n] for n in order])
    ax.set_xlabel("Share of the hedged P&L (%)")
    ax.xaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=8)

    # direct-label the two terms that carry the result, not every bar.
    # anchor past the interval's upper whisker so the text never sits on a cap.
    for n, share, ci, off in [("gamma", varsh, ci_var, h / 2),
                              ("gamma", gross, ci_gross, -h / 2),
                              ("vega", varsh, ci_var, h / 2),
                              ("vega", gross, ci_gross, -h / 2)]:
        i = order.index(n)
        ax.text(ci[n][1] + 1.8, y[i] + off, f"{share[n]:.1f}", va="center",
                ha="left", fontsize=8, color=INK)
    ax.set_xlim(right=max(ci_var["gamma"][1], ci_gross["gamma"][1]) + 9)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return gross, varsh


def fig_example_day(dates, prices, variances, day_index, path="fig_example_day.pdf"):
    """One session, bar by bar: does the decomposition track the realized error?

    Top panel is the reconstruction (realized vs summed terms). Bottom panel is
    the two terms that carry it. Same x-axis, no second y-scale anywhere.
    """
    S = prices[day_index]
    v = variances[day_index]
    nbars = len(S)
    last = nbars - 1

    cum_pnl, cum_terms = [0.0], {n: [0.0] for n in P.TERM_NAMES}
    running = {n: 0.0 for n in P.TERM_NAMES}
    total = 0.0
    for b in range(last):
        tau = P.ONE_DAY * (last - b) / last
        dt = P.ONE_DAY / last
        terms, pnl = P.one_step(S[b], v[b], S[b + 1], v[b + 1], tau, dt)
        total += pnl
        cum_pnl.append(total)
        for n in P.TERM_NAMES:
            running[n] += terms[n]
            cum_terms[n].append(running[n])

    cum_pnl = np.array(cum_pnl)
    cum_all = np.sum([cum_terms[n] for n in P.TERM_NAMES], axis=0)
    hours = 9.5 + np.arange(nbars) * 5.0 / 60.0

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.0, 4.4), sharex=True,
                                   gridspec_kw={"hspace": 0.18})

    ax1.plot(hours, 1e4 * cum_pnl, color=INK, linewidth=1.6,
             label="Realized hedged P&L", zorder=3)
    ax1.plot(hours, 1e4 * cum_all, color=BLUE, linewidth=1.6, linestyle=(0, (4, 2)),
             label="Sum of the seven terms", zorder=4)
    ax1.set_ylabel("Cumulative, $10^{-4}$")
    ax1.legend(loc="best", fontsize=8)

    ax2.plot(hours, 1e4 * np.array(cum_terms["gamma"]), color=BLUE, linewidth=1.6,
             label="Gamma term", zorder=3)
    ax2.plot(hours, 1e4 * np.array(cum_terms["vega"]), color=AMBER, linewidth=1.6,
             label="Vega term", zorder=3)
    ax2.set_ylabel("Cumulative, $10^{-4}$")
    ax2.set_xlabel("Time of day (ET)")
    ax2.legend(loc="best", fontsize=8)

    for ax in (ax1, ax2):
        ax.axhline(0, color=MUTED, linewidth=0.8, zorder=1)
        ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
    ax2.set_xticks([10, 11, 12, 13, 14, 15, 16])
    ax2.set_xticklabels(["10:00", "11:00", "12:00", "13:00", "14:00", "15:00", "16:00"])

    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

    denom = np.mean(np.abs(cum_pnl[1:]))
    return {
        "example_day": dates[day_index],
        "example_day_final_pnl": float(cum_pnl[-1]),
        "example_day_final_terms": float(cum_all[-1]),
        "example_day_mean_abs_gap": float(np.mean(np.abs(cum_pnl - cum_all)) / denom),
        "example_day_corr": float(np.corrcoef(cum_pnl, cum_all)[0, 1]),
    }


def main():
    dates, prices, variances, totals, pnls = load()
    gross, varsh = fig_shares(totals, pnls)

    # the day the paper already names as the most turbulent in the sample
    day_index = dates.index("2026-06-05") if "2026-06-05" in dates else 0
    stats = fig_example_day(dates, prices, variances, day_index)

    stats.update({
        "gamma_gross": gross["gamma"], "gamma_variance": varsh["gamma"],
        "vega_gross": gross["vega"], "vega_variance": varsh["vega"],
        "unexplained_variance": varsh["residual"],
        "gross_residual": P.residual_pct(totals, pnls),
        "ndays": len(prices), "umax": P.UMAX, "npoints": P.NPOINTS,
    })
    with open("fig_stats.json", "w") as fh:
        json.dump(stats, fh, indent=1)
    print("wrote fig_shares.pdf, fig_example_day.pdf, fig_stats.json")
    for k, val in stats.items():
        print(f"  {k}: {val}")


if __name__ == "__main__":
    main()
