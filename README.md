# Where Does the Hedging Error Go?
### A Greek-by-Greek Decomposition for At-the-Money 0DTE Options under Heston

This paper explores 0DTE options and their errors by decomposing them under Bergomi's compensated-Greek framework. By testing across SPX and NDX (European-style options) along with SPY and QQQ (American-style options), it has been discovered that Gamma and Vega are the leading causes of hedging error.

**[Read the paper (PDF)](paper1.pdf)** · DOI: [10.5281/zenodo.23271888](https://doi.org/10.5281/zenodo.23271888)

## Overview
This paper delta-hedges (and also tests a Vega hedge) an ATM 0DTE call option every 5m (other intervals have been tested as well) under the Heston model. The decomposition splits the hedging error into 7 discrete Greeks: Gamma, Vega, Vanna, Volga (Vomma), Charm, Veta, and Time Convexity (TConv), building on Bergomi's compensated-Greek decomposition structure, adding the time Greeks that matter in fast-moving markets (such as 0DTE). Without the time Greeks, hedgers have a 14.6% unexplained residual that cannot be placed into any Greek category. With these Greeks added, the residual shrinks to around 5.5%. ε and ρ are calibrated daily to each morning's option smile.

The data consists of 1015 SPY sessions, 985 QQQ sessions, 1052 SPX sessions, and 997 NDX sessions (five-minute panels). Volatility comes from each contract's own variance strip.

We find that Gamma drives 66% of the error's variance and Vega drives 38%.

For understanding purposes, let's assume an orchestra with 7 players (in this instance, the Greeks). The gross share is simply how loud they play, with Gamma and Vega leading in loudness. On the other hand, the variance share is how much they play in harmony. The terms with a higher variance share act as the "conductors" to the orchestra, leading other terms, while Vomma acts rebellious: when the orchestra swells, Vomma plays soft, and vice versa. Mathematically, since Vomma pulls the other way, Gamma and Vega can pick up the slack, making their combined total above 100% to compensate for the "rebels".

The hedge loses 2.41 bp per session on average (t = −8.87; Newey–West HAC-adjusted). Most of the hedging error is expressed as a floor that plain faster hedging cannot improve. The floor is fit as the linear regression error = a + b√Δt. The constant portion (a) does not shrink as hedging gets faster, but the √Δt portion does. The floor represents 86% of SPY's one-minute error, 85% of SPX's, 82% of QQQ's, and 81% of NDX's. Even when hedging every 10 seconds, 93–98% of the one-minute error is left over.

A secondary instrument is required to hedge, and hedgers cannot rely on rote faster hedging alone.

## Files

**Paper**
| File | What it is |
|---|---|
| `paper1.tex` | The paper's source |
| `paper1.pdf` | The compiled paper |
| `fig_example_day.pdf` | Figure 1(a): one example day, realized P&L against the sum of the seven terms |
| `fig_shares_tod.pdf` | Figure 1(b): gross shares of Gamma, Vega and the other five terms by time of day |
| `fig_vega_curve_3d.pdf` | Figure 2: the past-only time-of-day Vega curve on all four products |

**`tables/`**
| File | Table in the paper |
|---|---|
| `tables_pnlsweep.tex` | P&L and error at each hedging interval |
| `tables_thirdorder.tex` | Residual with seven terms against eight (adding Speed) |
| `tables_clock.tex` | Three end-of-day conventions |
| `tables_staleness.tex` | How often each product's quotes go stale |
| `tables_summary4.tex` | The decomposition on each product's full sample |
| `tables_vegasplit.tex` | Vega split into drift and surprise |
| `tables_ssrn.tex` | All four products on shared days, and the error floor |
| `tables_ticks_ssrn.tex` | Tick test: 10-second against 1-minute hedging |
| `tables_stale.tex` | Results under each stale-quote rule |
| `tables_fill.tex` | Results when the two refilled days are changed or dropped |
| `tables_params.tex` | Main run against fixed Heston settings |
| `tables_volhedge.tex` | Hedging Vega with a second option |
| `tables_floorrobust.tex` | Error-floor robustness: block bootstrap, free exponent, market marks |

**`code/`**
| File | What it does |
|---|---|
| `settings.py` | Constants: Heston parameters, clock, integration grid, half-days |
| `load_data.py` | Reads a panel into per-day prices and variances |
| `heston_pricer.py` | Heston price and Greeks, the one-step decomposition, the hedging loop, the simulator |
| `statistical_methods.py` | Variance and gross shares, Newey–West, bootstraps, Ljung–Box, jump test, floor fit, tick ratios |
| `build_panels.py` | Turns the downloaded quotes into the 5- and 1-minute panels: quote carry, put–call parity forward, Cboe variance strip, forward screen, strike, half-days |
| `calibration.py` | Fits ε and ρ to each morning's option smile |
| `make_tables.py` | One function per table, plus the numbers quoted in the text |
| `make_figs.py` | Builds the three figures |

**`puller/`** (data download from Massive.com; from the original data pipeline, AI-assisted, not part of the code rewrite)
| File | What it does |
|---|---|
| `client.py` | Connects to the Massive.com API (reads the key from your own `.env`) |
| `pilot.py` | One-session test pull and the date and contract checks the full pull uses |
| `strip.py` | The Cboe variance-strip rule (reference version) |
| `fast.py` | Bins quotes into one-minute bars and builds the variance strip |
| `bigpull.py` | Downloads the SPY 0DTE quotes, day by day |
| `spxpull.py` | The same for SPX |
| `altpull.py` | The same for QQQ and NDX |

**Other**
| File | What it is |
|---|---|
| `README.md` | This file |
| `requirements.txt` | Python packages, pinned to the versions used for the paper |
| `LICENSE` | MIT license |
| `.gitignore` | Keeps the data and other local files out of the repository |

## Requirements
Python 3.14 with the pinned packages in `requirements.txt` (numpy, pandas, scipy, matplotlib, pyarrow for the quote files, requests for the puller):
```bash
python -m venv venv
venv/bin/pip install -r requirements.txt
```

## Data
The option quotes come from Massive.com. The hedged P&L marked at market quotes is correlated with the P&L marked at model prices (r = 0.989). The date range for all the option quotes is 2022-06-08 to 2026-08-25. The data itself (about 300 GB) is not included; `puller/` downloads it with your own Massive.com key, set as `MASSIVE_API_KEY` in your environment or a `.env` file:
```bash
python -m puller.bigpull --start 2022-06-08 --end 2026-08-25      # SPY
python -m puller.spxpull --start 2022-06-08 --end 2026-08-25      # SPX
python -m puller.altpull qqq --start 2022-06-08 --end 2026-08-25  # QQQ (or ndx)
```
`code/build_panels.py` then turns the downloads into the panels the analysis reads.

A panel is one CSV per product and hedging grid, with one row per hedge time:

| Column | What it holds |
|---|---|
| `timestamp` | the hedge time: every 5 minutes (or every minute) from 09:31 to 15:51 ET, to 12:51 on half-days |
| `und_close` | the underlying's forward price at that time, from put–call parity on the option quotes |
| `vix1d_close` | the volatility from that contract's own Cboe variance strip, in volatility points (v = (vix1d_close / 100)²) |
| `strike` | the strike of the call being hedged that day: the listed strike nearest the opening forward |

A full day has 77 rows on the 5-minute grid (381 on the 1-minute grid); a half-day has 41 (201). The name `vix1d_close` is left over from an earlier version.

## Reproducing the results
Run everything from the repository root, in this order.

**0. Output folders.** The scripts do not create them:
```bash
mkdir -p results_v2/panels results_v2/runs calibration
```

**1. Download** the quotes (see Data). Two stress days, 2025-04-08 and 2025-04-09, were downloaded a second time with a wider strike range (strikes within ±15% of the forward), because the normal pull missed far strikes those afternoons. They sit in `data_repull_wide/<root>/<day>/bars.parquet`, laid out like `data/`. `build_panels.py` reads those two days only from there and skips them without a warning if the files are missing. The wider pull was a one-off script that is not in this repository.

**2. Panels**, for each product (`spy`, `spx`, `qqq`, `ndx`) and grid (`5` or `1` minutes):
```bash
python code/build_panels.py spy 5      # writes results_v2/panels/his_spy5m.csv
```

**3. Daily fits** of ε and ρ, for each product (reads the 5-minute panel):
```bash
python code/calibration.py spy         # writes calibration/fits_spy.csv
```

**4. Runs and tables.** Each table is one function, run by name with its inputs after it:
```bash
python code/make_tables.py table_drifts
python code/make_tables.py table_summary4 spx
```
Some functions save their results and later functions read them back:

| Saved file | Made by | Read by |
|---|---|---|
| `results_v2/runs/his_main_run.npz` | the first table that needs the main SPY run | every main-run table |
| `results_v2/runs/his_1m_<sym>.npz` | `run_1m <sym>` | `table_xsec4`, `table_floor4`, `table_floorrobust`, `table_ticks4` |
| `results_v2/runs/his_1m_mkt_<sym>.npz` | `run_1m_market <sym>` | `table_floorrobust` |
| `results_v2/runs/his_ticks_<sym>_<k>.npy` | `run_ticks <sym>` | `table_ticks4` |

These saved files are not checked against the panels, fits or settings. **Delete them after changing any input**, or the tables will reuse the old results.

**5. Figures:**
```bash
python code/make_figs.py
```

**Inputs this repository does not build.** These were made by one-off scripts during the project and are needed only by the tables listed:

| File | Needed by |
|---|---|
| `data_repull_wide/` | `build_panels.py` (2025-04-08 and 2025-04-09) |
| `results_v2/runs/ticks360/panels/<sym>_tick10.csv` | `run_ticks` |
| `results_v2/runs/ticks360/stale_<sym>.csv` | `table_staleness` |
| `results_v2/panels/stale_main1015/panel_<sym>_<rule>.csv` | `table_stale` |
| `results_v2/runs/v2_calib/market_pnl.csv` | `text_numbers` (the market-marked numbers) |

| Paper table | Function in `make_tables.py` |
|---|---|
| Main results | `table_drifts` |
| Hedging interval (P&L) | `table_pnlsweep` |
| Hedging interval (attribution) | `table_freq` |
| Closing early | `table_close_early` |
| Speed term | `table_thirdorder` |
| End-of-day convention | `table_clock` |
| Parameter settings | `table_params` |
| Four products, full samples | `table_summary4 <sym>` |
| Four products, shared days | `run_1m <sym>` for all four, then `table_xsec4` |
| Error floor | `run_1m <sym>` for all four, then `table_floor4` |
| Error-floor robustness | `run_1m <sym>` and `run_1m_market <sym>` for all four, then `table_floorrobust` |
| Tick test | `run_ticks <sym>`, then `table_ticks4` |
| Vega drift and surprise | `table_vegasplit <sym>` |
| Volatility hedge | `table_volhedge <sym>` |
| Quote staleness | `table_staleness` |
| Stale-quote rules | `table_stale <sym>` |
| Filled marks | `table_fill` |
| Numbers quoted in the text | `text_numbers` |
| Simulation checks | `validation_sim` |

## Preprint
https://zenodo.org/records/23271889

## License
MIT
