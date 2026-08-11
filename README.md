# Heston Error Decomposition for 0DTE Options

[**Read the paper (PDF)**](paper1.pdf)

## Abstract

This paper decomposes the delta-hedging error on at-the-money 0DTE SPY options into Heston-model Greeks (gamma, vega, vanna, volga, charm, veta, and time convexity), measured at five-minute intervals across 57 trading days. Gamma and vega dominate both attribution measures reported: the variance share of realized hedged P&L (64.6% / 31.6%) and the average absolute gross share (50.4% / 35.0%), with vega's gross contribution roughly doubling on the most turbulent days. A companion check shows that most of the apparent drift in the vega term is a mechanical artifact of VIX1D's intraday roll, which mostly disappears once that roll is removed (Section 4.3).

## How to run

```
git clone https://github.com/harshithghanta/Heston-Error-Decomposition-Paper.git
cd Heston-Error-Decomposition-Paper
pip install -r requirements.txt
python paper1_implementation.py main
```

`python paper1_implementation.py main` reproduces the paper's headline result in a couple of minutes, reading directly from the committed `intraday.csv` (no network access needed).

Other entry points, all run from the repo root:

```
python paper1_implementation.py full       # adds the terminal-interval ladder and the kappa/epsilon sweeps
python paper1_implementation.py detrend    # removes VIX1D's intraday roll and reruns (Section 4.3)
python paper1_implementation.py validate   # pricer battery, including UMAX convergence in the final bar
python paper1_implementation.py mc [n]     # Monte Carlo check of the decomposition algebra
python make_figures.py                     # regenerates fig_shares.pdf and fig_example_day.pdf
python fetch_data.py                       # re-pulls fresh data (see Data provenance below)
```

## Data provenance

`intraday.csv` holds 5-minute SPY closing prices and Cboe VIX1D closing levels, pulled via `yfinance` and committed to this repo. The sample covers **2026-05-11 to 2026-07-31** (57 full trading days, 78 five-minute bars per day, 09:30-15:55 ET); every number and figure in the paper is generated from this exact file.

Running `fetch_data.py` again will not reproduce this window: Yahoo only serves 5-minute index data for the trailing 60 trading days, so a fresh pull moves the sample forward to whatever the most recent 60 days are at the time it's run. Use the committed `intraday.csv` to reproduce the paper's reported numbers exactly.

## Files

`paper1.pdf` / `paper1.tex` are the paper. `paper1_implementation.py` holds the pricer, the decomposition, and every statistic the paper reports. `make_figures.py` regenerates both figures from the same pipeline. `fetch_data.py` rebuilds `intraday.csv` from `yfinance` (see caveat above). `intraday.csv` is the committed sample, with columns `timestamp`, `und_close`, `vix1d_close`. `requirements.txt` lists the Python dependencies (numpy, pandas, yfinance, matplotlib).

## Results and caveats

Two measures are reported: the variance share, a signed decomposition of the variance of the realized hedged P&L (Gamma 64.6%, Vega 31.6%, 5.6% unexplained), and the gross share, each term's average absolute daily contribution (Gamma 50.4%, Vega 35.0%, on a 20.1% gross residual). Vega's gross share roughly doubles from the calmest days to the most turbulent. The Heston parameters are uncalibrated round numbers apart from eta, so read this as a proof of concept rather than a calibrated estimate. VIX1D climbs mechanically through the session, averaging 10.9 at the open and 13.9 at the close, because it measures time in business minutes against a constant maturity of 405 of them while Cboe's near-term weight on today's expiring strip falls from 96.3% to 1.2% across the day. That roll manufactures the drift in the Vega term; Section 4.3 removes it and reruns, and the drift collapses from t = +6.5 to +0.6 while the attribution shares barely move.
