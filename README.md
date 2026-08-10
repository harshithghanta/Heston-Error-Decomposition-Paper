# Heston Error Decomposition for 0DTE Options

A Greek-by-Greek decomposition of the delta-hedging error on at-the-money 0DTE options under Heston,
measured at five-minute intervals on 57 trading days of SPY. The paper is `paper1.pdf`.

## Files

`paper1.pdf` / `paper1.tex` are the paper. `paper1_implementation.py` holds the pricer, the
decomposition, and every statistic the paper reports. `make_figures.py` regenerates both figures from
the same pipeline. `fetch_data.py` rebuilds `intraday.csv` from yfinance; Yahoo serves 5-minute data
for the trailing 60 trading days only, so re-running it moves the sample window forward.
`intraday.csv` is the sample, with columns `timestamp`, `und_close`, `vix1d_close`.

Needs numpy and pandas, plus yfinance to re-pull data. Run from the folder holding `intraday.csv`.

## Running it

```
python paper1_implementation.py            main results (a couple of minutes)
python paper1_implementation.py full       adds the terminal-interval ladder and the kappa and epsilon sweeps
python paper1_implementation.py detrend    removes VIX1D's intraday roll and reruns (section 4.3)
python paper1_implementation.py validate   pricer battery, including UMAX convergence in the final bar
python paper1_implementation.py mc [n]     Monte Carlo check of the decomposition algebra
python make_figures.py                     regenerates both figures
python fetch_data.py                       re-pulls the data
```

Every number and figure in the paper comes out of these.

## Results and caveats

Two measures are reported: the variance share, a signed decomposition of the variance of the realized
hedged P&L (Gamma 64.6%, Vega 31.6%, 5.6% unexplained), and the gross share, each term's average
absolute daily contribution (Gamma 50.4%, Vega 35.0%, on a 20.1% gross residual). Vega's gross share
roughly doubles from the calmest days to the most turbulent. The Heston parameters are uncalibrated
round numbers apart from eta, so read this as a proof of concept rather than a calibrated estimate.
VIX1D climbs mechanically through the session, averaging 10.9 at the open and 13.9 at the close,
because it measures time in business minutes against a constant maturity of 405 of them while Cboe's
near-term weight on today's expiring strip falls from 96.3% to 1.2% across the day. That roll
manufactures the drift in the Vega term; section 4.3 removes it and reruns, and the drift collapses
from t = +6.5 to +0.6 while the attribution shares barely move.
