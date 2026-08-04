## Intro
This is the Heston Decomposition Paper for Zero Days to Expiration (0DTE) Options. I decomposed the hedging error at five minute intervals 
and distilled my findings in this paper.

## Repository Guide 
`intraday.csv` is the file that holds all of the data used (columns: `timestamp`, `und_close`, `vix1d_close`).

`fetch_data.py` rebuilds `intraday.csv` from yfinance. Yahoo only serves 5-minute index data for the trailing 60 trading days, so re-running it moves the sample window forward.

`paper1.pdf` is the visible pdf format of the paper (Use this to read the paper)

`paper1.tex` is the LaTeX script used to generate `paper1.pdf`

`paper1_implementation.py` is the Python Script used to error decompose the greeks

Please install pandas and numpy in your venv to run the program (plus yfinance if you want to re-pull the data). Please ensure that `intraday.csv` is in the same folder when running. 

Command to run: `python paper1_implementation.py`

That prints everything in the paper except the sweeps and the terminal-interval ladder: the attribution on
both measures, bootstrap intervals, Newey-West standard errors, the bipower jump test and jump-free
subsample, both regime sorts, and the variance-proxy diagnostics. It takes a couple of minutes.

Other entry points:

- `python paper1_implementation.py full` — adds the terminal-interval ladder and the κ and ε sweeps (several minutes)
- `python paper1_implementation.py validate` — pricer battery: Black-Scholes limit, put-call parity, grid convergence, finite-difference bump stability
- `python paper1_implementation.py mc [npaths]` — Monte Carlo validation of the decomposition algebra on simulated Heston paths

Command to re-pull the data: `python fetch_data.py`

## Methodology & Limitations 
The full methodology section is found inside `paper1.pdf`. The following is a summary:
The Heston parameters are uncalibrated guesses due to a lack of better data. With better data, different results may occur. Currently, 
the data used is yfinance SPY data taken at 5 minute intervals, covering the 57 full trading days from May 11th, 2026 through July 31st, 2026. There were a variety of market states 
during this time, making it a prime example of the proof of concept of 
Error Decomposition over different regimes. With that, Gamma carried the largest share of the error, with Vega 
following in second and no longer far behind. Vega's error increases over regimes with increasing volatility. Volatility was measured with (VIX1D/100)^2 — VIX1D 
is the 1-day volatility index, which is horizon-matched to a 0DTE option, unlike the 30-day VIX. 
The derivation is also found inside `paper1.pdf`, but the base reasoning is that instead of taking the limit as ∆t → 0, making the model continuous, 
the limit is not taken, keeping the model discrete.

Results are reported on two measures. The variance share is a signed decomposition of the variance of the realized hedged P&L 
(Gamma 64.6%, Vega 31.5%, everything else near zero, 5.9% unexplained). The gross share is the average absolute size of each term's 
daily contribution (Gamma 50.0%, Vega 34.7%, 20.2% unattributed on the harsher gross residual metric). The baseline vol-of-vol is 
ε = 1.0; ε = 0.5 and 2.0 are reported as a sweep in the paper.

Caveat worth knowing before reading the results: VIX1D climbs mechanically through the session (10.9 at the open to 13.9 at the close 
on average, up on 55 of 57 days) because its 24-hour window rolls onto the overnight gap. That roll manufactures the drift in the Vega 
term, so the paper treats term-level drifts as proxy artifacts rather than as risk premia. 
