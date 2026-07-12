## Intro
This is the Heston Decomposition Paper for Zero Days to Expiration (0DTE) Options. I decomposed the hedging error at five minute intervals 
and distilled my findings in this paper.

## Repository Guide 
`intraday.csv` is the file that holds all of the data used.

`paper1.pdf` is the visible pdf format of the paper (Use this to read the paper)

`paper1.tex` is the LaTeX script used to generate `paper1.pdf`

`paper1_implementation.py` is the Python Script used to error decompose the greeks

Please install pandas and numpy in your venv to run the program. Please ensure that `intraday.csv` is in the same folder when running. 

Command to run: `python paper1_implementation.py`

## Methodology & Limitations 
The full methodology section is found inside `paper1.pdf`. The following is a summary:
The Heston parameters are uncalibrated guesses due to a lack of better data. With better data, different results may occur. Currently, 
the data used is yfinance SPX data taken at 5 minute intervals starting from March 31st, 2026. There were a variety of market states 
during this time, making it a prime example of the proof of concept of 
Error Decomposition over different regimes. With that, Gamma dominated most of the error, with Vega 
following in second. Vega's error increases over regimes with increasing volatility. Volatility was measured with (VIX/100)^2. 
The derivation is also found inside `paper1.pdf`, but the base reasoning is that instead of taking the limit as ∆t → 0, making the model continuous, 
the limit is not taken, keeping the model discrete. There is 9% of error that is unattributed to any greek. 
