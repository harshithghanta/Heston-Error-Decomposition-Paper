import pandas as pd
import settings


def load_fits(sym, dates):
    f = pd.read_csv(f'calibration/fits_{sym}.csv').set_index('date')
    return [(f.loc[d, 'eta'], f.loc[d, 'eps'], f.loc[d, 'rho']) for d in dates]


def load_days(path):
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index)
    df['date'] = df.index.date
    counts = df.groupby('date').size()
    full = int(counts.max())
    half = int((full-1) * 200 / 380) + 1
    dates = []
    prices = []
    variances = []

    for day, rows in df.groupby('date'):
        n = len(rows)
        if not (n == full or (n == half and str(day) in settings.HALF_DAYS)):
            continue
        dates.append(str(day))
        prices.append(rows['und_close'].to_numpy() / rows['strike'].iloc[0])
        variances.append((rows['vix1d_close'].to_numpy() / 100) ** 2)
    return dates, prices, variances
