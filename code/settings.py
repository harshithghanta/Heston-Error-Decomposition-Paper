import math
import numpy as np

KAPPA = 3
EPSILON = 1
RHO = -0.7
R = 0.045
ETA = 0.02
YEARLY_MIN = 102060
ONE_DAY = 380/YEARLY_MIN
TAU_OFF = 9/YEARLY_MIN
UMAX = 10000
NPOINTS = 8192
GREEKS = ['gamma', 'volga', 'vanna', 'vega', 'charm', 'veta', 'tconv']
HALF_DAYS = ['2022-11-25', '2023-07-03', '2023-11-24', '2024-07-03', '2024-11-29',
             '2024-12-24', '2025-07-03', '2025-11-28', '2025-12-24']

ABS_NORMAL_MEAN = np.sqrt(2/np.pi)
ABS_NORMAL_MOMENT = 2**(2/3) * math.gamma(7/6) / math.gamma(1/2) 
