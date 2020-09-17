import numpy as np


def ballisticsODE(y, v, T, D, M, g):
    if v < 0:
        x = -1
    else:
        x = 1
    dvdt = (T - x * D * (v ** 2)) / M - g
    dydt = v
    return dydt, dvdt
