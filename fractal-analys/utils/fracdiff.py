import numpy as np
import pandas as pd


def get_weights_ffd(d, threshold=1e-5, max_terms=1000):
    """
    Рекурсивно обчислює ваги w_k^{(d)} = (k - 1 - d) / k * w_{k-1},
    поки |w_k| >= threshold або поки не досягнемо max_terms.
    Повертає numpy.array ваг [w_0=1, w_1, w_2, ..., w_K].
    """
    w = [1.0]
    k = 1
    while k < max_terms:
        w_k = - w[k - 1] * (d - k + 1) / k
        if abs(w_k) < threshold:
            break
        w.append(w_k)
        k += 1
    return np.array(w, dtype=float)


def frac_diff_ffd(series, weights):
    """
    Застосовує фракційне диференціювання з вагою weights до pandas.Series.
    Повертає pd.Series тієї самої довжини (NaN на тих індексах, де не можна порахувати).
    """
    if not isinstance(series, (pd.Series, pd.DataFrame)):
        series = pd.Series(series)

    T = len(series)
    diffed = [np.nan] * T
    for i in range(T):
        val = 0.0
        for k, w_k in enumerate(weights):
            if i - k < 0:
                break
            val += w_k * series.iloc[i - k]
        diffed[i] = val

    return pd.Series(diffed, index=series.index)
