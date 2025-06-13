import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.signal import periodogram
from scipy.optimize import curve_fit


def get_weights_ffd(d, threshold=1e-5, max_terms=1000):
    """
    Генерує ваги для дробового диференціювання (FFD).
    """
    w = [1.0]
    k = 1
    while True:
        w_k = -w[-1] * (d - k + 1) / k
        if abs(w_k) < threshold or k >= max_terms:
            break
        w.append(w_k)
        k += 1
    return np.array(w)


def optimized_frac_diff_ffd(series, d, threshold=1e-5):
    """
    Застосовує FFD до серії з вагою threshold.
    Повертає нову pd.Series з дробово диференційованими значеннями.
    """
    if not isinstance(series, (pd.Series, pd.DataFrame)):
        series = pd.Series(series)

    w = get_weights_ffd(d, threshold=threshold)
    n = len(series)
    k_max = len(w)

    diffed = np.zeros(n)
    diffed[:k_max - 1] = np.nan

    for i in range(k_max - 1, n):
        val = 0.0
        for k in range(k_max):
            if i - k < 0:
                break
            val += w[k] * series.iloc[i - k]
        diffed[i] = val

    return pd.Series(diffed, index=series.index)


def fit_arfima(train, p, q, d, threshold=1e-5, max_terms=1000):
    """
    Навчає ARFIMA(p,d,q) на train. Повертає словник з параметрами та тренованою моделлю.
    """
    train_fd = optimized_frac_diff_ffd(train, d, threshold=threshold).dropna()

    try:
        arma_model = SARIMAX(
            train_fd,
            order=(p, 0, q),
            seasonal_order=(0, 0, 0, 0),
            trend='n',
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit(disp=False)

        return {
            'd': d,
            'weights': get_weights_ffd(d, threshold, max_terms),
            'arma_model': arma_model,
            'x_last_window': train.iloc[-max_terms:].values
        }
    except:
        return None


def forecast_arfima(arfima_res, steps):
    """
    Реконструює прогноз із результатів fit_arfima на steps кроків уперед.
    Повертає pd.Series довжини steps.
    """
    if arfima_res is None:
        return None

    w = arfima_res['weights']
    arma_model = arfima_res['arma_model']
    last_window = arfima_res['x_last_window']

    # forecast in fractional-differenced space
    pred_fd = arma_model.forecast(steps=steps)
    recon = []

    for i in range(steps):
        val = 0.0
        for k in range(len(w)):
            if k == 0:
                val += w[k] * last_window[-1]
            elif i - k >= 0:
                val += w[k] * pred_fd.iloc[i - k]
            else:
                val += w[k] * last_window[-(k - i)]
        recon.append(val)

    return pd.Series(recon)


def select_best_arfima(train, test, d_values=None, p_values=(0, 1, 2), q_values=(0, 1, 2)):
    """
    Пошук найкращої комбінації (d,p,q) за мінімальним MSE.
    Повертає (best_cfg, best_forecast, best_metrics).
    """
    if d_values is None:
        d_values = np.arange(0.0, 0.6, 0.1)

    best_score = float('inf')
    best_cfg = None
    best_forecast = pd.Series([train.iloc[-1]] * len(test), index=test.index)
    best_metrics = (np.nan, np.nan, np.nan)

    for d in d_values:
        for p in p_values:
            for q in q_values:
                arfima_res = fit_arfima(train, p, q, d)
                if arfima_res is None:
                    continue

                yhat = forecast_arfima(arfima_res, steps=len(test))
                yhat.index = test.index

                mse = mean_squared_error(test, yhat)
                if mse < best_score:
                    mae = mean_absolute_error(test, yhat)
                    mape = np.mean(np.abs((test - yhat) / test)) * 100
                    best_score = mse
                    best_cfg = (d, p, q)
                    best_forecast = yhat.copy()
                    best_metrics = (mse, mae, mape)

    return best_cfg, best_forecast, best_metrics


def rolling_forecast_arfima(train, test, d, p, q, threshold=1e-5, max_terms=1000):
    """
    One-step rolling-forecast для ARFIMA(p,d,q):
      - Щоразу переобучаємо модель на всій історії + попередні прогнози,
      - Робимо прогноз на 1 крок вперед,
      - Додаємо справжнє значення в історію.
    Повертає (forecast_series, (mse, mae, mape)).
    """
    history = train.copy()
    preds = []
    for t in range(len(test)):
        # Fit ARFIMA на поточній історії
        res = fit_arfima(history, p, q, d, threshold=threshold, max_terms=max_terms)
        # One-step ahead forecast
        yhat = forecast_arfima(res, steps=1).iloc[0]
        preds.append(yhat)
        # Додаємо реальне значення з тесту
        history = pd.concat([history, pd.Series([test.iloc[t]], index=[test.index[t]])])

    forecast_series = pd.Series(preds, index=test.index)
    mse = mean_squared_error(test, forecast_series)
    mae = mean_absolute_error(test, forecast_series)
    mape = np.mean(np.abs((test - forecast_series) / test)) * 100
    return forecast_series, (mse, mae, mape)


def gph_estimate(series):
    """
    Оцінка параметра довгої пам'яті d методом Geweke-Porter-Hudak (GPH).
    """
    freqs, spectrum = periodogram(series, detrend='linear')
    freqs = freqs[1:]
    spectrum = spectrum[1:]
    n_freqs = len(freqs)

    low_freqs = freqs[:n_freqs // 4]
    low_spectrum = spectrum[:n_freqs // 4]

    log_freqs = np.log(low_freqs)
    log_spectrum = np.log(low_spectrum)

    def lin_func(x, a, b): return a * x + b

    params, _ = curve_fit(lin_func, log_freqs, log_spectrum)
    return -params[0] / 2
