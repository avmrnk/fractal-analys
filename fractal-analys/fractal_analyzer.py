import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss, acf
from numpy.fft import fft
from sklearn.linear_model import LinearRegression
import warnings
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

warnings.filterwarnings("ignore")
plt.rcParams['figure.figsize'] = (12, 6)


class FractalAnalyzer:
    def __init__(self, series):
        self.original_series = self._prepare_series(series)
        self.series = self.original_series.copy()
        self.results = {
            'metadata': {},
            'stationarity': {},
            'seasonality': {},
            'fractal': {},
            'interpretation': []
        }

    def _prepare_series(self, series):
        """Підготовка ряду: видалення NA, перетворення у numpy array"""
        if isinstance(series, pd.Series) or isinstance(series, pd.DataFrame):
            series = series.dropna().values.flatten()
        elif isinstance(series, list):
            series = np.array(series)

        if len(series) < 50:
            warnings.warn(f"Дуже короткий ряд ({len(series)} точок). Результати можуть бути ненадійними.")
        return series

    def _hurst_rs_confidence(self, H, n):
        """Розрахунок довірчого інтервалу для R/S методу"""
        std = 1.0 / np.sqrt(n)
        ci_low = H - 1.96 * std
        ci_high = H + 1.96 * std
        return ci_low, ci_high

    def hurst_rs(self, min_window=100, max_window=None, step=100, plot=False):
        """Покращений R/S метод з корекцією для H>1"""
        N = len(self.series)
        if max_window is None:
            max_window = N // 2

        windows = np.arange(min_window, max_window, step)
        log_RS = []
        log_T = []

        for tau in windows:
            segments = [self.series[i:i + tau] for i in range(0, N - tau, tau)]
            rescaled_ranges = []

            for seg in segments:
                if len(seg) < 2:
                    continue

                X = seg - np.mean(seg)
                Z = np.cumsum(X)
                R = np.max(Z) - np.min(Z)
                S = np.std(seg, ddof=1)

                if S > 0:
                    rescaled_ranges.append(R / S)

            if rescaled_ranges:
                log_RS.append(np.log(np.mean(rescaled_ranges)))
                log_T.append(np.log(tau))

        if len(log_T) < 2:
            raise ValueError("Недостатньо даних для розрахунку H")

        model = LinearRegression()
        model.fit(np.array(log_T).reshape(-1, 1), log_RS)
        H = model.coef_[0]

        # Корекція для H>1
        if H > 1:
            H_corrected = 1 - (H - 1)
            self.results['interpretation'].append(
                "Увага: Raw H>1 (можлива сильна нестаціонарність). "
                "Скориговане значення використано для аналізу."
            )
            H = H_corrected

        ci_low, ci_high = self._hurst_rs_confidence(H, len(log_T))

        if plot:
            self._plot_rs_analysis(log_T, log_RS, model, H, ci_low, ci_high)

        self.results['fractal'].update({
            'Hurst_RS': H,
            'Hurst_RS_CI': (ci_low, ci_high),
            'Hurst_RS_raw': model.coef_[0]
        })
        return H, (ci_low, ci_high)

    def _plot_rs_analysis(self, log_T, log_RS, model, H, ci_low, ci_high):
        """Візуалізація R/S аналізу"""
        plt.figure(figsize=(12, 6))
        plt.scatter(log_T, log_RS, label='Спостереження')
        plt.plot(log_T, model.predict(np.array(log_T).reshape(-1, 1)),
                 color='red', label=f'Лінійна регресія (H={H:.3f})')
        plt.fill_between(log_T,
                         [x * ci_low / H for x in model.predict(np.array(log_T).reshape(-1, 1))],
                         [x * ci_high / H for x in model.predict(np.array(log_T).reshape(-1, 1))],
                         color='red', alpha=0.2, label='95% ДІ')
        plt.xlabel('log(τ)')
        plt.ylabel('log(R/S)')
        plt.title('R/S Аналіз: log(R/S) vs log(τ)')
        plt.legend()
        plt.grid(True)
        plt.show()

    def _gph_optimal_m(self, method='elbow'):
        """Визначення оптимального m для GPH методу"""
        N = len(self.series)
        if method == 'sqrt':
            return int(np.sqrt(N))
        elif method == 'elbow':
            spectrum = np.abs(fft(self.series - np.mean(self.series))) ** 2
            spectrum = spectrum[1:N // 2]
            diff = np.diff(spectrum)
            m = np.argmax(diff < np.mean(diff)) + 1
            return max(10, min(m, N // 4))
        return int(N ** 0.6)


def gph_estimation(self, m=None, diff_order=None, plot=False):
    """Покращений GPH метод з урахуванням диференціювання"""
    working_series = np.diff(self.series, n=diff_order) if diff_order else self.series
    N = len(working_series)

    if m is None:
        m = self._gph_optimal_m()
    m = min(m, N // 2)

    detrended = working_series - np.mean(working_series)
    spectrum = np.abs(fft(detrended)) ** 2
    freqs = 2 * np.pi * np.arange(1, N // 2 + 1) / N

    freqs = freqs[:m]
    spectrum = spectrum[1:m + 1]
    spectrum = np.maximum(spectrum, 1e-12)

    log_freqs = np.log(4 * (np.sin(freqs / 2)) ** 2)
    log_spec = np.log(spectrum)

    weights = 1 / np.arange(1, m + 1)
    model = LinearRegression()
    model.fit(log_freqs.reshape(-1, 1), log_spec, sample_weight=weights)
    d = -model.coef_[0] / 2

    residuals = log_spec - model.predict(log_freqs.reshape(-1, 1))
    se = np.sqrt(np.sum(residuals ** 2) / (m - 2)) / np.sqrt(np.sum((log_freqs - np.mean(log_freqs)) ** 2))
    ci_low = d - 1.96 * se
    ci_high = d + 1.96 * se

    if plot:
        self._plot_gph_analysis(log_freqs, log_spec, model, d, se)

    self.results['fractal'].update({
        'GPH_d': d,
        'GPH_d_CI': (ci_low, ci_high),
        'Fractal_dimension': 2 - d,
        'GPH_diff_order': diff_order if diff_order else 0
    })
    return d, (ci_low, ci_high)


def _plot_gph_analysis(self, log_freqs, log_spec, model, d, se):
    """Візуалізація GPH аналізу"""
    plt.figure(figsize=(12, 6))
    plt.scatter(log_freqs, log_spec, label='Спостереження')
    plt.plot(log_freqs, model.predict(log_freqs.reshape(-1, 1)),
             color='red', label=f'Лінійна регресія (d={d:.3f})')
    plt.fill_between(log_freqs,
                     model.predict(log_freqs.reshape(-1, 1)) - 1.96 * se,
                     model.predict(log_freqs.reshape(-1, 1)) + 1.96 * se,
                     color='red', alpha=0.2, label='95% ДІ')
    plt.xlabel('log(4sin²(ω/2))')
    plt.ylabel('log(I(ω))')
    plt.title('GPH Оцінка: log(Періодограма) vs log(Частота)')
    plt.legend()
    plt.grid(True)
    plt.show()


def stationarity_tests(self, max_diff=3):
    """Розширена перевірка стаціонарності з автоматичним диференціюванням"""
    diff_order = 0
    series_to_test = self.series.copy()

    for i in range(max_diff + 1):
        adf_result = adfuller(series_to_test)
        try:
            kpss_result = kpss(series_to_test, regression='c')
        except:
            kpss_result = (np.nan, np.nan)

        is_stationary = (adf_result[1] <= 0.05) and (kpss_result[1] > 0.05)

        self.results['stationarity'][f'Diff_{i}'] = {
            'ADF_statistic': adf_result[0],
            'ADF_pvalue': adf_result[1],
            'KPSS_statistic': kpss_result[0],
            'KPSS_pvalue': kpss_result[1],
            'is_stationary': is_stationary,
            'series_length': len(series_to_test)
        }

        if is_stationary:
            self.results['metadata']['optimal_diff_order'] = i
            if i > 0:
                self.results['interpretation'].append(
                    f"Для досягнення стаціонарності потрібне диференціювання порядку {i}"
                )
            break

        series_to_test = np.diff(series_to_test)
        diff_order += 1

    if not self.results['stationarity'][f'Diff_{i}']['is_stationary']:
        self.results['interpretation'].append(
            "Увага: Не вдалося досягти стаціонарності при максимальному порядку диференціювання"
        )

    return self.results['stationarity']


def analyze_seasonality(self, max_lag=24):
    """Аналіз сезонності з ACF"""
    acf_values = acf(self.series, nlags=max_lag, fft=True)
    significant_lags = np.where(np.abs(acf_values[1:]) > 1.96 / np.sqrt(len(self.series)))[0] + 1

    if len(significant_lags) > 0:
        seasonality_period = significant_lags[np.argmax(acf_values[significant_lags])]
        strength = acf_values[seasonality_period]
    else:
        seasonality_period = None
        strength = 0

    self.results['seasonality'] = {
        'period': seasonality_period,
        'strength': strength,
        'ACF_values': acf_values[1:max_lag + 1]
    }

    return seasonality_period, strength


def full_analysis(self, make_stationary=True, plot=True):
    """Повний фрактальний аналіз з автоматичним диференціюванням"""
    self.results['metadata'].update({
        'series_length': len(self.series),
        'mean': np.mean(self.series),
        'std': np.std(self.series, ddof=1),
        'min': np.min(self.series),
        'max': np.max(self.series)
    })

    self.stationarity_tests()

    if make_stationary and 'optimal_diff_order' in self.results['metadata']:
        diff_order = self.results['metadata']['optimal_diff_order']
        if diff_order > 0:
            self.series = np.diff(self.series, n=diff_order)
            self.results['interpretation'].append(
                f"Для аналізу використано ряд, диференційований {diff_order} разів"
            )

    self.analyze_seasonality()
    self.hurst_rs(plot=plot)
    optimal_diff = self.results['metadata'].get('optimal_diff_order', 0)
    self.gph_estimation(diff_order=optimal_diff, plot=plot)

    # Застереження: compute_Hurst видалено (не підтримується у вашій версії)
    # try:
    #     H_alt = compute_Hurst(self.original_series, kind='price')
    #     self.results['fractal']['Hurst_alt'] = H_alt
    # except:
    #     pass

    if plot:
        self._plot_series()
        self._plot_acf_pacf()

    self._interpret_results()
    return self.results


def _plot_series(self):
    """Візуалізація часового ряду"""
    plt.figure(figsize=(14, 6))
    plt.plot(self.original_series, label='Оригінальний ряд', alpha=0.7)
    if len(self.series) != len(self.original_series):
        plt.plot(
            np.arange(len(self.original_series) - len(self.series), len(self.original_series)),
            self.series,
            label='Перетворений ряд',
            linewidth=2
        )
    plt.title('Часовий ряд')
    plt.xlabel('Індекс')
    plt.ylabel('Значення')
    plt.legend()
    plt.grid(True)
    plt.show()


def _plot_acf_pacf(self, lags=40):
    """Візуалізація ACF/PACF"""
    fig, ax = plt.subplots(2, 1, figsize=(12, 8))
    plot_acf(self.series, lags=lags, ax=ax[0], title='Автокореляція (ACF)')
    plot_pacf(self.series, lags=lags, ax=ax[1], title='Часткова автокореляція (PACF)')
    for a in ax:
        a.axhline(y=1.96 / np.sqrt(len(self.series)), linestyle='--', color='gray')
        a.axhline(y=-1.96 / np.sqrt(len(self.series)), linestyle='--', color='gray')
    plt.tight_layout()
    plt.show()


def _interpret_results(self):
    """Автоматична інтерпретація результатів"""
    H = self.results['fractal'].get('Hurst_RS', np.nan)
    d = self.results['fractal'].get('GPH_d', np.nan)

    if not np.isnan(H):
        if H < 0.5:
            interp = f"H={H:.3f} < 0.5: Антиперсистентність (mean-reverting процес)"
        elif H > 0.5:
            interp = f"H={H:.3f} > 0.5: Персистентність (трендова поведінка)"
        else:
            interp = f"H≈0.5: Випадкове блукання (Brownian motion)"

        if H > 0.9 or H < 0.1:
            interp += " (екстремальне значення - перевірте якість даних)"
        self.results['interpretation'].insert(0, interp)

    if not np.isnan(d):
        if d < 0:
            interp = f"d={d:.3f} < 0: Можливе over-differencing"
        elif d > 0.5:
            interp = f"d={d:.3f} > 0.5: Нестаціонарний ряд (можливо потрібне додаткове диференціювання)"
        elif d > 0:
            interp = f"d={d:.3f} ∈ (0,0.5): Дробова інтеграція (long memory)"
        else:
            interp = "d≈0: Коротка пам'ять"
        self.results['interpretation'].insert(1, interp)

    optimal_diff = self.results['metadata'].get('optimal_diff_order', -1)
    if optimal_diff == 0:
        self.results['interpretation'].append("Ряд є стаціонарним без диференціювання")
    elif optimal_diff > 0:
        self.results['interpretation'].append(
            f"Ряд став стаціонарним після {optimal_diff} диференціювань"
        )
    else:
        self.results['interpretation'].append(
            "Увага: Ряд не став стаціонарним після максимального диференціювання"
        )

    season_strength = self.results['seasonality'].get('strength', 0)
    if season_strength > 0.5:
        self.results['interpretation'].append(
            f"Виявлено сильну сезонність (період {self.results['seasonality']['period']})"
        )
    elif season_strength > 0.3:
        self.results['interpretation'].append("Можлива слабка сезонність")
