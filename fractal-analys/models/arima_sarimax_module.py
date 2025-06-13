# models/arima_sarimax_module.py

import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.graphics.gofplots import qqplot

def plot_acf_pacf_for_series(series, lags=40):
    """
    Створює два підграфіки: ACF та PACF для заданого series.
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    plot_acf(series, lags=lags, ax=axes[0], title='ACF')
    plot_pacf(series, lags=lags, ax=axes[1], title='PACF')
    plt.tight_layout()
    plt.show()


def plot_arima_diagnostics(model_results):
    """
    Малює діагностичні графіки для ARIMA/SARIMA: залишки, ACF залишків, Q-Q plot.
    """
    resid = model_results.resid
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # 1) Залишки
    axes[0].plot(resid)
    axes[0].set_title("Residuals")
    axes[0].axhline(0, linestyle='--', color='gray')

    # 2) ACF залишків
    plot_acf(resid, lags=40, ax=axes[1], title='ACF of Residuals')

    # 3) Q-Q plot
    qqplot(resid, line='s', ax=axes[2])
    axes[2].set_title("Q-Q Plot of Residuals")

    plt.tight_layout()
    plt.show()

