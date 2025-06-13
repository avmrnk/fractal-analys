import os
import sys
import logging
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # Matplotlib для побудови графіків

from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

from fractal_analyzer import FractalAnalyzer
from utils.data_preprocessing import load_time_series
from models.arima_sarimax_module import (
    plot_acf_pacf_for_series,
    plot_arima_diagnostics
)
from models.arfima_module import (
    get_weights_ffd,
    optimized_frac_diff_ffd,
    fit_arfima,
    forecast_arfima,
    estimate_d_gph
)


def interactive_forecast_plot(train, test, forecast, title=""):
    """
    Інтерактивний графік прогнозу.
    Якщо встановлено plotly, будуємо інтерактивний графік, інакше — matplotlib.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        # fallback на Matplotlib
        plt.figure(figsize=(12, 5))
        plt.plot(train.index, train, label="Train", color="gray", alpha=0.6)
        plt.plot(test.index, test, label="Test (actual)", color="red", alpha=0.8)
        plt.plot(forecast.index, forecast, label="Forecast", color="blue", linestyle="--")
        plt.title(title.replace("<br>", "\n"))
        plt.xlabel("time_index")
        plt.ylabel("value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()  # ← Обов’язковий виклик plt.show()
        return

    # Якщо plotly доступний, створюємо інтерактивний графік
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=train.index, y=train, name="Train", line=dict(color='gray', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=test.index, y=test, name="Test (actual)", line=dict(color='red', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=forecast.index, y=forecast, name="Forecast",
        line=dict(color='blue', width=2, dash='dot')
    ))
    fig.update_layout(
        title=title, xaxis_title="time_index", yaxis_title="value",
        hovermode="x unified", template="plotly_white"
    )
    fig.show()


def select_arima_params(train_series, d, seasonal_period=None,
                        max_p=2, max_q=2, max_P=1, max_Q=1):
    """
    Автоматичний підбір параметрів ARIMA/SARIMA за мінімальним AIC.
    Якщо seasonal_period заданий (>1), перебираємо p,q ∈ [0..max_p]×[0..max_q]
    та P,Q ∈ [0..max_P]×[0..max_Q], D=1.
    Якщо сезонності немає, просто ARIMA без сезонності.
    Повертає (best_order, best_seasonal_order).
    """
    best_aic = np.inf
    best_order = (0, d, 0)
    best_seasonal = (0, 0, 0, 0)

    if seasonal_period and seasonal_period > 1:
        D = 1
        total_runs = (max_p + 1) * (max_q + 1) * (max_P + 1) * (max_Q + 1)
        run_count = 0
        for p in range(max_p + 1):
            for q in range(max_q + 1):
                for P in range(max_P + 1):
                    for Q in range(max_Q + 1):
                        run_count += 1
                        try:
                            model = SARIMAX(
                                train_series,
                                order=(p, d, q),
                                seasonal_order=(P, D, Q, seasonal_period),
                                trend='c',
                                enforce_stationarity=False,
                                enforce_invertibility=False
                            )
                            res = model.fit(disp=False)
                            logging.info(
                                f"[{run_count}/{total_runs}] ARIMA({p},{d},{q})×({P},{D},{Q},{seasonal_period}) → AIC={res.aic:.2f}"
                            )
                            if res.aic < best_aic:
                                best_aic = res.aic
                                best_order = (p, d, q)
                                best_seasonal = (P, D, Q, seasonal_period)
                        except Exception as e:
                            logging.debug(
                                f"  Пропущено ARIMA({p},{d},{q})×({P},{D},{Q},{seasonal_period}) через: {e}"
                            )
                            continue
    else:
        total_runs = (max_p + 1) * (max_q + 1)
        run_count = 0
        for p in range(max_p + 1):
            for q in range(max_q + 1):
                run_count += 1
                try:
                    model = SARIMAX(
                        train_series,
                        order=(p, d, q),
                        seasonal_order=(0, 0, 0, 0),
                        trend='c',
                        enforce_stationarity=False,
                        enforce_invertibility=False
                    )
                    res = model.fit(disp=False)
                    logging.info(f"[{run_count}/{total_runs}] ARIMA({p},{d},{q}) → AIC={res.aic:.2f}")
                    if res.aic < best_aic:
                        best_aic = res.aic
                        best_order = (p, d, q)
                        best_seasonal = (0, 0, 0, 0)
                except Exception as e:
                    logging.debug(f"  Пропущено ARIMA({p},{d},{q}) через: {e}")
                    continue

    return best_order, best_seasonal


def improved_estimate_d(series, methods=('gph', 'rs'), verbose=False):
    """
    Комбінована оцінка дробового параметра d:
      - через GPH (estimate_d_gph), якщо доступний
      - через R/S (Hurst), звідки d = H - 0.5
    Якщо жоден метод не спрацював – повертає 0.0.
    """
    estimates = []

    if 'gph' in methods:
        try:
            d_gph, ci_gph = estimate_d_gph(series)
            if d_gph is not None:
                estimates.append(d_gph)
                if verbose:
                    print(f"[GPH] d={d_gph:.3f}, CI={ci_gph}")
        except Exception:
            if verbose:
                print("[GPH] пропускаємо: gph недоступний")

    if 'rs' in methods:
        analyzer = FractalAnalyzer(series)
        H, _ = analyzer.hurst_rs(plot=False)
        d_rs = H - 0.5
        estimates.append(d_rs)
        if verbose:
            print(f"[R/S] H={H:.3f} → d={d_rs:.3f}")

    if not estimates:
        return 0.0

    d_avg = np.mean(estimates)
    return float(max(0.0, min(0.5, d_avg)))


def evaluate_arfima_on_validation(train_series, val_series, d, pfd, qfd):
    """
    Внутрішня валідація ARFIMA(d, pfd, qfd):
      1) оптимізоване дробове диференціювання(train_series, d)
      2) фіт ARMA(pfd,qfd) на дробово-диференційованому ряді
      3) прогноз FD на len(val_series)
      4) інверсія прогнозу FD → звичайний рівень
      5) обчислення MAPE на валідації
    Повертає MAPE (%) або np.inf, якщо щось пішло не так.
    """
    train_fd = optimized_frac_diff_ffd(train_series, d, threshold=1e-7).dropna()
    if len(train_fd) < max(pfd, qfd) + 5:
        return np.inf

    try:
        arma_model = SARIMAX(
            train_fd,
            order=(pfd, 0, qfd),
            seasonal_order=(0, 0, 0, 0),
            trend='n',
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit(disp=False)
    except:
        return np.inf

    n_val = len(val_series)
    pred_fd = arma_model.forecast(steps=n_val)

    # Інверсія дробового диференціювання (вручну):
    x_last = train_series.iloc[-1]
    w = get_weights_ffd(d, threshold=1e-7, max_terms=1000)
    recon = []
    for i in range(n_val):
        val_rec = x_last
        for k in range(1, len(w)):
            if i - (k - 1) < 0:
                break
            val_rec += w[k] * pred_fd.iloc[i - (k - 1)]
        recon.append(val_rec)
    recon = pd.Series(recon, index=val_series.index)

    mape = np.mean(np.abs((val_series - recon) / val_series)) * 100
    return mape


def select_best_arfima(train, test):
    """
    Deep Search ARFIMA:
      - Розбиваємо train на fit_train (87.5%) та val (12.5%)
      - Перебираємо d ∈ {0.0,0.1,0.2,0.3}, pfd,qfd ∈ {0,1,2}
      - Оцінюємо MAPE на val
      - Вибираємо найменший MAPE, потім фітимо ARFIMA(d*,p*,q*) на весь train
      - Прогноз на тест, розрахунок фінальних метрик
    Повертає (best_config, best_forecast, (mse, mae, mape)).
    """
    n_train = len(train)
    n_val = int(n_train * 0.125)
    n_fit = n_train - n_val

    fit_train = train.iloc[:n_fit]
    val = train.iloc[n_fit:]
    best_mape = np.inf
    best_config = None

    logging.info("=== ARFIMA Deep Search: початок ===")
    for d in [0.0, 0.1, 0.2, 0.3]:
        for pfd in range(0, 3):
            for qfd in range(0, 3):
                mape_val = evaluate_arfima_on_validation(fit_train, val, d, pfd, qfd)
                logging.info(f"  d={d:.1f}, pfd={pfd}, qfd={qfd}, MAPE_val={mape_val:.2f}%")
                if mape_val < best_mape:
                    best_mape = mape_val
                    best_config = (d, pfd, qfd)

    if best_config is None:
        best_config = (0.0, 0, 0)
        best_mape = np.inf

    d_best, pfd_best, qfd_best = best_config
    logging.info(f"Вибрано ARFIMA: d={d_best:.2f}, pfd={pfd_best}, qfd={qfd_best}, MAPE_val={best_mape:.2f}%")

    arfima_results = fit_arfima(
        train,
        p=pfd_best, q=qfd_best,
        d=d_best,
        threshold=1e-7,
        max_terms=1000,
        auto_arima=False,
        verbose=False
    )
    arfima_results['x_last'] = train.iloc[-1]

    n_test = len(test)
    best_forecast = forecast_arfima(arfima_results, steps=n_test)
    best_forecast.index = test.index

    mse_test = mean_squared_error(test, best_forecast)
    mae_test = mean_absolute_error(test, best_forecast)
    mape_test = np.mean(np.abs((test - best_forecast) / test)) * 100

    logging.info(f"ARFIMA тест: MSE={mse_test:.4f}, MAE={mae_test:.4f}, MAPE={mape_test:.2f}%")
    return best_config, best_forecast, (mse_test, mae_test, mape_test)


def select_file_and_run():
    """
    Основний пайплайн:
      1) Завантажити CSV/Excel (колонки relative_time,value)
      2) Перетворити індекс на рівномірний (0,1,2,…)
      3) Фрактальний аналіз
      4) Автовибір ARIMA vs SARIMA (за MSE)
      5) Діагностика залишків
      6) Сортуємо ARIMA за двома видами прогнозу (статичний vs rolling) і залишаємо тільки кращий
      7) Deep Search ARFIMA (за запитом) та його візуалізація
    """
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S"
    )

    logging.info("---- Старт аналізу ----")
    print("CWD:", os.getcwd())
    print("sys.path:", sys.path[:3], "...")

    # 1) Вибір і завантаження файлу
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Оберіть CSV/Excel з колонками relative_time,value",
        filetypes=[("CSV", "*.csv"), ("Excel", "*.xlsx"), ("Усі файли", "*.*")]
    )
    if not file_path:
        logging.info("Файл не вибрано. Вихід.")
        return

    try:
        raw = load_time_series(file_path, time_col='relative_time', value_col='value')
    except Exception as e:
        messagebox.showerror("Помилка", f"Не вдалося прочитати файл:\n{str(e)}")
        return

    if len(raw) < 50:
        if not messagebox.askokcancel(
                "Попередження",
                f"Усього {len(raw)} точок. Результати можуть бути ненадійними. Продовжити?"
        ):
            return

    # 2) Рівномірний індекс (0,1,2,…), щоб SARIMAX бачив рівні проміжки
    series = pd.Series(raw.values, index=np.arange(len(raw)))
    series.index.name = 'time_index'
    series.name = 'value'

    # 3) Фрактальний аналіз
    analyzer = FractalAnalyzer(series)
    fractal_results = analyzer.full_analysis(plot=False)
    logging.info("=== Фрактальний аналіз завершено ===")
    logging.info(f"Hurst (R/S): {fractal_results['fractal']['Hurst_RS']}")
    logging.info(f"GPH d: {fractal_results['fractal']['GPH_d']}")
    logging.info(f"Fractal dimension: {fractal_results['fractal']['Fractal_dimension']}")
    logging.info(f"Interpretation: {fractal_results['interpretation']}")
    print()

    # 4) Розбиття train/test (80%/20%)
    n_total = len(series)
    split_idx = int(n_total * 0.8)
    train = series.iloc[:split_idx]
    test = series.iloc[split_idx:]
    logging.info(f"Train length = {len(train)}, Test length = {len(test)}")
    print()

    # Визначаємо d_for_arima за фрактальним аналізом
    optimal_diff = fractal_results['metadata'].get('optimal_diff_order', 0)
    d_for_arima = optimal_diff
    logging.info(f"Використовуємо d = {d_for_arima} (з фрактального аналізу)\n")

    # 5) Побудова ACF/PACF перед автопідбором
    logging.info("Будуємо ACF/PACF для train без differencing")
    plot_acf_pacf_for_series(train, lags=50)
    plt.show()  # ← Обов’язковий виклик plt.show()

    train_diff1 = train.diff(d_for_arima).dropna()
    logging.info(f"Будуємо ACF/PACF для train.diff({d_for_arima}).dropna()")
    plot_acf_pacf_for_series(train_diff1, lags=50)
    plt.show()  # ← Обов’язковий виклик plt.show()

    # 6) ARIMA без сезонності
    logging.info("=== ARIMA (без сезонності) ===")
    order_arima, seasonal_arima = select_arima_params(
        train, d=d_for_arima,
        seasonal_period=None,
        max_p=2, max_q=2
    )
    logging.info(f"  Обрано ARIMA order={order_arima}, seasonal_order={seasonal_arima}")

    model_arima = SARIMAX(
        train,
        order=order_arima,
        seasonal_order=(0, 0, 0, 0),
        trend='c',
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit(disp=False)
    arima_pred = model_arima.get_forecast(steps=len(test)).predicted_mean
    arima_pred.index = test.index
    mse0 = mean_squared_error(test, arima_pred)
    mae0 = mean_absolute_error(test, arima_pred)
    mape0 = np.mean(np.abs((test - arima_pred) / test)) * 100
    logging.info(f"  ARIMA без сезонності → MSE={mse0:.4f}, MAE={mae0:.4f}, MAPE={mape0:.2f}%")
    # 7) SARIMA із потенційним сезоном (s із fractal_results['seasonality']['period'])
    s = fractal_results['seasonality'].get('period', None)
    use_sarima = False
    model_sarima = None
    sarima_pred = None
    mse1 = np.inf

    if s is not None and s > 1:
        logging.info(f"=== SARIMA (з сезоном s={s}) ===")
        order_sarima, seasonal_sarima = select_arima_params(
            train, d=d_for_arima,
            seasonal_period=s,
            max_p=2, max_q=2,
            max_P=1, max_Q=1
        )
        logging.info(f"  Обрано SARIMA order={order_sarima}, seasonal_order={seasonal_sarima}")

        model_sarima = SARIMAX(
            train,
            order=order_sarima,
            seasonal_order=seasonal_sarima,
            trend='c',
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit(disp=False)
        sarima_pred = model_sarima.get_forecast(steps=len(test)).predicted_mean
        sarima_pred.index = test.index
        mse1 = mean_squared_error(test, sarima_pred)
        mae1 = mean_absolute_error(test, sarima_pred)
        mape1 = np.mean(np.abs((test - sarima_pred) / test)) * 100
        logging.info(f"  SARIMA (s={s}) → MSE={mse1:.4f}, MAE={mae1:.4f}, MAPE={mape1:.2f}%")

        if mse1 < mse0:
            use_sarima = True
            static_pred = sarima_pred
            static_metrics = (mse1, mae1, mape1)
            logging.info("  Вибрано SARIMA як кращу (менший MSE).")
        else:
            static_pred = arima_pred
            static_metrics = (mse0, mae0, mape0)
            logging.info("  Вибрано ARIMA (без сезонності) як кращу (менший MSE).")
    else:
        static_pred = arima_pred
        static_metrics = (mse0, mae0, mape0)
        logging.info("Сезонність не виявлено або s ≤ 1, використовується ARIMA без сезонності.")

    # 8) Діагностичні графіки залишків остаточної моделі (ARIMA або SARIMA)
    used_model = model_sarima if use_sarima else model_arima
    logging.info("Будуємо діагностичні графіки (залишки, ACF залишків, Q-Q plot).")
    plot_arima_diagnostics(used_model)
    plt.show()  # ← Обов’язковий виклик plt.show()

    # 9) Rolling-forecast (оновлення кожним справжнім тестом)
    logging.info("=== Rolling-forecast (оновлення кожним справжнім тестом) ===")
    history = train.copy()
    preds_roll = []

    # Місце збереження порядків з моделі:
    arima_order_used = used_model.model.order
    seasonal_order_used = used_model.model.seasonal_order

    for t in range(len(test)):
        mod = SARIMAX(
            history,
            order=arima_order_used,
            seasonal_order=seasonal_order_used,
            trend='c',
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit(disp=False)
        yhat = mod.forecast(steps=1).iloc[0]
        preds_roll.append(yhat)

        # Додаємо новий фактичний тестовий рядок до history:
        new_point = pd.Series([test.iloc[t]])
        history = pd.concat([history, new_point], ignore_index=True)

    roll_pred = pd.Series(preds_roll, index=test.index)
    mse_roll = mean_squared_error(test, roll_pred)
    mae_roll = mean_absolute_error(test, roll_pred)
    mape_roll = np.mean(np.abs((test - roll_pred) / test)) * 100
    logging.info(f"Rolling-forecast: MSE={mse_roll:.4f}, MAE={mae_roll:.4f}, MAPE={mape_roll:.2f}%")

    # 10) Відбір «кращого» ARIMA-прогнозу: static vs rolling
    mse_static, mae_static, mape_static = static_metrics

    if mse_roll < mse_static:
        # Якщо rolling має менший MSE — малюємо і виводимо ТІЛЬКИ rolling
        title_roll = (
            f"Rolling-Forecast ({'SARIMA' if use_sarima else 'ARIMA'})\n"
            f"MSE={mse_roll:.3f}, MAE={mae_roll:.3f}, MAPE={mape_roll:.2f}%"
        )
        interactive_forecast_plot(train, test, roll_pred, title=title_roll)
        logging.info("Відобразжено лише Rolling-forecast як кращий (менший MSE).")
    else:
        # Інакше — малюємо і виводимо лише static прогноз
        title_static = (
            f"Статичний Forecast ({'SARIMA' if use_sarima else 'ARIMA'})\n"
            f"order={used_model.model.order}, seasonal_order={used_model.model.seasonal_order}\n"
            f"MSE={mse_static:.3f}, MAE={mae_static:.3f}, MAPE={mape_static:.2f}%"
        )
        interactive_forecast_plot(train, test, static_pred, title=title_static)
        logging.info("Відобразжено лише Статичний Forecast як кращий (менший MSE).")

    # 11) ARFIMA Deep Search (за запитом користувача)
    if messagebox.askyesno("ARFIMA", "Запустити ARFIMA-аналіз?"):
        logging.info("=== ARFIMA: початок Deep Search ===")

        d_choice = improved_estimate_d(train, methods=('gph', 'rs'), verbose=True)
        logging.info(f"Комбінована оцінка d = {d_choice:.4f}")

        best_cfg, best_forecast_ts, metrics = select_best_arfima(train, test)

        title_arfima = (
            f"ARFIMA: Train, Test + Forecast\n"
            f"Найкраща: d={best_cfg[0]:.2f}, pfd={best_cfg[1]}, qfd={best_cfg[2]}\n"
            f"MSE={metrics[0]:.3f}, MAE={metrics[1]:.3f}, MAPE={metrics[2]:.2f}%"
        )
        interactive_forecast_plot(train, test, best_forecast_ts, title=title_arfima)

    logging.info("=== Усі аналізи завершено ===")


if __name__ == "__main__":
    select_file_and_run()
