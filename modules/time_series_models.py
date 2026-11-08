"""
==============================================================================
Module:      time_series_models.py
Description: Time-series forecasting routines: linear OLS trend, rolling
             moving averages, and ARIMA modeling with ADF stationarity testing.
==============================================================================
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from typing import Tuple, Dict, Any


def forecast_linear_trend(sales: np.ndarray, target_month: int = 13) -> Tuple[float, float, float, float]:
    """Berechnet die lineare Trendregression fuer Zeitreihendaten und prognostiziert Folgemonate."""
    n = len(sales)
    x = np.arange(1, n + 1).reshape(-1, 1)
    y = np.asarray(sales, dtype=float)

    model = LinearRegression()
    model.fit(x, y)

    intercept = float(model.intercept_)
    slope = float(model.coef_[0])
    pred = float(model.predict(np.array([[target_month]]))[0])
    mse = float(mean_squared_error(y, model.predict(x)))

    return intercept, slope, pred, mse


def forecast_moving_average(sales: np.ndarray, window_size: int = 3) -> Tuple[np.ndarray, float]:
    """Berechnet geglaettete Werte ueber Faltung und prognostiziert Folgemonate."""
    y = np.asarray(sales, dtype=float)
    kernel = np.ones(window_size) / window_size
    smoothed = np.convolve(y, kernel, mode='valid')
    pred = float(np.mean(y[-window_size:]))
    return smoothed, pred


def fit_arima_forecast(sales: np.ndarray, order: Tuple[int, int, int] = (1, 1, 1)) -> Dict[str, Any]:
    """Prueft Stationaritaet via ADF-Test und trainiert ein ARIMA-Modell zur Einzelschrittprognose."""
    y = np.asarray(sales, dtype=float)
    adf_result = adfuller(y)
    adf_stat = float(adf_result[0])
    p_val = float(adf_result[1])

    model = ARIMA(y, order=order)
    model_fit = model.fit()
    forecast_val = float(model_fit.forecast(steps=1)[0])

    return {
        "adf_statistic": adf_stat,
        "p_value": p_val,
        "is_stationary": p_val <= 0.05,
        "forecast_step_1": forecast_val,
        "aic": float(model_fit.aic),
        "bic": float(model_fit.bic)
    }
