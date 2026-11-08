"""
==============================================================================
Module:      polynomial_regression.py
Description: Data preprocessing for Auto MPG, univariate/multivariate OLS,
             higher-order polynomial transformations, and interaction effects.
==============================================================================
"""

import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from typing import Tuple, Dict


def load_auto_mpg_data(url: str = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data") -> pd.DataFrame:
    """Laedt und bereinigt den Auto MPG Benchmark-Datensatz."""
    cols = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin', 'car name']
    data = pd.read_csv(url, delim_whitespace=True, names=cols, na_values='?')
    return data.dropna()


def prepare_train_test_split(data: pd.DataFrame, test_size: float = 0.3, random_state: int = 42):
    """Teilt Merkmale und Zielgroesse in Trainings- und Testdaten auf."""
    features = data[['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin']]
    target = data['mpg']
    return train_test_split(features, target, test_size=test_size, random_state=random_state)


def evaluate_polynomial_variants(x_train: pd.DataFrame, y_train: pd.Series,
                                 x_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Dict[str, float]]:
    """Trainiert und vergleicht lineare, rein quadratische und quadratische Modelle mit Interaktion."""
    # Basis: Displacement und Horsepower
    xt_base = x_train[['displacement', 'horsepower']].copy()
    xte_base = x_test[['displacement', 'horsepower']].copy()

    # 1. Lineares OLS
    m_lin = sm.OLS(y_train, sm.add_constant(xt_base)).fit()
    pred_lin = m_lin.predict(sm.add_constant(xte_base))

    # 2. Quadratisches Modell (ohne Interaktion)
    xt_poly = xt_base.copy()
    xt_poly['disp_sq'] = xt_poly['displacement'] ** 2
    xt_poly['hp_sq'] = xt_poly['horsepower'] ** 2
    xte_poly = xte_base.copy()
    xte_poly['disp_sq'] = xte_poly['displacement'] ** 2
    xte_poly['hp_sq'] = xte_poly['horsepower'] ** 2

    m_poly = sm.OLS(y_train, sm.add_constant(xt_poly)).fit()
    pred_poly = m_poly.predict(sm.add_constant(xte_poly))

    # 3. Quadratisches Modell mit Interaktion
    xt_inter = xt_poly.copy()
    xt_inter['disp_hp'] = xt_inter['displacement'] * xt_inter['horsepower']
    xte_inter = xte_poly.copy()
    xte_inter['disp_hp'] = xte_inter['displacement'] * xte_inter['horsepower']

    m_inter = sm.OLS(y_train, sm.add_constant(xt_inter)).fit()
    pred_inter = m_inter.predict(sm.add_constant(xte_inter))

    def calc_metrics(y_true, y_pred):
        return {
            "R2": float(r2_score(y_true, y_pred)),
            "MSE": float(mean_squared_error(y_true, y_pred)),
            "MAE": float(mean_absolute_error(y_true, y_pred))
        }

    return {
        "Linear OLS": calc_metrics(y_test, pred_lin),
        "Polynomial (2nd Order)": calc_metrics(y_test, pred_poly),
        "Polynomial + Interaction": calc_metrics(y_test, pred_inter)
    }
