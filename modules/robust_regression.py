"""
==============================================================================
Module:      robust_regression.py
Description: Robust Huber loss regression versus classical OLS and statistical
             residual outlier identification using percentile thresholds.
==============================================================================
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import r2_score
from typing import Tuple, Dict, Any


def compare_ols_and_huber(x_train: pd.DataFrame, y_train: pd.Series,
                          x_test: pd.DataFrame, y_test: pd.Series,
                          percentile: float = 95.0) -> Dict[str, Any]:
    """Trainiert OLS und Huber-Regression, vergleicht Test-R2 und isoliert Ausreisser."""
    features = ['displacement', 'horsepower', 'weight']
    xt_sub = x_train[features]
    xte_sub = x_test[features]

    # OLS-Modell
    xt_const = sm.add_constant(xt_sub)
    xte_const = sm.add_constant(xte_sub)
    model_ols = sm.OLS(y_train, xt_const).fit()
    pred_ols_test = model_ols.predict(xte_const)
    pred_ols_train = model_ols.predict(xt_const)

    # Huber-Regressor
    model_huber = HuberRegressor()
    model_huber.fit(xt_sub, y_train)
    pred_hub_test = model_huber.predict(xte_sub)

    # Ausreißeranalyse über Residuen
    train_residuals = y_train - pred_ols_train
    abs_res = np.abs(train_residuals)
    threshold = float(np.percentile(abs_res, percentile))
    outlier_indices = np.where(abs_res > threshold)[0].tolist()

    return {
        "r2_test_ols": float(r2_score(y_test, pred_ols_test)),
        "r2_test_huber": float(r2_score(y_test, pred_hub_test)),
        "outlier_threshold": threshold,
        "outlier_indices": outlier_indices,
        "outlier_count": len(outlier_indices)
    }
