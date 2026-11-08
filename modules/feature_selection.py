"""
==============================================================================
Module:      feature_selection.py
Description: Exhaustive subset evaluation, greedy forward stepwise selection,
             and backward elimination algorithms based on RMSE minimization.
==============================================================================
"""

import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from typing import List, Tuple, Dict, Any


def compute_rmse(x_tr: pd.DataFrame, y_tr: pd.Series, x_te: pd.DataFrame, y_te: pd.Series) -> float:
    """Berechnet den Root Mean Squared Error (RMSE) einer linearen Regression auf den Testdaten."""
    model = LinearRegression()
    model.fit(x_tr, y_tr)
    pred = model.predict(x_te)
    return float(np.sqrt(mean_squared_error(y_te, pred)))


def exhaustive_feature_search(x_train: pd.DataFrame, y_train: pd.Series,
                              x_test: pd.DataFrame, y_test: pd.Series) -> Tuple[Tuple[str, ...], float, List[Tuple[int, float]]]:
    """Testet alle moeglichen Feature-Kombinationen von 1 bis n Variablen."""
    all_features = list(x_train.columns)
    results = []

    for k in range(1, len(all_features) + 1):
        for subset in combinations(all_features, k):
            rmse_val = compute_rmse(x_train[list(subset)], y_train, x_test[list(subset)], y_test)
            results.append((subset, rmse_val))

    results.sort(key=lambda item: item[1])
    best_subset, best_rmse = results[0]
    dimension_curve = [(len(sub), rmse) for sub, rmse in results]

    return best_subset, best_rmse, dimension_curve


def greedy_forward_selection(x_train: pd.DataFrame, y_train: pd.Series,
                             x_test: pd.DataFrame, y_test: pd.Series) -> Tuple[List[str], float]:
    """Fuehrt eine schrittweise Vorwaertsauswahl (Forward Stepwise Selection) durch."""
    remaining = list(x_train.columns)
    selected = []
    best_rmse = float('inf')

    while remaining:
        candidates = []
        for feat in remaining:
            trial_features = selected + [feat]
            rmse_val = compute_rmse(x_train[trial_features], y_train, x_test[trial_features], y_test)
            candidates.append((feat, rmse_val))

        best_candidate, cand_rmse = min(candidates, key=lambda x: x[1])
        if cand_rmse < best_rmse:
            best_rmse = cand_rmse
            selected.append(best_candidate)
            remaining.remove(best_candidate)
        else:
            break

    return selected, best_rmse


def backward_elimination(x_train: pd.DataFrame, y_train: pd.Series,
                         x_test: pd.DataFrame, y_test: pd.Series) -> Tuple[List[str], float]:
    """Fuehrt eine schrittweise Rueckwaertselimination (Backward Elimination) durch."""
    current_features = list(x_train.columns)
    best_rmse = compute_rmse(x_train[current_features], y_train, x_test[current_features], y_test)

    while len(current_features) > 1:
        candidates = []
        for feat in current_features:
            trial_features = [f for f in current_features if f != feat]
            rmse_val = compute_rmse(x_train[trial_features], y_train, x_test[trial_features], y_test)
            candidates.append((feat, rmse_val))

        worst_feat, trial_rmse = min(candidates, key=lambda x: x[1])
        if trial_rmse < best_rmse:
            best_rmse = trial_rmse
            current_features.remove(worst_feat)
        else:
            break

    return current_features, best_rmse
