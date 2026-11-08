"""
==============================================================================
Project:     Predictive Regression & Feature Selection Suite
File:        main.py
Author:      Yevhen Biezhyn
Description: Unified CLI orchestrator executing polynomial regression, robust
             Huber modeling, greedy feature selection, time-series forecasting,
             and triangular fuzzy arithmetic.
==============================================================================
"""

import argparse
import numpy as np

from modules.time_series_models import forecast_linear_trend, forecast_moving_average, fit_arima_forecast
from modules.polynomial_regression import load_auto_mpg_data, prepare_train_test_split, evaluate_polynomial_variants
from modules.robust_regression import compare_ols_and_huber
from modules.feature_selection import exhaustive_feature_search, greedy_forward_selection, backward_elimination
from modules.fuzzy_operations import add_triangular_fuzzy, subtract_triangular_fuzzy, multiply_triangular_fuzzy


def print_banner(title: str):
    print("\n" + "=" * 78)
    print(f"  {title.upper()}")
    print("=" * 78)


def run_time_series_suite():
    print_banner("1. Modul: Zeitreihenprognose & ARIMA (Absatzanalyse)")
    sales_data = np.array([30, 35, 40, 38, 42, 45, 50, 48, 52, 55, 53, 60])

    # 1.1 OLS-Trend
    b0, b1, pred_ols, mse_val = forecast_linear_trend(sales_data, target_month=13)
    print(f"[*] Lineare Regression: Y = {b0:.2f} + {b1:.2f}X | MSE: {mse_val:.2f}")
    print(f"    Prognose Monat 13 : {pred_ols:.2f} Tsd. Einheiten")

    # 1.2 Gleitender Durchschnitt
    _, pred_ma = forecast_moving_average(sales_data, window_size=3)
    print(f"[*] Gleitender Durchschnitt (3 Monate): Monat 13 = {pred_ma:.2f} Tsd. Einheiten")

    # 1.3 ARIMA(1, 1, 1) mit ADF-Test
    arima_res = fit_arima_forecast(sales_data, order=(1, 1, 1))
    print(f"[*] ADF-Stationaritaetstest : Stat = {arima_res['adf_statistic']:.4f} (p = {arima_res['p_value']:.4f})")
    print(f"    ARIMA(1,1,1) Prognose   : Monat 13 = {arima_res['forecast_step_1']:.2f} Tsd. Einheiten (AIC: {arima_res['aic']:.1f})")


def run_auto_mpg_suite():
    print_banner("2. Modul: Polynommodelle & Robuste Regression (Auto MPG)")
    print("[*] Lade Auto MPG Benchmark-Datensatz...")
    data = load_auto_mpg_data()
    x_tr, x_te, y_tr, y_te = prepare_train_test_split(data, test_size=0.3, random_state=42)

    # 2.1 Polynommodelle
    poly_metrics = evaluate_polynomial_variants(x_tr, y_tr, x_te, y_te)
    print("\n[*] Vergleich der Modellkomplexitaet auf den Testdaten:")
    for m_name, m_vals in poly_metrics.items():
        print(f"    {m_name:<26} -> R²: {m_vals['R2']:.3f} | MSE: {m_vals['MSE']:.3f} | MAE: {m_vals['MAE']:.3f}")

    # 2.2 Robuste Huber-Regression & Ausreißer
    hub_res = compare_ols_and_huber(x_tr, y_tr, x_te, y_te, percentile=95.0)
    print("\n[*] Robuste Regression vs. Standard-OLS:")
    print(f"    OLS Test R²       : {hub_res['r2_test_ols']:.4f}")
    print(f"    Huber Test R²     : {hub_res['r2_test_huber']:.4f}")
    print(f"    Ausreisser-Grenze : {hub_res['outlier_threshold']:.3f} (Anzahl: {hub_res['outlier_count']})")

    # 2.3 Feature-Selektion
    print_banner("3. Modul: Heuristische Feature-Selektion (RMSE-Minimierung)")
    best_sub, best_rmse_ex, _ = exhaustive_feature_search(x_tr, y_tr, x_te, y_te)
    print(f"[*] Bester vollstaendiger Subset: {best_sub}")
    print(f"    Minimaler Test-RMSE         : {best_rmse_ex:.4f}")

    fwd_feats, fwd_rmse = greedy_forward_selection(x_tr, y_tr, x_te, y_te)
    print(f"[*] Greedy Forward Selection    : {fwd_feats} (RMSE: {fwd_rmse:.4f})")

    bwd_feats, bwd_rmse = backward_elimination(x_tr, y_tr, x_te, y_te)
    print(f"[*] Backward Elimination        : {bwd_feats} (RMSE: {bwd_rmse:.4f})")


def run_fuzzy_arithmetic_suite():
    print_banner("4. Modul: Arithmetik von Dreiecks-Fuzzy-Zahlen (Lab 4)")
    # Dreiecks-Zahlen
    a, b = (2.0, 5.0, 8.0), (3.0, 6.0, 9.0)
    c, d = (1.0, 3.0, 5.0), (2.0, 4.0, 6.0)
    e, f = (2.0, 4.0, 6.0), (1.0, 2.0, 3.0)

    res_add = add_triangular_fuzzy(a, b)
    res_sub = subtract_triangular_fuzzy(c, d)
    res_mul = multiply_triangular_fuzzy(e, f)

    print(f"[*] Addition       : A{a} + B{b} = {res_add}")
    print(f"[*] Subtraktion    : C{c} - D{d} = {res_sub}")
    print(f"[*] Multiplikation : E{e} * F{f} = {res_mul}")


def main():
    parser = argparse.ArgumentParser(
        description="Predictive Regression & Feature Selection Suite",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--module",
        choices=["all", "timeseries", "regression", "fuzzy"],
        default="all",
        help=(
            "all        : Fuehrt alle Module vollstaendig aus (Standard)\n"
            "timeseries : OLS-Trend, Gleitender Durchschnitt & ARIMA\n"
            "regression : Auto MPG Polynome, Huber-Regression & Feature-Selektion\n"
            "fuzzy      : Dreiecks-Fuzzy-Arithmetik (Lab 4)"
        )
    )
    args = parser.parse_args()

    print("=" * 78)
    print("  PREDICTIVE REGRESSION & FEATURE SELECTION SUITE")
    print("=" * 78)

    if args.module in ["all", "timeseries"]:
        run_time_series_suite()
    if args.module in ["all", "regression"]:
        run_auto_mpg_suite()
    if args.module in ["all", "fuzzy"]:
        run_fuzzy_arithmetic_suite()

    print("\n" + "=" * 78)
    print("  BERECHNUNGEN ERFOLGREICH BEENDET.")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
