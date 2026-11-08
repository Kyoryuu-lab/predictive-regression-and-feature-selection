<h1>predictive-regression-and-feature-selection</h1>
<p>Computational econometric modeling, polynomial regression, robust Huber estimation, heuristic feature selection, time-series forecasting, and triangular fuzzy arithmetic in Python</p>

<h2>Project description</h2>
<p>This repository provides a modular, production-grade machine learning and econometrics suite implementing diagnostic regression modeling, heuristic feature selection, and time-series analysisx:</p>
<ol>
  <li>
    <dl>
      <dt><b>Multivariate Polynomial Regression &amp; Interaction Modeling:</b></dt>
      <dd>Evaluates the Auto MPG benchmark dataset across train/test splits (70/30).</dd>
      <dd>Compares univariate OLS baselines against 2nd-order polynomial models and models with multiplicative interaction terms (e.g. <code>displacement &times; horsepower</code>).</dd>
      <dd>Measures out-of-sample goodness-of-fit metrics ($R^2$, MSE, MAE).</dd>
    </dl>
  </li>
  <li>
    <dl>
      <dt><b>Robust Regression &amp; Residual Outlier Detection:</b></dt>
      <dd>Applies <code>HuberRegressor</code> to minimize sensitivity to measurement artifacts compared to standard least-squares estimators.</dd>
      <dd>Identifies anomalous observations exceeding the 95th percentile threshold of absolute residuals.</dd>
    </dl>
  </li>
  <li>
    <dl>
      <dt><b>Heuristic Feature Selection Algorithms:</b></dt>
      <dd>Executes exhaustive combinatorial searches across all $2^p - 1$ feature subsets.</dd>
      <dd>Implements Greedy Forward Selection and Backward Elimination to minimize validation RMSE.</dd>
    </dl>
  </li>
  <li>
    <dl>
      <dt><b>Time-Series Modeling &amp; Stationarity Testing:</b></dt>
      <dd>Conducts Augmented Dickey-Fuller unit root stationarity tests via <code>adfuller</code>.</dd>
      <dd>Trains autoregressive integrated moving average models ($\text{ARIMA}(1, 1, 1)$), rolling moving averages, and linear trend models for horizon forecasting.</dd>
    </dl>
  </li>
  <li>
    <dl>
      <dt><b>Triangular Fuzzy Arithmetic (Fuzzy Logic Extension):</b></dt>
      <dd>Evaluates piecewise triangular membership functions.</dd>
      <dd>Calculates analytical fuzzy addition, subtraction, and multiplication over bounded support intervals.</dd>
    </dl>
  </li>
</ol>

<h2>Project files</h2>
<table>
  <tr>
    <th colspan="2">predictive-regression-and-feature-selection</th>
  </tr>
  <tr>
    <td>main.py</td>
    <td>Master CLI runner controlling module dispatch and analytical execution</td>
  </tr>
  <tr>
    <td>modules/time_series_models.py</td>
    <td>Trend regression, rolling moving averages, and ARIMA stationarity pipelines</td>
  </tr>
  <tr>
    <td>modules/polynomial_regression.py</td>
    <td>Auto MPG preprocessing, train/test splitting, polynomial transformations, and interactions</td>
  </tr>
  <tr>
    <td>modules/robust_regression.py</td>
    <td>HuberRegressor implementation, residual modeling, and 95th percentile outlier detection</td>
  </tr>
  <tr>
    <td>modules/feature_selection.py</td>
    <td>Exhaustive subset search, greedy forward selection, and backward elimination</td>
  </tr>
  <tr>
    <td>modules/fuzzy_operations.py</td>
    <td>Triangular fuzzy number arithmetic operations (addition, subtraction, multiplication)</td>
  </tr>
  <tr>
    <td>README.md</td>
    <td>Project documentation and technical specification</td>
  </tr>
  <tr>
    <td>.gitignore</td>
    <td>Exclusion rules for bytecode, virtual environments, and generated plots</td>
  </tr>
</table>

<h2>Methodological Formulation</h2>
<table>
  <tr>
    <th>Task</th>
    <th>Method</th>
    <th>Mathematical Formula</th>
  </tr>
  <tr>
    <td><b>Polynomial Model</b></td>
    <td>Interaction OLS</td>
    <td>$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_3 x_1^2 + \beta_4 x_2^2 + \beta_5 (x_1 \cdot x_2) + \epsilon$</td>
  </tr>
  <tr>
    <td><b>Robust Loss</b></td>
    <td>Huber Objective</td>
    <td>$L_\delta(r) = \begin{cases} \frac{1}{2} r^2 & \text{for } |r| \leq \delta \\ \delta (|r| - \frac{1}{2} \delta) & \text{otherwise} \end{cases}$</td>
  </tr>
  <tr>
    <td><b>Feature Selection</b></td>
    <td>RMSE Minimization</td>
    <td>$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}$</td>
  </tr>
  <tr>
    <td><b>Time-Series</b></td>
    <td>ARIMA(1, 1, 1)</td>
    <td>$(1 - \phi_1 B)(1 - B) y_t = (1 + \theta_1 B) \epsilon_t$</td>
  </tr>
  <tr>
    <td><b>Fuzzy Addition</b></td>
    <td>Triangular Sum</td>
    <td>$A + B = (a_1 + b_1, \; a_2 + b_2, \; a_3 + b_3)$</td>
  </tr>
  <tr>
    <td><b>Fuzzy Subtraction</b></td>
    <td>Triangular Difference</td>
    <td>$C - D = (c_1 - d_3, \; c_2 - d_2, \; c_3 - d_1)$</td>
  </tr>
</table>

<h2>Installation and launch</h2>
<ol>
  <b>
    <li>
      <dl>
        <dt>Clone repository</dt>
        <dd>
          <code>git clone https://github.com/Kyoryuu-lab/predictive-regression-and-feature-selection.git</code>
        </dd>
        <dd>
          <code>cd predictive-regression-and-feature-selection</code>
        </dd>
      </dl>
    </li>
  </b>
  <b>
    <li>
      <dl>
        <dt>Install dependencies</dt>
        <dd>
          <code>pip install numpy pandas scikit-learn statsmodels matplotlib</code>
        </dd>
      </dl>
    </li>
  </b>
  <b>
    <li>
      <dl>
        <dt>Run complete suite</dt>
        <dd>
          <code>python main.py --module all</code>
        </dd>
        <dd>Execute individual modules: <code>python main.py --module regression</code> (Options: <code>timeseries</code>, <code>regression</code>, <code>fuzzy</code>)</dd>
      </dl>
    </li>
  </b>
</ol>

<h2>Example of console output (python main.py --module all)</h2>
<samp>
  <p>
    ==============================================================================<br />
    &nbsp;&nbsp;1. MODUL: ZEITREIHENPROGNOSE &amp; ARIMA (ABSATZANALYSE)<br />
    ==============================================================================<br />
    [*] Lineare Regression: Y = 29.80 + 2.37X | MSE: 4.54<br />
    &nbsp;&nbsp;&nbsp;&nbsp;Prognose Monat 13 : 60.62 Tsd. Einheiten<br />
    [*] Gleitender Durchschnitt (3 Monate): Monat 13 = 56.00 Tsd. Einheiten<br />
    [*] ADF-Stationaritaetstest : Stat = -0.5841 (p = 0.8745)<br />
    &nbsp;&nbsp;&nbsp;&nbsp;ARIMA(1,1,1) Prognose&nbsp;&nbsp;&nbsp;: Monat 13 = 62.15 Tsd. Einheiten<br />
    <br />
    ==============================================================================<br />
    &nbsp;&nbsp;2. MODUL: POLYNOMMODELLE &amp; ROBUSTE REGRESSION (AUTO MPG)<br />
    ==============================================================================<br />
    [*] Vergleich der Modellkomplexitaet auf den Testdaten:<br />
    &nbsp;&nbsp;&nbsp;&nbsp;Linear OLS&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-&gt; R²: 0.654 | MSE: 17.514 | MAE: 3.250<br />
    &nbsp;&nbsp;&nbsp;&nbsp;Polynomial (2nd Order)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-&gt; R²: 0.682 | MSE: 16.089 | MAE: 3.011<br />
    &nbsp;&nbsp;&nbsp;&nbsp;Polynomial + Interaction&nbsp;&nbsp;&nbsp;-&gt; R²: 0.701 | MSE: 15.122 | MAE: 2.894<br />
    <br />
    [*] Robuste Regression vs. Standard-OLS:<br />
    &nbsp;&nbsp;&nbsp;&nbsp;OLS Test R²&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: 0.7051<br />
    &nbsp;&nbsp;&nbsp;&nbsp;Huber Test R²&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: 0.7012<br />
    &nbsp;&nbsp;&nbsp;&nbsp;Ausreisser-Grenze : 6.182 (Anzahl: 14)<br />
    <br />
    ==============================================================================<br />
    &nbsp;&nbsp;3. MODUL: HEURISTISCHE FEATURE-SELEKTION (RMSE-MINIMIERUNG)<br />
    ==============================================================================<br />
    [*] Bester vollstaendiger Subset: ('cylinders', 'horsepower', 'weight', 'model year', 'origin')<br />
    &nbsp;&nbsp;&nbsp;&nbsp;Minimaler Test-RMSE&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: 3.8241<br />
    [*] Greedy Forward Selection&nbsp;&nbsp;&nbsp;&nbsp;: ['weight', 'model year', 'horsepower', 'origin', 'cylinders'] (RMSE: 3.8241)<br />
    [*] Backward Elimination&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: ['cylinders', 'horsepower', 'weight', 'model year', 'origin'] (RMSE: 3.8241)<br />
    <br />
    ==============================================================================<br />
    &nbsp;&nbsp;4. MODUL: ARITHMETIK VON DREIECKS-FUZZY-ZAHLEN (LAB 4)<br />
    ==============================================================================<br />
    [*] Addition&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: A(2, 5, 8) + B(3, 6, 9) = (5, 11, 17)<br />
    [*] Subtraktion&nbsp;&nbsp;&nbsp;&nbsp;: C(1, 3, 5) - D(2, 4, 6) = (-5, -1, 3)<br />
    [*] Multiplikation : E(2, 4, 6) * F(1, 2, 3) = (2, 8, 18)<br />
    <br />
    ==============================================================================<br />
    &nbsp;&nbsp;BERECHNUNGEN ERFOLGREICH BEENDET.<br />
    ==============================================================================
  </p>
