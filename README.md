# EUR/CHF Forecasting: Statistical & Machine Learning Models (2015–2025)

## 1. Project Context  
The foreign exchange (FX) market is one of the most liquid and data-rich environments in finance. Among currency pairs, **EUR/CHF** is particularly relevant due to its economic and geopolitical importance in Europe and Switzerland.  
This project investigates whether **short-term forecasting at horizon T+1 (one day ahead)** is feasible using both classical statistical approaches and modern machine learning techniques.  

The work follows a full **end-to-end pipeline**: from data collection and cleaning, to baseline models, statistical benchmarks, and advanced ML implementations, with a focus on **reproducibility and financial interpretation**.

---

## 2. Objectives  
The project is designed around two central goals:  

- **Forecasting objective**: evaluate whether the daily EUR/CHF exchange rate returns can be predicted one day ahead.  
- **Methodological comparison**: benchmark **classical statistical models** (ARIMA, GARCH, Holt-Winters) against **machine learning models** (Ridge Regression, Random Forest, XGBoost, Stacking).  

The ultimate aim is not only to test predictive power but also to draw lessons on the **limits of predictability in FX markets** and the conditions under which models provide business value (e.g., risk sizing, volatility awareness).

---

## 3. Pipeline Overview  
The project follows a structured 9-part pipeline to ensure clarity and reproducibility:  

1. **Data collection**  
   - Source: Yahoo Finance (`EURCHF=X`).  
   - Period: 2015–2025 (≈ 2,800 business days).  

2. **Data preparation**  
   - Compute log-returns for stationarity.  
   - Perform chronological **train/test split** (2015–2022 for training, 2023–2025 for testing).  
   - Create **engineered features**: lagged returns, rolling means, rolling volatilities.  

3. **Modeling**  
   - **Baselines**: Naïve, Moving Average, Simple Exponential Smoothing (SES).  
   - **Statistical models**: ARIMA, SARIMA, GARCH, Holt-Winters.  
   - **Machine Learning (ML) baselines**: Ridge, Lasso.  
   - **Advanced ML**: Random Forest, Gradient Boosting, XGBoost, LightGBM, Stacking & Blending.  

4. **Evaluation**  
   - Metrics: RMSE, MAE (MAPE discarded due to near-zero returns).  
   - Validation: expanding/rolling splits and **walk-forward backtesting**.  
   - Diagnostics: residual analysis, volatility regime breakdown, feature importance.  
---

---

## 4. Project Structure  

The repository is organized to ensure clarity, modularity, and reproducibility:  

```bash
eur_chf_forecasting/
│
├── data/
│   ├── raw/           # original downloaded data (Yahoo Finance, untouched)
│   ├── processed/     # cleaned datasets, train/test splits, engineered features
│
├── src/               # Python scripts (Part1.py … Part8.py)
│   ├── utils/         # helper functions (data loading, evaluation, plotting)
│
├── notebooks/         # optional Jupyter notebooks for exploration/demo
│
├── results/
│   ├── baseline/      # outputs from Naïve, MA, SES
│   ├── stats/         # ARIMA, SARIMA, Holt-Winters results
│   ├── ml_baseline/   # Linear Regression, SES vs ML comparisons
│   ├── ml_advanced/   # RF, XGB, LGBM, Stacking, Blending
│   ├── final/         # consolidated evaluation (Part 8)
│
├── reports/
│   ├── Report.pdf     # final report (exported from Word/LaTeX)
│   ├── figs/          # main figures (leaderboard, forecasts, residuals)
│
├── environment.yml    # Conda environment for reproducibility
├── requirements.txt   # alternative (pip) dependencies
├── LICENSE            # license file (e.g., MIT)
└── README.md          # project documentation (this file)
 ```


**Key principles**  
- Clear separation between raw and processed data → ensures traceability.  
- Scripts (`src/`) generate results reproducibly and save outputs in `results/`.  
- Figures and final report are centralized in `reports/`.  
- Environment files (`environment.yml`, `requirements.txt`) guarantee that anyone can recreate the same Python environment.  


---

## 5. Key Results  

The analysis compared **baselines, classical statistical models, and machine learning models** on the EUR/CHF daily log-returns (2015–2025).  

### 🔹 Main findings  
- **Best overall models**:  
  - **SES (Simple Exponential Smoothing)** and **ARIMA(2,0,1)** achieved the lowest RMSE and MAE.  
  - Both are simple, robust, and highly competitive benchmarks.  

- **Machine Learning (ML)**:  
  - **Linear Ridge** performed on par with SES/ARIMA.  
  - **Random Forest** and **XGBoost** captured non-linearities but did not significantly improve over statistical models.  
  - **Stacking (RF + GBDT with Ridge meta-model)** matched SES/ARIMA but did not outperform them.  

- **Conclusion**:  
  In noisy FX markets, **simplicity and robustness outperform complexity** unless enriched with **exogenous features** (macro variables, volatility indices, calendar effects).  

---

###  Illustrative Figures  

The following figures (available in `/reports/figs/`) summarize the key results:  

1. **EUR/CHF Exchange Rate (2015–2025)**  
   - Historical series used for modeling.  

2. **Distribution of Log-Returns**  
   - Heavy tails and skewness confirm challenges in forecasting.  

3. **Leaderboard of Models (RMSE on Test Set 2023–2025)**  
   - Bar chart ranking SES, ARIMA, Linear, RF, XGB, etc.  

4. **Forecast vs Actual (Top-3 Models)**  
   - Overlay showing SES, ARIMA, and Stacking predictions vs real returns.  

5. **Residual Distributions**  
   - Histograms showing residuals centered around zero with fat tails.  

---

### Quantitative Summary (Test 2023–2025)  

| Model        | RMSE     | MAE     | Δ vs SES |
|--------------|----------|---------|----------|
| SES          | 0.003321 | 0.002506 | — |
| ARIMA (2,0,1)| 0.003322 | 0.002509 | ≈0% |
| Ridge        | 0.003342 | 0.002515 | +0.5% |
| RandomForest | 0.003362 | 0.002528 | +1.1% |
| XGBoost      | 0.003410 | 0.002556 | +2.5% |
| LightGBM     | 0.003569 | 0.002676 | +7.3% |
| Stacking     | 0.003324 | 0.002506 | ≈0% |

 **SES and ARIMA remain the champions**, with Ridge and Stacking as close challengers.  
Complex models add little value without exogenous variables.  

---

## 6. Recommended Figures  

The following figures are suggested to showcase the project results.  
They are stored in `/reports/figs/` and can be included in presentations or dashboards:  

1. **EUR/CHF Exchange Rate (2015–2025)**  
   - Historical time series used for modeling.  

2. **Histogram of Log-Returns**  
   - Shows fat tails and strong skewness typical of FX returns.  

3. **Leaderboard of Models (Test RMSE)**  
   - Bar chart comparing SES, ARIMA, Ridge, Random Forest, XGBoost, etc.  

4. **Forecast vs Actual (Top-3 Models)**  
   - Overlay of SES, ARIMA, and Stacking predictions against real returns.  

5. **Residual Distributions**  
   - Histograms confirming residuals are centered near zero, with fat tails remaining.  

---

## 7. Reproducibility  

This repository is designed to ensure **full reproducibility**.  
Anyone can replicate the results by following these steps:  

1. **Clone the repository**  
   ```bash
   git clone https://github.com/USERNAME/eur_chf_forecasting.git
   cd eur_chf_forecasting

2. Create the Conda environment

Create the Conda environment
conda env create -f environment.yml
conda activate forecast


3. Run the pipeline
Execute the scripts sequentially to reproduce all results:

python src/Part1.py
python src/Part2.py
python src/Part3.py
python src/Part4.py
python src/Part5.py
python src/Part6.py
python src/Part7.py
python src/Part8.py


4. Outputs

All datasets and predictions are saved in /data/processed/ and /results/.

Figures are exported to /reports/figs/.

A consolidated final report is available in /reports/Report.pdf.

With these steps, you can reproduce the same metrics, figures, and report as documented.

