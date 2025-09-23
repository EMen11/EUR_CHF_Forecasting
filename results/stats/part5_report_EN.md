# Chapter 5 – Classical Statistical Models

_Generated on 2025-09-20 02:08_

## 5.5 Documentation & Report (EN)

### 🎯 Objective

- Test **classical statistical models** (ARIMA, SARIMA, Holt-Winters) for **EUR/CHF log-return** forecasting.

- Establish a **“classical benchmark”** to compare against **Machine Learning** models (Chapter 6).

### 🧪 Methods

- **ARIMA(2,0,1)** selected via AIC on stationary returns (d=0).

- **SARIMA** with **weekly seasonality (s=5)** → best by AIC: **SARIMA(2,0,0)×(0,1,1,5)**.

- **Holt-Winters**: SES (no trend), Holt (additive trend), seasonal (s=5, additive).

- Time split: **train 2015–2022**, **test 2023–2025**. No leakage; clean datetime index.

### 📊 Consolidated Results (RMSE/MAE/MAPE)

| model                    |     rmse |      mae | mape_pct   |
|:-------------------------|---------:|---------:|:-----------|
| ARIMA predictions        | 0.003322 | 0.002509 | 7,056      |
| HW ses                   | 0.003322 | 0.002509 | 6,885      |
| HW holt add trend        | 0.003323 | 0.002509 | 8,236      |
| HW seasonal add s5       | 0.00334  | 0.00254  | 23,246     |
| SARIMA (2,0,0) (0,1,1,5) | 0.00335  | 0.002521 | 27,150     |

**Best (statistical) model**: **ARIMA predictions** → RMSE=0.003322, MAE=0.002509.

> Note: **MAPE** is not meaningful on near-zero returns (denominator issues).

### 🖼️ Key Figures

- `results/stats/arima(2, 0, 1)_forecast.png`
- `results/stats/hw_holt_add_trend_forecast.png`
- `results/stats/hw_seasonal_add_s5_forecast.png`
- `results/stats/hw_seasonal_add_s5_resid_acf.png`
- `results/stats/hw_ses_forecast.png`
- `results/stats/part5_mae_comparison.png`
- `results/stats/part5_residuals_ARIMA_predictions.png`
- `results/stats/part5_residuals_HW_holt_add_trend.png`
- `results/stats/part5_residuals_HW_seasonal_add_s5.png`
- `results/stats/part5_residuals_HW_ses.png`
- `results/stats/part5_residuals_SARIMA_2_0_0_0_1_1_5_.png`
- `results/stats/part5_rmse_comparison.png`
- `results/stats/part5_top3_forecasts_vs_test.png`
- `results/stats/sarima_(2,0,0)_(0,1,1,5)_forecast.png`
- `results/stats/sarima_(2,0,0)_(0,1,1,5)_resid_acf.png`

### ⚠️ Limitations

- **Rigid to shocks**: linear models do not adapt well to structural breaks/vol spikes.

- **Weak seasonality**: weekly signal (s=5) does not improve forecasts.

- **Volatility not captured**: these models target the conditional mean, not variance dynamics.

- **Potential overfitting** if excessive seasonal parameters are used with weak signal.

### ✅ Conclusion & Benchmark

- **ARIMA(2,0,1)** and **HW-SES** deliver the **lowest errors** (RMSE≈0.003322, MAE≈0.002508), essentially tied.

- **Classical benchmark**: keep **HW-SES** (simplicity/robustness) and **ARIMA(2,0,1)** (ARMA reference) for comparison against **Chapter 6 (ML)**.
