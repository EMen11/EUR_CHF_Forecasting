# Chapitre 5 – Modèles statistiques classiques

_Généré le 2025-09-20 02:08_

## 5.5 Documentation & Rapport (FR)

### 🎯 Objectif

- Tester des **modèles statistiques classiques** (ARIMA, SARIMA, Holt-Winters) pour la prévision des **log-returns EUR/CHF**.

- Établir une **référence “classique”** à comparer aux modèles de **Machine Learning** (Chapitre 6).

### 🧪 Méthodes

- **ARIMA(2,0,1)** sélectionné par AIC sur returns stationnaires (d=0).

- **SARIMA** avec détection de **saisonnalité hebdo (s=5)** → meilleur par AIC : **SARIMA(2,0,0)×(0,1,1,5)**.

- **Holt-Winters** : variantes SES (sans tendance), Holt (tendance additive), saisonnier (s=5, additif).

- Split temporel: **train 2015–2022**, **test 2023–2025**. Pas de fuite, index datetime propre.

### 📊 Résultats consolidés (RMSE/MAE/MAPE)

| model                    |     rmse |      mae | mape_pct   |
|:-------------------------|---------:|---------:|:-----------|
| ARIMA predictions        | 0.003322 | 0.002509 | 7,056      |
| HW ses                   | 0.003322 | 0.002509 | 6,885      |
| HW holt add trend        | 0.003323 | 0.002509 | 8,236      |
| HW seasonal add s5       | 0.00334  | 0.00254  | 23,246     |
| SARIMA (2,0,0) (0,1,1,5) | 0.00335  | 0.002521 | 27,150     |

**Meilleur modèle (statistique)** : **ARIMA predictions** → RMSE=0.003322, MAE=0.002509.

> Remarque : le **MAPE** n’est pas pertinent sur des retours proches de 0 (dénominateur quasi nul).

### 🖼️ Figures clés

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

### ⚠️ Limites

- **Rigidité face aux chocs** : les modèles linéaires captent mal les ruptures (ex. chocs de volatilité).

- **Saisonniété faible** : le signal hebdomadaire (s=5) n’améliore pas la prévision.

- **Volatilité mal capturée** : ces modèles se concentrent sur la moyenne, pas sur la dynamique de variance.

- **Sur-ajustement possible** si l’on empile trop de paramètres saisonniers peu informatifs.

### ✅ Conclusion & Référence

- **ARIMA(2,0,1)** et **HW-SES** fournissent les **meilleures erreurs** (RMSE≈0.003322, MAE≈0.002508), au coude-à-coude.

- **Référence “classique”** : retenir **HW-SES** (simplicité/robustesse) et **ARIMA(2,0,1)** (référence ARMA) pour la comparaison avec le **Chapitre 6 (ML)**.
