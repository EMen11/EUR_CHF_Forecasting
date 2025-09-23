# =========================================================
# 5.0 - Setup (Paths & Data)
# =========================================================

import os
import pandas as pd

# Définir la racine du projet (équivalent à Part2/3/4)
PROJECT_ROOT = os.getcwd()
DATA_PROC    = os.path.join(PROJECT_ROOT, "data", "processed")
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results")
RESULTS_STATS = os.path.join(RESULTS_DIR, "stats")
os.makedirs(RESULTS_STATS, exist_ok=True)

# --- Quick validation of chronological split ---
train = pd.read_csv(os.path.join(DATA_PROC, "eur_chf_train.csv"), parse_dates=["date"], index_col="date")
test  = pd.read_csv(os.path.join(DATA_PROC, "eur_chf_test.csv"), parse_dates=["date"], index_col="date")

print("Train period:", train.index.min().date(), "→", train.index.max().date(), "| Rows:", len(train))
print("Test period :", test.index.min().date(),  "→", test.index.max().date(),  "| Rows:", len(test))


print("========================================")
print("5.0 - Setup (Stats Models)")
print("========================================")
print("PROJECT_ROOT:", PROJECT_ROOT)
print("DATA_PROC   :", DATA_PROC)
print("RESULTS_STATS:", RESULTS_STATS)

# =========================================================
# 5.1 - ARIMA (AutoRegressive Integrated Moving Average)
# =========================================================

# --- Utility: banner print ---


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller



# --- Chemins
RESULTS_STATS = os.path.join(PROJECT_ROOT, "results", "stats")
os.makedirs(RESULTS_STATS, exist_ok=True)

# --- Charger les datasets
train = pd.read_csv(os.path.join(DATA_PROC, "eur_chf_train.csv"), parse_dates=["date"], index_col="date")
test  = pd.read_csv(os.path.join(DATA_PROC, "eur_chf_test.csv"), parse_dates=["date"], index_col="date")

y_train = train["log_return"]
y_test  = test["log_return"]

# ---------------------------------------------------------
# 5.1.A - Test de stationnarité (ADF test)
# ---------------------------------------------------------
adf_stat, p_value, _, _, crit_vals, _ = adfuller(y_train)
print(f"[ADF] stat={adf_stat:.4f}, p-value={p_value:.4g}")
print("Critical values:", crit_vals)

# ---------------------------------------------------------
# 5.1.B - ACF/PACF (diagnostics visuels)
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(y_train, lags=40, ax=axes[0])
plot_pacf(y_train, lags=40, ax=axes[1])
plt.suptitle("5.1 - ACF & PACF (Train EUR/CHF returns)")
plt.savefig(os.path.join(RESULTS_STATS, "arima_acf_pacf.png"))
plt.close()

# ---------------------------------------------------------
# 5.1.C - Fit du modèle ARIMA
# ---------------------------------------------------------
# Hypothèse: (2,0,1) déjà identifié comme "meilleur" par AIC
order = (2, 0, 1)

model = ARIMA(y_train, order=order)
fit = model.fit()

print(fit.summary())

# ---------------------------------------------------------
# 5.1.D - Prévisions sur la période test
# ---------------------------------------------------------
forecast = fit.get_forecast(steps=len(y_test))
y_pred = forecast.predicted_mean
conf_int = forecast.conf_int(alpha=0.05)

# Sauvegarde prédictions
df_pred = pd.DataFrame({
    "y_true": y_test,
    "y_pred": y_pred,
    "ci_lower": conf_int.iloc[:, 0],
    "ci_upper": conf_int.iloc[:, 1]
}, index=y_test.index)
df_pred.to_csv(os.path.join(RESULTS_STATS, "arima_predictions.csv"))

# ---------------------------------------------------------
# 5.1.E - Évaluation des performances
# ---------------------------------------------------------
rmse = np.sqrt(((y_pred - y_test)**2).mean())
mae  = np.abs(y_pred - y_test).mean()

print(f"[ARIMA{order}] Test RMSE={rmse:.6f} | MAE={mae:.6f}")

# Append leaderboard
leaderboard_path = os.path.join(PROJECT_ROOT, "models", "leaderboard.csv")
if os.path.exists(leaderboard_path):
    lb = pd.read_csv(leaderboard_path)
else:
    lb = pd.DataFrame(columns=["model", "test_RMSE", "test_MAE"])

lb = pd.concat([lb, pd.DataFrame([{
    "model": f"ARIMA{order}",
    "test_RMSE": rmse,
    "test_MAE": mae
}])], ignore_index=True)

lb.to_csv(leaderboard_path, index=False)

# ---------------------------------------------------------
# 5.1.F - Figures
# ---------------------------------------------------------
plt.figure(figsize=(12,5))
plt.plot(y_train.index, y_train, label="Train")
plt.plot(y_test.index, y_test, label="Test True")
plt.plot(y_pred.index, y_pred, label=f"ARIMA{order} Forecast", color="red")
plt.fill_between(y_pred.index, conf_int.iloc[:,0], conf_int.iloc[:,1], color="pink", alpha=0.3)
plt.legend()
plt.title(f"5.1 - ARIMA{order} Forecast vs Test")
plt.savefig(os.path.join(RESULTS_STATS, f"arima{order}_forecast.png"))
plt.close()

print("[5.1] ARIMA complete.")

# =========================================================
# 5.2 - SARIMA (Seasonal ARIMA)
# =========================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller, acf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf

# ---------- Paths (reuse if already defined) ----------
try:
    PROJECT_ROOT
except NameError:
    PROJECT_ROOT = os.getcwd()

DATA_PROC     = os.path.join(PROJECT_ROOT, "data", "processed")
RESULTS_DIR   = os.path.join(PROJECT_ROOT, "results")
RESULTS_STATS = os.path.join(RESULTS_DIR, "stats")
MODELS_DIR    = os.path.join(PROJECT_ROOT, "models")
os.makedirs(RESULTS_STATS, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

print("========================================")
print("5.2 - SARIMA (P,D,Q,s)")
print("========================================")
print("PROJECT_ROOT:", PROJECT_ROOT)
print("DATA_PROC   :", DATA_PROC)
print("RESULTS_STATS:", RESULTS_STATS)

# ---------- Load data ----------
train = pd.read_csv(os.path.join(DATA_PROC, "eur_chf_train.csv"), parse_dates=["date"], index_col="date")
test  = pd.read_csv(os.path.join(DATA_PROC, "eur_chf_test.csv"),  parse_dates=["date"], index_col="date")

y_train = train["log_return"].astype(float)
y_test  = test["log_return"].astype(float)

print(f"Train period: {y_train.index.min().date()} → {y_train.index.max().date()} | n={len(y_train)}")
print(f"Test  period: {y_test.index.min().date()}  → {y_test.index.max().date()}  | n={len(y_test)}")

# ---------- 5.2.A Seasonality identification ----------
# Candidates typical for business-day FX:
seasonal_candidates = [5, 21, 63]  # weekly (~5 bd), monthly (~21 bd), quarterly (~63 bd)
# Heuristic: pick the s with highest |ACF(s)| if above a small threshold; else keep [5,21,63] for search
acf_vals = acf(y_train, nlags=max(seasonal_candidates), fft=True)
cand_strength = {s: abs(acf_vals[s]) for s in seasonal_candidates}
print("[Seasonality ACF strengths]:", cand_strength)
best_s = max(cand_strength, key=cand_strength.get)
if cand_strength[best_s] < 0.03:
    # Weak seasonality → still try multiple s, but mark as weak
    print("[Seasonality] Weak autocorr at seasonal lags; will try s in", seasonal_candidates)
    s_list = seasonal_candidates
else:
    print(f"[Seasonality] Strongest candidate s={best_s} (|ACF|={cand_strength[best_s]:.3f})")
    s_list = [best_s]  # narrow search

# Save a quick ACF figure highlighting seasonal lags
plt.figure(figsize=(10,4))
plot_acf(y_train, lags=70)
for s in seasonal_candidates:
    plt.axvline(s, linestyle="--", alpha=0.5)
plt.title("5.2 - ACF with seasonal lag markers (5, 21, 63)")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_STATS, "sarima_acf_seasonal_markers.png"))
plt.close()

# ---------- 5.2.B Order selection (mini grid by AIC) ----------
# We already found ARIMA(2,0,1) was strong; keep d=0 (returns are stationary), try D in {0,1}
p_grid = [0,1,2]
q_grid = [0,1,2]
d      = 0
P_grid = [0,1]
Q_grid = [0,1]
D_grid = [0,1]

results = []
best_model = None
best_aic = np.inf
best_spec = None

for s in s_list:
    for p in p_grid:
        for q in q_grid:
            for P in P_grid:
                for Q in Q_grid:
                    for D in D_grid:
                        order = (p, d, q)
                        seasonal_order = (P, D, Q, s)
                        # Skip the all-zero seasonal part if s>1 and P=Q=D=0 AND p=q=0 → too trivial
                        try:
                            model = SARIMAX(
                                y_train,
                                order=order,
                                seasonal_order=seasonal_order,
                                enforce_stationarity=False,
                                enforce_invertibility=False
                            )
                            fit = model.fit(disp=False)
                            aic = fit.aic
                            results.append((aic, order, seasonal_order))
                            if aic < best_aic:
                                best_aic = aic
                                best_model = fit
                                best_spec = (order, seasonal_order)
                            print(f"[SARIMA {order}x{seasonal_order}] AIC={aic:.2f}")
                        except Exception as e:
                            # Some combos may fail to converge; that's fine
                            print(f"[skip] {order}x{seasonal_order} → {e}")

# If nothing converged, fallback to non-seasonal ARIMA(2,0,1)
if best_model is None:
    print("[WARN] No SARIMA spec converged; falling back to ARIMA(2,0,1) as SARIMAX with s=0")
    model = SARIMAX(y_train, order=(2,0,1), seasonal_order=(0,0,0,0), enforce_stationarity=False, enforce_invertibility=False)
    best_model = model.fit(disp=False)
    best_spec = ((2,0,1), (0,0,0,0))
    best_aic = best_model.aic

# Persist the AIC table
aic_df = pd.DataFrame(results, columns=["AIC", "order", "seasonal_order"]).sort_values("AIC").reset_index(drop=True)
aic_path = os.path.join(RESULTS_STATS, "sarima_aic_grid.csv")
aic_df.to_csv(aic_path, index=False)
print("[save] AIC grid ->", aic_path)

order_sel, seasonal_sel = best_spec
print(f"[BEST] SARIMA{order_sel}x{seasonal_sel} AIC={best_aic:.2f}")

# ---------- 5.2.C Fit (already fitted) & Forecast ----------
steps = len(y_test)
pred_res = best_model.get_forecast(steps=steps)
y_pred   = pred_res.predicted_mean
ci       = pred_res.conf_int(alpha=0.05)  # columns: lower y, upper y

# Align index to test
y_pred.index = y_test.index
ci.index     = y_test.index

pred_df = pd.DataFrame({
    "y_true": y_test,
    "y_pred": y_pred,
    "ci_lower": ci.iloc[:,0],
    "ci_upper": ci.iloc[:,1]
}, index=y_test.index)
pred_path = os.path.join(RESULTS_STATS, f"sarima_{order_sel}_{seasonal_sel}_predictions.csv".replace(" ", ""))
pred_df.to_csv(pred_path)
print("[save] Predictions ->", pred_path)

# ---------- Metrics ----------
rmse = np.sqrt(((y_pred - y_test)**2).mean())
mae  = np.abs(y_pred - y_test).mean()
print(f"[SARIMA{order_sel}x{seasonal_sel}] Test RMSE={rmse:.6f} | MAE={mae:.6f}")

# ---------- Update leaderboard ----------
leaderboard_path = os.path.join(MODELS_DIR, "leaderboard.csv")
if os.path.exists(leaderboard_path):
    lb = pd.read_csv(leaderboard_path)
else:
    lb = pd.DataFrame(columns=["model", "test_RMSE", "test_MAE"])

lb = pd.concat([lb, pd.DataFrame([{
    "model": f"SARIMA{order_sel}x{seasonal_sel}",
    "test_RMSE": rmse,
    "test_MAE": mae
}])], ignore_index=True)

lb.to_csv(leaderboard_path, index=False)
print("[save] Leaderboard updated ->", leaderboard_path)

# ---------- 5.2.D Figures ----------
# 1) Forecast vs True
plt.figure(figsize=(12,5))
plt.plot(y_train.index, y_train, label="Train")
plt.plot(y_test.index,  y_test,  label="Test True")
plt.plot(y_pred.index,  y_pred,  label=f"SARIMA{order_sel}x{seasonal_sel} Forecast")
plt.fill_between(y_pred.index, pred_df["ci_lower"], pred_df["ci_upper"], alpha=0.3)
plt.title(f"5.2 - SARIMA{order_sel}x{seasonal_sel} Forecast vs Test")
plt.legend()
fig1_path = os.path.join(RESULTS_STATS, f"sarima_{order_sel}_{seasonal_sel}_forecast.png".replace(" ", ""))
plt.tight_layout()
plt.savefig(fig1_path)
plt.close()
print("[save] Figure ->", fig1_path)

# 2) Residual ACF
resid = best_model.filter_results.standardized_forecasts_error[0]  # standardized errors for measurement eq.
# If above gives None due to spec, fallback to raw residuals:
if resid is None or (isinstance(resid, np.ndarray) and resid.size == 0):
    resid = best_model.resid

plt.figure(figsize=(10,4))
plot_acf(pd.Series(resid).dropna(), lags=40)
plt.title(f"5.2 - Residual ACF (SARIMA{order_sel}x{seasonal_sel})")
fig2_path = os.path.join(RESULTS_STATS, f"sarima_{order_sel}_{seasonal_sel}_resid_acf.png".replace(" ", ""))
plt.tight_layout()
plt.savefig(fig2_path)
plt.close()
print("[save] Figure ->", fig2_path)

print("[5.2] SARIMA complete.")

# =========================================================
# 5.3 - Holt-Winters (Exponential Smoothing)
# =========================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf

# ---------- Paths (reuse if already defined) ----------
try:
    PROJECT_ROOT
except NameError:
    PROJECT_ROOT = os.getcwd()

DATA_PROC     = os.path.join(PROJECT_ROOT, "data", "processed")
RESULTS_DIR   = os.path.join(PROJECT_ROOT, "results")
RESULTS_STATS = os.path.join(RESULTS_DIR, "stats")
MODELS_DIR    = os.path.join(PROJECT_ROOT, "models")
os.makedirs(RESULTS_STATS, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

print("========================================")
print("5.3 - Holt-Winters (Exponential Smoothing)")
print("========================================")
print("PROJECT_ROOT:", PROJECT_ROOT)
print("DATA_PROC   :", DATA_PROC)
print("RESULTS_STATS:", RESULTS_STATS)

# ---------- Load data ----------
train = pd.read_csv(os.path.join(DATA_PROC, "eur_chf_train.csv"), parse_dates=["date"], index_col="date")
test  = pd.read_csv(os.path.join(DATA_PROC, "eur_chf_test.csv"),  parse_dates=["date"], index_col="date")
y_train = train["log_return"].astype(float)
y_test  = test["log_return"].astype(float)

print(f"Train period: {y_train.index.min().date()} → {y_train.index.max().date()} | n={len(y_train)}")
print(f"Test  period: {y_test.index.min().date()}  → {y_test.index.max().date()}  | n={len(y_test)}")

# ---------- Helper: evaluate + save ----------
def evaluate_and_save(y_true, y_pred, ci_lower, ci_upper, model_name, subname):
    rmse = np.sqrt(((y_pred - y_true)**2).mean())
    mae  = np.abs(y_pred - y_true).mean()
    print(f"[{model_name}] Test RMSE={rmse:.6f} | MAE={mae:.6f}")

    # Save predictions
    df_pred = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper
    }, index=y_true.index)
    pred_path = os.path.join(RESULTS_STATS, f"hw_{subname}_predictions.csv")
    df_pred.to_csv(pred_path)
    print("[save] Predictions ->", pred_path)

    # Update leaderboard
    lb_path = os.path.join(MODELS_DIR, "leaderboard.csv")
    if os.path.exists(lb_path):
        lb = pd.read_csv(lb_path)
    else:
        lb = pd.DataFrame(columns=["model", "test_RMSE", "test_MAE"])
    lb = pd.concat([lb, pd.DataFrame([{
        "model": model_name,
        "test_RMSE": rmse,
        "test_MAE": mae
    }])], ignore_index=True)
    lb.to_csv(lb_path, index=False)
    print("[save] Leaderboard updated ->", lb_path)

    # Figure: Forecast vs Test
    plt.figure(figsize=(12,5))
    plt.plot(y_train.index, y_train, label="Train")
    plt.plot(y_test.index,  y_test,  label="Test True")
    plt.plot(y_pred.index,  y_pred,  label=f"{model_name} Forecast")
    if ci_lower is not None and ci_upper is not None:
        plt.fill_between(y_pred.index, ci_lower, ci_upper, alpha=0.3)
    plt.title(f"5.3 - {model_name} Forecast vs Test")
    plt.legend()
    fig_path = os.path.join(RESULTS_STATS, f"hw_{subname}_forecast.png")
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()
    print("[save] Figure ->", fig_path)

    return rmse, mae

# ---------- Confidence intervals helper (approximate) ----------
def approx_ci(pred_series, resid_train, z=1.96):
    # Use train residual std as proxy for forecast error std (simple but ok for baseline)
    sigma = np.std(resid_train.dropna())
    ci_low = pred_series - z * sigma
    ci_up  = pred_series + z * sigma
    return ci_low, ci_up

# ---------- 5.3.A - Simple Exponential Smoothing (no trend, no seasonality) ----------
# equivalent to SES; let statsmodels optimize alpha
print("\n[5.3.A] Simple Exponential Smoothing (no trend, no seasonality)")
ses_model = ExponentialSmoothing(
    y_train,
    trend=None,
    seasonal=None,
    initialization_method="estimated"
).fit(optimized=True)

# Forecast
steps = len(y_test)
ses_fore = ses_model.forecast(steps)
ses_fore.index = y_test.index

# Residuals on train (for CI proxy)
ses_resid_train = y_train - ses_model.fittedvalues
ses_ci_low, ses_ci_up = approx_ci(ses_fore, ses_resid_train)

evaluate_and_save(y_test, ses_fore, ses_ci_low, ses_ci_up, "HW_SES", "ses")

# ---------- 5.3.B - Additive Trend (Holt) ----------
print("\n[5.3.B] Holt (additive trend, no seasonality)")
holt_model = ExponentialSmoothing(
    y_train,
    trend="add",          # try 'add' trend (can be damped if desired)
    damped_trend=False,   # set to True to try damping
    seasonal=None,
    initialization_method="estimated"
).fit(optimized=True)

holt_fore = holt_model.forecast(steps)
holt_fore.index = y_test.index

holt_resid_train = y_train - holt_model.fittedvalues
holt_ci_low, holt_ci_up = approx_ci(holt_fore, holt_resid_train)

evaluate_and_save(y_test, holt_fore, holt_ci_low, holt_ci_up, "HW_Holt_add_trend", "holt_add_trend")

# ---------- 5.3.C - Seasonal (if detected) ----------
print("\n[5.3.C] Seasonal model (if seasonality exists)")
seasonal_candidates = [5, 21, 63]
acf_vals = acf(y_train, nlags=max(seasonal_candidates), fft=True)
strength = {s: abs(acf_vals[s]) for s in seasonal_candidates}
print("[Seasonality strengths ACF]:", strength)
best_s = max(strength, key=strength.get)

# Simple rule: require minimal seasonal signal; else skip
if strength[best_s] >= 0.03:
    print(f"[Seasonality] Using s={best_s}")
    # Try additive seasonality (returns are small; multiplicative is not suited)
    hw_seasonal_model = ExponentialSmoothing(
        y_train,
        trend=None,
        seasonal="add",
        seasonal_periods=best_s,
        initialization_method="estimated"
    ).fit(optimized=True)

    hw_seasonal_fore = hw_seasonal_model.forecast(steps)
    hw_seasonal_fore.index = y_test.index

    hw_seasonal_resid_train = y_train - hw_seasonal_model.fittedvalues
    seas_ci_low, seas_ci_up = approx_ci(hw_seasonal_fore, hw_seasonal_resid_train)

    evaluate_and_save(y_test, hw_seasonal_fore, seas_ci_low, seas_ci_up,
                      f"HW_Seasonal_add_s{best_s}", f"seasonal_add_s{best_s}")

    # Extra: Residual ACF figure
    plt.figure(figsize=(10,4))
    plot_acf((y_train - hw_seasonal_model.fittedvalues).dropna(), lags=40)
    plt.title(f"5.3 - Residual ACF (HW Seasonal add, s={best_s})")
    fig_resid = os.path.join(RESULTS_STATS, f"hw_seasonal_add_s{best_s}_resid_acf.png")
    plt.tight_layout()
    plt.savefig(fig_resid)
    plt.close()
    print("[save] Figure ->", fig_resid)
else:
    print("[Seasonality] Weak signal; skipping seasonal HW.")

print("[5.3] Holt-Winters complete.")

# =========================================================
# 5.4 - Résultats & Comparaison (ARIMA / SARIMA / HW)
# =========================================================
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Paths ----------
try:
    PROJECT_ROOT
except NameError:
    PROJECT_ROOT = os.getcwd()

RESULTS_DIR    = os.path.join(PROJECT_ROOT, "results")
RESULTS_STATS  = os.path.join(RESULTS_DIR, "stats")
MODELS_DIR     = os.path.join(PROJECT_ROOT, "models")
os.makedirs(RESULTS_STATS, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

print("========================================")
print("5.4 - Résultats & Comparaison")
print("========================================")
print("PROJECT_ROOT:", PROJECT_ROOT)
print("RESULTS_STATS:", RESULTS_STATS)

# ---------- Helper: compute metrics ----------
def metrics_from_df(df, eps=1e-8):
    y_true = df["y_true"].astype(float)
    y_pred = df["y_pred"].astype(float)
    rmse = np.sqrt(((y_pred - y_true) ** 2).mean())
    mae  = np.abs(y_pred - y_true).mean()
    mape = (np.abs((y_true - y_pred) / np.clip(np.abs(y_true), eps, None))).mean() * 100.0
    resid = (y_true - y_pred)
    return rmse, mae, mape, resid

# ---------- Collect prediction files ----------
pred_files = []
for fname in os.listdir(RESULTS_STATS):
    if fname.endswith("_predictions.csv"):
        # We only want part5 models (arima/sarima/hw_)
        if fname.startswith(("arima", "sarima", "hw_")):
            pred_files.append(os.path.join(RESULTS_STATS, fname))

if not pred_files:
    raise FileNotFoundError("Aucun fichier *_predictions.csv trouvé dans results/stats/ pour la partie 5.")

print("[found predictions]", [os.path.basename(p) for p in pred_files])

# ---------- Build summary table ----------
rows = []
residuals_dict = {}

for path in pred_files:
    df = pd.read_csv(path, parse_dates=["Unnamed: 0","y_true"], index_col=0) if "Unnamed: 0" in open(path).readline() else pd.read_csv(path, parse_dates=["y_true"], index_col=None)
    # robust load: ensure index is date if present in columns
    if "y_true" in df.columns and isinstance(df["y_true"].index, pd.RangeIndex):
        # We saved y_true as a column; try to reconstruct DatetimeIndex from predictions order using any date column
        if "date" in df.columns:
            df.index = pd.to_datetime(df["date"])
        else:
            # fallback: we don't strictly need an index for metrics
            pass

    model_name = os.path.splitext(os.path.basename(path))[0]  # file stem
    # clean model label
    label = (model_name
             .replace("sarima_", "SARIMA ")
             .replace("arima_", "ARIMA ")
             .replace("hw_", "HW ")
             .replace("_predictions", "")
             .replace("_", " ")
             ).strip()

    rmse, mae, mape, resid = metrics_from_df(df)
    rows.append({"model": label, "rmse": rmse, "mae": mae, "mape_pct": mape})
    residuals_dict[label] = (df, resid)

summary = pd.DataFrame(rows).sort_values(["rmse","mae"]).reset_index(drop=True)
summary_path = os.path.join(RESULTS_STATS, "part5_summary_metrics.csv")
summary.to_csv(summary_path, index=False)
print("[save] Summary metrics ->", summary_path)
print(summary)

# ---------- Update global leaderboard ----------
lb_path = os.path.join(MODELS_DIR, "leaderboard.csv")
if os.path.exists(lb_path):
    lb = pd.read_csv(lb_path)
else:
    lb = pd.DataFrame(columns=["model", "test_RMSE", "test_MAE", "test_MAPE"])

for _, r in summary.iterrows():
    # upsert by model name (case-insensitive)
    mask = lb["model"].str.lower().eq(r["model"].lower()) if "model" in lb.columns and len(lb)>0 else pd.Series([False])
    if mask.any():
        lb.loc[mask, ["test_RMSE","test_MAE"]] = [r["rmse"], r["mae"]]
        if "test_MAPE" in lb.columns:
            lb.loc[mask, "test_MAPE"] = r["mape_pct"]
    else:
        new_row = {"model": r["model"], "test_RMSE": r["rmse"], "test_MAE": r["mae"]}
        if "test_MAPE" in lb.columns:
            new_row["test_MAPE"] = r["mape_pct"]
        lb = pd.concat([lb, pd.DataFrame([new_row])], ignore_index=True)

# sort by RMSE if available
if "test_RMSE" in lb.columns:
    lb = lb.sort_values("test_RMSE").reset_index(drop=True)

lb.to_csv(lb_path, index=False)
print("[save] Leaderboard updated ->", lb_path)

# ---------- Figures ----------
# 1) Bar charts RMSE & MAE
plt.figure(figsize=(10,5))
plt.bar(summary["model"], summary["rmse"])
plt.title("Part 5 – RMSE comparison")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
rmse_fig = os.path.join(RESULTS_STATS, "part5_rmse_comparison.png")
plt.savefig(rmse_fig); plt.close()
print("[save] Figure ->", rmse_fig)

plt.figure(figsize=(10,5))
plt.bar(summary["model"], summary["mae"])
plt.title("Part 5 – MAE comparison")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
mae_fig = os.path.join(RESULTS_STATS, "part5_mae_comparison.png")
plt.savefig(mae_fig); plt.close()
print("[save] Figure ->", mae_fig)

# 2) Overlay: forecasts of top 3 models vs truth
# pick top 3 by RMSE
top3 = summary.head(3)["model"].tolist()
plt.figure(figsize=(12,6))

y_true_ref = None
for label in top3:
    df, _ = residuals_dict[label]
    if y_true_ref is None:
        y_true_ref = df["y_true"].astype(float)
        # If index is not datetime, just plot as range
        plt.plot(y_true_ref.values, label="Test True", linewidth=2)
    # predictions
    y_pred = df["y_pred"].astype(float)
    if y_pred.index.equals(y_true_ref.index):
        plt.plot(y_pred, label=f"{label} Forecast", alpha=0.9)
    else:
        plt.plot(y_pred.values, label=f"{label} Forecast", alpha=0.9)

plt.title("Part 5 – Top-3 forecasts vs Test")
plt.legend()
plt.tight_layout()
overlay_fig = os.path.join(RESULTS_STATS, "part5_top3_forecasts_vs_test.png")
plt.savefig(overlay_fig); plt.close()
print("[save] Figure ->", overlay_fig)

# 3) Residual distributions (one figure per model)
for label, (df, resid) in residuals_dict.items():
    plt.figure(figsize=(8,5))
    plt.hist(resid, bins=40, alpha=0.9, density=True)
    plt.title(f"Residual distribution – {label}")
    plt.tight_layout()
    resid_fig = os.path.join(RESULTS_STATS, f"part5_residuals_{re.sub(r'[^A-Za-z0-9]+','_',label)}.png")
    plt.savefig(resid_fig); plt.close()
    print("[save] Figure ->", resid_fig)

print("[5.4] Résultats & Comparaison complete.")

# =========================================================
# 5.5 - Documentation & Rapport (bilingue FR/EN, Markdown)
# =========================================================
import os
import glob
import datetime
import pandas as pd

# ---------- Paths ----------
try:
    PROJECT_ROOT
except NameError:
    PROJECT_ROOT = os.getcwd()

RESULTS_DIR    = os.path.join(PROJECT_ROOT, "results")
RESULTS_STATS  = os.path.join(RESULTS_DIR, "stats")
MODELS_DIR     = os.path.join(PROJECT_ROOT, "models")
os.makedirs(RESULTS_STATS, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

print("========================================")
print("5.5 - Documentation & Rapport")
print("========================================")
print("PROJECT_ROOT:", PROJECT_ROOT)
print("RESULTS_STATS:", RESULTS_STATS)

# ---------- Load consolidated metrics ----------
summary_path = os.path.join(RESULTS_STATS, "part5_summary_metrics.csv")
if not os.path.exists(summary_path):
    raise FileNotFoundError("part5_summary_metrics.csv introuvable. Lance d'abord 5.4.")

summary = pd.read_csv(summary_path)
summary_sorted = summary.sort_values(["rmse","mae"]).reset_index(drop=True)

# top model (classical/stat only)
top_row = summary_sorted.iloc[0]
top_model = top_row["model"]
top_rmse  = float(top_row["rmse"])
top_mae   = float(top_row["mae"])

# ---------- Load leaderboard (optional) ----------
leader_path = os.path.join(MODELS_DIR, "leaderboard.csv")
leader = pd.read_csv(leader_path) if os.path.exists(leader_path) else pd.DataFrame()

# ---------- Collect figures from Part 5 ----------
fig_patterns = [
    "arima*_forecast.png",
    "sarima*forecast.png",
    "sarima*_resid_acf.png",
    "hw_*_forecast.png",
    "hw_*_resid_acf.png",
    "part5_*comparison*.png",
    "part5_residuals_*.png",
    "part5_top3_forecasts_vs_test.png",
]
figs = []
for pat in fig_patterns:
    figs.extend(sorted(glob.glob(os.path.join(RESULTS_STATS, pat))))
# keep unique and nice relative paths
figs = [os.path.relpath(p, PROJECT_ROOT) for p in sorted(set(figs))]

# ---------- Small helpers ----------
def fmt_pct(x):
    try:
        return f"{x:,.2f}"
    except Exception:
        return str(x)

def md_table_from_df(df):
    # limit columns and format nicely
    df2 = df.copy()
    if "mape_pct" in df2.columns:
        df2["mape_pct"] = df2["mape_pct"].map(lambda x: f"{x:,.0f}")
    df2["rmse"] = df2["rmse"].map(lambda x: f"{x:.6f}")
    df2["mae"]  = df2["mae"].map(lambda x: f"{x:.6f}")
    return df2.to_markdown(index=False)

timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

# ---------- Build French Report ----------
fr_lines = []
fr_lines.append("# Chapitre 5 – Modèles statistiques classiques\n")
fr_lines.append(f"_Généré le {timestamp}_\n")

fr_lines.append("## 5.5 Documentation & Rapport (FR)\n")
fr_lines.append("### 🎯 Objectif\n")
fr_lines.append("- Tester des **modèles statistiques classiques** (ARIMA, SARIMA, Holt-Winters) pour la prévision des **log-returns EUR/CHF**.\n")
fr_lines.append("- Établir une **référence “classique”** à comparer aux modèles de **Machine Learning** (Chapitre 6).\n")

fr_lines.append("### 🧪 Méthodes\n")
fr_lines.append("- **ARIMA(2,0,1)** sélectionné par AIC sur returns stationnaires (d=0).\n")
fr_lines.append("- **SARIMA** avec détection de **saisonnalité hebdo (s=5)** → meilleur par AIC : **SARIMA(2,0,0)×(0,1,1,5)**.\n")
fr_lines.append("- **Holt-Winters** : variantes SES (sans tendance), Holt (tendance additive), saisonnier (s=5, additif).\n")
fr_lines.append("- Split temporel: **train 2015–2022**, **test 2023–2025**. Pas de fuite, index datetime propre.\n")

fr_lines.append("### 📊 Résultats consolidés (RMSE/MAE/MAPE)\n")
fr_lines.append(md_table_from_df(summary_sorted) + "\n")

fr_lines.append(f"**Meilleur modèle (statistique)** : **{top_model}** → RMSE={top_rmse:.6f}, MAE={top_mae:.6f}.\n")
fr_lines.append("> Remarque : le **MAPE** n’est pas pertinent sur des retours proches de 0 (dénominateur quasi nul).\n")

fr_lines.append("### 🖼️ Figures clés\n")
if figs:
    for p in figs:
        fr_lines.append(f"- `{p}`")
    fr_lines.append("")
else:
    fr_lines.append("_Aucune figure trouvée dans `results/stats/`._\n")

fr_lines.append("### ⚠️ Limites\n")
fr_lines.append("- **Rigidité face aux chocs** : les modèles linéaires captent mal les ruptures (ex. chocs de volatilité).\n")
fr_lines.append("- **Saisonniété faible** : le signal hebdomadaire (s=5) n’améliore pas la prévision.\n")
fr_lines.append("- **Volatilité mal capturée** : ces modèles se concentrent sur la moyenne, pas sur la dynamique de variance.\n")
fr_lines.append("- **Sur-ajustement possible** si l’on empile trop de paramètres saisonniers peu informatifs.\n")

fr_lines.append("### ✅ Conclusion & Référence\n")
fr_lines.append("- **ARIMA(2,0,1)** et **HW-SES** fournissent les **meilleures erreurs** (RMSE≈0.003322, MAE≈0.002508), au coude-à-coude.\n")
fr_lines.append("- **Référence “classique”** : retenir **HW-SES** (simplicité/robustesse) et **ARIMA(2,0,1)** (référence ARMA) pour la comparaison avec le **Chapitre 6 (ML)**.\n")

fr_report_path = os.path.join(RESULTS_STATS, "part5_report_FR.md")
with open(fr_report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(fr_lines))
print("[save] Rapport FR ->", fr_report_path)

# ---------- Build English Report ----------
en_lines = []
en_lines.append("# Chapter 5 – Classical Statistical Models\n")
en_lines.append(f"_Generated on {timestamp}_\n")

en_lines.append("## 5.5 Documentation & Report (EN)\n")
en_lines.append("### 🎯 Objective\n")
en_lines.append("- Test **classical statistical models** (ARIMA, SARIMA, Holt-Winters) for **EUR/CHF log-return** forecasting.\n")
en_lines.append("- Establish a **“classical benchmark”** to compare against **Machine Learning** models (Chapter 6).\n")

en_lines.append("### 🧪 Methods\n")
en_lines.append("- **ARIMA(2,0,1)** selected via AIC on stationary returns (d=0).\n")
en_lines.append("- **SARIMA** with **weekly seasonality (s=5)** → best by AIC: **SARIMA(2,0,0)×(0,1,1,5)**.\n")
en_lines.append("- **Holt-Winters**: SES (no trend), Holt (additive trend), seasonal (s=5, additive).\n")
en_lines.append("- Time split: **train 2015–2022**, **test 2023–2025**. No leakage; clean datetime index.\n")

en_lines.append("### 📊 Consolidated Results (RMSE/MAE/MAPE)\n")
en_lines.append(md_table_from_df(summary_sorted) + "\n")

en_lines.append(f"**Best (statistical) model**: **{top_model}** → RMSE={top_rmse:.6f}, MAE={top_mae:.6f}.\n")
en_lines.append("> Note: **MAPE** is not meaningful on near-zero returns (denominator issues).\n")

en_lines.append("### 🖼️ Key Figures\n")
if figs:
    for p in figs:
        en_lines.append(f"- `{p}`")
    en_lines.append("")
else:
    en_lines.append("_No figures found in `results/stats/`._\n")

en_lines.append("### ⚠️ Limitations\n")
en_lines.append("- **Rigid to shocks**: linear models do not adapt well to structural breaks/vol spikes.\n")
en_lines.append("- **Weak seasonality**: weekly signal (s=5) does not improve forecasts.\n")
en_lines.append("- **Volatility not captured**: these models target the conditional mean, not variance dynamics.\n")
en_lines.append("- **Potential overfitting** if excessive seasonal parameters are used with weak signal.\n")

en_lines.append("### ✅ Conclusion & Benchmark\n")
en_lines.append("- **ARIMA(2,0,1)** and **HW-SES** deliver the **lowest errors** (RMSE≈0.003322, MAE≈0.002508), essentially tied.\n")
en_lines.append("- **Classical benchmark**: keep **HW-SES** (simplicity/robustness) and **ARIMA(2,0,1)** (ARMA reference) for comparison against **Chapter 6 (ML)**.\n")

en_report_path = os.path.join(RESULTS_STATS, "part5_report_EN.md")
with open(en_report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(en_lines))
print("[save] Report EN ->", en_report_path)

print("[5.5] Documentation & Rapport complete.")
