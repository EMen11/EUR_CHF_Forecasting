# =========================================================
# 7.0 - Setup
# =========================================================

import os
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# --- Dossiers ---
PROJECT_ROOT = "/Users/eliemenassa/Desktop/Projet Forecasting"
DATA_PROC = os.path.join(PROJECT_ROOT, "data", "processed")
RESULTS_ML_ADV = os.path.join(PROJECT_ROOT, "results", "ml_advanced")
os.makedirs(RESULTS_ML_ADV, exist_ok=True)

# --- Leaderboard safeguard ---
LEADERBOARD = os.path.join(PROJECT_ROOT, "models", "leaderboard.csv")
os.makedirs(os.path.dirname(LEADERBOARD), exist_ok=True)

def update_leaderboard(row_dict):
    cols = ["model","rmse","mae","mape_pct"]
    df_new = pd.DataFrame([row_dict], columns=cols)
    if os.path.exists(LEADERBOARD):
        df = pd.read_csv(LEADERBOARD)
        df = pd.concat([df, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(LEADERBOARD, index=False)
    print(f"[leaderboard] updated → {LEADERBOARD}")




# =========================================================
# 7.1 - Random Forest Regressor (RF)
# =========================================================

import json
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance

# ---- Add this near the top of 7.1 (after imports) ----
TARGET_CANDIDATES = ["target_next_return", "target", "log_return", "r_t", "y"]


# --- Dossiers de sortie ---
FIG_DIR = os.path.join(RESULTS_ML_ADV, "figs")
PRED_DIR = os.path.join(RESULTS_ML_ADV, "preds")
CV_DIR   = os.path.join(RESULTS_ML_ADV, "cv")
IMP_DIR  = os.path.join(RESULTS_ML_ADV, "imp")
for d in [FIG_DIR, PRED_DIR, CV_DIR, IMP_DIR]:
    os.makedirs(d, exist_ok=True)

def _read_df(path):
    df = pd.read_csv(path, parse_dates=["date"])
    return df.set_index("date").sort_index()
    # y (target) – detect the correct target column
   

def _load_Xy():
    # X (features)
    X_tr = _read_df(os.path.join(DATA_PROC, "eur_chf_features_train.csv"))
    X_te = _read_df(os.path.join(DATA_PROC, "eur_chf_features_test.csv"))
    y_tr_df = _read_df(os.path.join(DATA_PROC, "eur_chf_train_y.csv"))
    y_te_df = _read_df(os.path.join(DATA_PROC, "eur_chf_test_y.csv"))

    ycol = None
    for c in TARGET_CANDIDATES:
        if c in y_tr_df.columns and c in y_te_df.columns:
            ycol = c
            break
    if ycol is None:
        raise ValueError(f"No target column found in y files. "
                         f"Columns train={list(y_tr_df.columns)}, test={list(y_te_df.columns)}")

    y_tr = y_tr_df[ycol].rename("target")
    y_te = y_te_df[ycol].rename("target")

      # remove any target-like column from X
    for c in TARGET_CANDIDATES:
        if c in X_tr.columns: X_tr = X_tr.drop(columns=[c])
        if c in X_te.columns: X_te = X_te.drop(columns=[c])


    # alignement par date
    X_tr, y_tr = X_tr.align(y_tr, join="inner", axis=0)
    X_te, y_te = X_te.align(y_te, join="inner", axis=0)
    print(f"[7.1] X_train:{X_tr.shape}  X_test:{X_te.shape}  | y_train:{y_tr.shape}  y_test:{y_te.shape}")
    return X_tr, y_tr, X_te, y_te

X_train_df, y_train_s, X_test_df, y_test_s = _load_Xy()
feature_names = X_train_df.columns.tolist()

# ------------------- Modélisation RF + GridSearch -------------------
tscv = TimeSeriesSplit(n_splits=5)
rf = RandomForestRegressor(random_state=42, n_jobs=-1)

param_grid = {
    "n_estimators":   [300, 600],
    "max_depth":      [None, 6, 10],
    "min_samples_leaf":[1, 5],
    "max_features":   ["sqrt"],
}

# scoring multi-métriques; on refit sur RMSE
scoring = {
    "rmse": "neg_root_mean_squared_error",
    "mae":  "neg_mean_absolute_error"
}

gcv = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    scoring=scoring,
    refit="rmse",
    cv=tscv,
    n_jobs=-1,
    verbose=0,
    return_train_score=False
)

gcv.fit(X_train_df.values, y_train_s.values)
best_rf = gcv.best_estimator_
best_params = gcv.best_params_
best_idx = gcv.best_index_

# CV metrics (moyenne ± std) pour la meilleure config
mean_rmse = -gcv.cv_results_["mean_test_rmse"][best_idx]
std_rmse  =  gcv.cv_results_["std_test_rmse"][best_idx]
mean_mae  = -gcv.cv_results_["mean_test_mae"][best_idx]
std_mae   =  gcv.cv_results_["std_test_mae"][best_idx]

cv_summary = pd.DataFrame([{
    "model": "ML_RF",
    "mean_rmse": mean_rmse, "std_rmse": std_rmse,
    "mean_mae":  mean_mae,  "std_mae":  std_mae,
    "best_params": json.dumps(best_params)
}])
cv_summary.to_csv(os.path.join(CV_DIR, "part7_rf_cv_metrics.csv"), index=False)
pd.DataFrame(gcv.cv_results_).to_csv(os.path.join(CV_DIR, "part7_rf_cv_full.csv"), index=False)

print(f"[7.1][CV] RMSE={mean_rmse:.6f} ± {std_rmse:.6f} | MAE={mean_mae:.6f} ± {std_mae:.6f}")
print(f"[7.1] Best params: {best_params}")

# ------------------- Évaluation Test -------------------
y_pred_test  = best_rf.predict(X_test_df.values)
y_pred_train = best_rf.predict(X_train_df.values)

def _metrics(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    eps  = 1e-12
    mape = float(np.mean(np.abs((y_true + eps) - (y_pred + eps)) / (np.abs(y_true) + eps)))
    return rmse, mae, mape

rmse_te, mae_te, mape_te = _metrics(y_test_s.values, y_pred_test)
rmse_tr, mae_tr, mape_tr = _metrics(y_train_s.values, y_pred_train)

print(f"[7.1][TEST] RMSE={rmse_te:.6f} | MAE={mae_te:.6f} | MAPE={mape_te:.6f}")
print(f"[7.1][TRAIN] RMSE={rmse_tr:.6f} | MAE={mae_tr:.6f} | MAPE={mape_tr:.6f}")

# ------------------- Sauvegardes: prédictions & résidus -------------------
preds_test_df = pd.DataFrame({
    "date": X_test_df.index, "y_true": y_test_s.values, "y_pred": y_pred_test
})
preds_test_df.to_csv(os.path.join(PRED_DIR, "part7_rf_test_predictions.csv"), index=False)

resid_test_df = pd.DataFrame({
    "date": X_test_df.index, "residual": y_test_s.values - y_pred_test
})
resid_test_df.to_csv(os.path.join(PRED_DIR, "part7_rf_test_residuals.csv"), index=False)

# ------------------- Figures: test vs pred, histogramme des résidus -------------------
plt.figure(figsize=(12,4))
plt.plot(X_test_df.index, y_test_s.values, label="y_true")
plt.plot(X_test_df.index, y_pred_test, label="y_pred")
plt.title("Part 7 - RF: Test vs Prediction")
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "part7_rf_test_vs_pred.png"), dpi=150); plt.close()

plt.figure(figsize=(6,4))
plt.hist(y_test_s.values - y_pred_test, bins=50)
plt.title("Part 7 - RF: Residuals (Test)")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "part7_rf_residuals_hist.png"), dpi=150); plt.close()

# ------------------- Permutation Importance (sur Test) -------------------
perm = permutation_importance(
    best_rf, X_test_df.values, y_test_s.values,
    n_repeats=30, random_state=42, n_jobs=-1,
    scoring="neg_root_mean_squared_error"
)
imp_df = pd.DataFrame({
    "feature": feature_names,
    "importance_mean": perm.importances_mean,
    "importance_std":  perm.importances_std
}).sort_values("importance_mean", ascending=False)
imp_df.to_csv(os.path.join(IMP_DIR, "part7_rf_permutation_importance.csv"), index=False)

top_k = min(15, len(imp_df))
plt.figure(figsize=(8, max(4, 0.35*top_k)))
plt.barh(imp_df.head(top_k)["feature"][::-1], imp_df.head(top_k)["importance_mean"][::-1])
plt.title("Part 7 - RF: Permutation Importance (Top)")
plt.tight_layout()
plt.savefig(os.path.join(IMP_DIR, "part7_rf_permimp.png"), dpi=150); plt.close()

# ------------------- Sauvegarde du modèle + leaderboard -------------------
joblib.dump(best_rf, os.path.join(RESULTS_ML_ADV, "part7_rf_best.joblib"))

update_leaderboard({
    "model": "ML_RF",
    "rmse": rmse_te,
    "mae": mae_te,
    "mape_pct": mape_te * 100.0
})

print("[7.1] Random Forest complete.")

# =========================================================
# 7.2 - Gradient Boosting Regressor (GBDT - sklearn)
# =========================================================

import json
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd
import os

# --- Dossiers de sortie (idempotent) ---
FIG_DIR = os.path.join(RESULTS_ML_ADV, "figs")
PRED_DIR = os.path.join(RESULTS_ML_ADV, "preds")
CV_DIR   = os.path.join(RESULTS_ML_ADV, "cv")
IMP_DIR  = os.path.join(RESULTS_ML_ADV, "imp")
for d in [FIG_DIR, PRED_DIR, CV_DIR, IMP_DIR]:
    os.makedirs(d, exist_ok=True)

# --- Candidats pour nom de cible (si non déjà défini) ---
try:
    TARGET_CANDIDATES
except NameError:
    TARGET_CANDIDATES = ["target_next_return", "target", "log_return", "r_t", "y"]

# --- Helpers de chargement (si non déjà définis) ---
def _read_df(path):
    df = pd.read_csv(path, parse_dates=["date"])
    return df.set_index("date").sort_index()

def _load_Xy():
    # X (features)
    X_tr = _read_df(os.path.join(DATA_PROC, "eur_chf_features_train.csv"))
    X_te = _read_df(os.path.join(DATA_PROC, "eur_chf_features_test.csv"))
    # y (target)
    y_tr_df = _read_df(os.path.join(DATA_PROC, "eur_chf_train_y.csv"))
    y_te_df = _read_df(os.path.join(DATA_PROC, "eur_chf_test_y.csv"))

    ycol = None
    for c in TARGET_CANDIDATES:
        if c in y_tr_df.columns and c in y_te_df.columns:
            ycol = c; break
    if ycol is None:
        raise ValueError(f"No target column found in y files. "
                         f"Columns train={list(y_tr_df.columns)}, test={list(y_te_df.columns)}")

    y_tr = y_tr_df[ycol].rename("target")
    y_te = y_te_df[ycol].rename("target")

    # retirer toute éventuelle cible dans X
    for c in TARGET_CANDIDATES:
        if c in X_tr.columns: X_tr = X_tr.drop(columns=[c])
        if c in X_te.columns: X_te = X_te.drop(columns=[c])

    # alignement par date
    X_tr, y_tr = X_tr.align(y_tr, join="inner", axis=0)
    X_te, y_te = X_te.align(y_te, join="inner", axis=0)

    print(f"[7.2] X_train:{X_tr.shape}  X_test:{X_te.shape}  | y_train:{y_tr.shape}  y_test:{y_te.shape}")
    return X_tr, y_tr, X_te, y_te

# Réutiliser X/y si déjà calculés dans 7.1, sinon recharger
if all(name in globals() for name in ["X_train_df","y_train_s","X_test_df","y_test_s"]):
    X_tr_df, y_tr_s, X_te_df, y_te_s = X_train_df, y_train_s, X_test_df, y_test_s
else:
    X_tr_df, y_tr_s, X_te_df, y_te_s = _load_Xy()

feature_names = X_tr_df.columns.tolist()

# ------------------- Modélisation GBDT + GridSearch -------------------
tscv = TimeSeriesSplit(n_splits=5)
gbdt = GradientBoostingRegressor(random_state=42)

param_grid = {
    "n_estimators":  [400, 800],
    "learning_rate": [0.03, 0.06],
    "max_depth":     [2, 3],
    "subsample":     [0.7, 1.0],
}

scoring = {
    "rmse": "neg_root_mean_squared_error",
    "mae":  "neg_mean_absolute_error"
}

gcv = GridSearchCV(
    estimator=gbdt,
    param_grid=param_grid,
    scoring=scoring,
    refit="rmse",
    cv=tscv,
    n_jobs=-1,
    verbose=0,
    return_train_score=False
)

gcv.fit(X_tr_df.values, y_tr_s.values)
best_gbdt = gcv.best_estimator_
best_params = gcv.best_params_
best_idx = gcv.best_index_

# CV metrics (moyenne ± std) pour la meilleure config
mean_rmse = -gcv.cv_results_["mean_test_rmse"][best_idx]
std_rmse  =  gcv.cv_results_["std_test_rmse"][best_idx]
mean_mae  = -gcv.cv_results_["mean_test_mae"][best_idx]
std_mae   =  gcv.cv_results_["std_test_mae"][best_idx]

cv_summary = pd.DataFrame([{
    "model": "ML_GBDT",
    "mean_rmse": mean_rmse, "std_rmse": std_rmse,
    "mean_mae":  mean_mae,  "std_mae":  std_mae,
    "best_params": json.dumps(best_params)
}])
cv_summary.to_csv(os.path.join(CV_DIR, "part7_gbdt_cv_metrics.csv"), index=False)
pd.DataFrame(gcv.cv_results_).to_csv(os.path.join(CV_DIR, "part7_gbdt_cv_full.csv"), index=False)

print(f"[7.2][CV] RMSE={mean_rmse:.6f} ± {std_rmse:.6f} | MAE={mean_mae:.6f} ± {std_mae:.6f}")
print(f"[7.2] Best params: {best_params}")

# ------------------- Évaluation sur Test -------------------
def _metrics(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    eps  = 1e-12
    mape = float(np.mean(np.abs((y_true + eps) - (y_pred + eps)) / (np.abs(y_true) + eps)))
    return rmse, mae, mape

y_pred_test  = best_gbdt.predict(X_te_df.values)
y_pred_train = best_gbdt.predict(X_tr_df.values)

rmse_te, mae_te, mape_te = _metrics(y_te_s.values, y_pred_test)
rmse_tr, mae_tr, mape_tr = _metrics(y_tr_s.values, y_pred_train)

print(f"[7.2][TEST] RMSE={rmse_te:.6f} | MAE={mae_te:.6f} | MAPE={mape_te:.6f}")
print(f"[7.2][TRAIN] RMSE={rmse_tr:.6f} | MAE={mae_tr:.6f} | MAPE={mape_tr:.6f}")

# ------------------- Sauvegardes: prédictions & résidus -------------------
preds_test_df = pd.DataFrame({
    "date": X_te_df.index, "y_true": y_te_s.values, "y_pred": y_pred_test
})
preds_test_df.to_csv(os.path.join(PRED_DIR, "part7_gbdt_test_predictions.csv"), index=False)

resid_test_df = pd.DataFrame({
    "date": X_te_df.index, "residual": y_te_s.values - y_pred_test
})
resid_test_df.to_csv(os.path.join(PRED_DIR, "part7_gbdt_test_residuals.csv"), index=False)

# ------------------- Figures: test vs pred, histogramme des résidus -------------------
plt.figure(figsize=(12,4))
plt.plot(X_te_df.index, y_te_s.values, label="y_true")
plt.plot(X_te_df.index, y_pred_test, label="y_pred")
plt.title("Part 7 - GBDT: Test vs Prediction")
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "part7_gbdt_test_vs_pred.png"), dpi=150); plt.close()

plt.figure(figsize=(6,4))
plt.hist(y_te_s.values - y_pred_test, bins=50)
plt.title("Part 7 - GBDT: Residuals (Test)")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "part7_gbdt_residuals_hist.png"), dpi=150); plt.close()

# ------------------- Importances de features (gain-based) -------------------
# sklearn GBDT expose .feature_importances_ (réduction MSE)
imp_values = getattr(best_gbdt, "feature_importances_", None)
if imp_values is not None:
    imp_df = pd.DataFrame({"feature": feature_names, "importance": imp_values}) \
                .sort_values("importance", ascending=False)
    imp_df.to_csv(os.path.join(IMP_DIR, "part7_gbdt_feature_importance.csv"), index=False)

    top_k = min(15, len(imp_df))
    plt.figure(figsize=(8, max(4, 0.35*top_k)))
    plt.barh(imp_df.head(top_k)["feature"][::-1], imp_df.head(top_k)["importance"][::-1])
    plt.title("Part 7 - GBDT: Feature Importance (Top)")
    plt.tight_layout()
    plt.savefig(os.path.join(IMP_DIR, "part7_gbdt_feature_importance.png"), dpi=150); plt.close()

# ------------------- Sauvegarde du modèle + leaderboard -------------------
joblib.dump(best_gbdt, os.path.join(RESULTS_ML_ADV, "part7_gbdt_best.joblib"))

update_leaderboard({
    "model": "ML_GBDT",
    "rmse": rmse_te,
    "mae": mae_te,
    "mape_pct": mape_te * 100.0
})

print("[7.2] Gradient Boosting complete.")

# =========================================================
# 7.3 - XGBoost & LightGBM (lite, fast & stable)
# =========================================================

import os, json, joblib, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error

FIG_DIR = os.path.join(RESULTS_ML_ADV, "figs")
PRED_DIR = os.path.join(RESULTS_ML_ADV, "preds")
CV_DIR   = os.path.join(RESULTS_ML_ADV, "cv")
IMP_DIR  = os.path.join(RESULTS_ML_ADV, "imp")
for d in [FIG_DIR, PRED_DIR, CV_DIR, IMP_DIR]:
    os.makedirs(d, exist_ok=True)

try:
    TARGET_CANDIDATES
except NameError:
    TARGET_CANDIDATES = ["target_next_return", "target", "log_return", "r_t", "y"]

def _read_df(path):
    df = pd.read_csv(path, parse_dates=["date"])
    return df.set_index("date").sort_index()

def _load_Xy():
    X_tr = _read_df(os.path.join(DATA_PROC, "eur_chf_features_train.csv"))
    X_te = _read_df(os.path.join(DATA_PROC, "eur_chf_features_test.csv"))
    y_tr_df = _read_df(os.path.join(DATA_PROC, "eur_chf_train_y.csv"))
    y_te_df = _read_df(os.path.join(DATA_PROC, "eur_chf_test_y.csv"))
    ycol = None
    for c in TARGET_CANDIDATES:
        if c in y_tr_df.columns and c in y_te_df.columns:
            ycol = c; break
    if ycol is None:
        raise ValueError(f"No target column found in y files. "
                         f"Columns train={list(y_tr_df.columns)}, test={list(y_te_df.columns)}")
    y_tr = y_tr_df[ycol].rename("target")
    y_te = y_te_df[ycol].rename("target")
    for c in TARGET_CANDIDATES:
        if c in X_tr.columns: X_tr = X_tr.drop(columns=[c])
        if c in X_te.columns: X_te = X_te.drop(columns=[c])
    X_tr, y_tr = X_tr.align(y_tr, join="inner", axis=0)
    X_te, y_te = X_te.align(y_te, join="inner", axis=0)
    print(f"[7.3-lite] X_train:{X_tr.shape}  X_test:{X_te.shape} | y_train:{y_tr.shape} y_test:{y_te.shape}")
    return X_tr, y_tr, X_te, y_te

if all(name in globals() for name in ["X_train_df","y_train_s","X_test_df","y_test_s"]):
    X_tr_df, y_tr_s, X_te_df, y_te_s = X_train_df, y_train_s, X_test_df, y_test_s
else:
    X_tr_df, y_tr_s, X_te_df, y_te_s = _load_Xy()

feature_names = X_tr_df.columns.tolist()
tscv = TimeSeriesSplit(n_splits=5)
gs_n_jobs = max(1, (os.cpu_count() or 2) // 2)  # évite la sur-parallélisation

def _metrics(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    eps  = 1e-12
    mape = float(np.mean(np.abs((y_true + eps) - (y_pred + eps)) / (np.abs(y_true) + eps)))
    return rmse, mae, mape

# ------------------- XGBoost (lite grid) -------------------
try:
    import xgboost as xgb

    xgb_est = xgb.XGBRegressor(
        n_jobs=1,                  # important: laisser GridSearch paralléliser
        random_state=42,
        objective="reg:squarederror",
        tree_method="hist",
        booster="gbtree",
        verbosity=0
    )

    xgb_grid = {
        "n_estimators":     [300, 600],
        "learning_rate":    [0.06, 0.03],
        "max_depth":        [3],          # peu profond = stable/rapide
        "subsample":        [0.8, 1.0],
        "colsample_bytree": [0.8],
        "reg_lambda":       [1.0, 5.0],
    }
    scoring = {"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error"}

    xgb_gcv = GridSearchCV(
        estimator=xgb_est, param_grid=xgb_grid, scoring=scoring, refit="rmse",
        cv=tscv, n_jobs=gs_n_jobs, verbose=0, return_train_score=False
    )
    xgb_gcv.fit(X_tr_df.values, y_tr_s.values)

    xgb_best = xgb_gcv.best_estimator_
    xgb_best_params = xgb_gcv.best_params_
    xgb_idx = xgb_gcv.best_index_

    xgb_mean_rmse = -xgb_gcv.cv_results_["mean_test_rmse"][xgb_idx]
    xgb_std_rmse  =  xgb_gcv.cv_results_["std_test_rmse"][xgb_idx]
    xgb_mean_mae  = -xgb_gcv.cv_results_["mean_test_mae"][xgb_idx]
    xgb_std_mae   =  xgb_gcv.cv_results_["std_test_mae"][xgb_idx]

    pd.DataFrame([{
        "model":"ML_XGB","mean_rmse":xgb_mean_rmse,"std_rmse":xgb_std_rmse,
        "mean_mae":xgb_mean_mae,"std_mae":xgb_std_mae,
        "best_params":json.dumps(xgb_best_params)
    }]).to_csv(os.path.join(CV_DIR,"part7_xgb_cv_metrics.csv"), index=False)
    pd.DataFrame(xgb_gcv.cv_results_).to_csv(os.path.join(CV_DIR,"part7_xgb_cv_full.csv"), index=False)

    print(f"[7.3-lite][XGB][CV] RMSE={xgb_mean_rmse:.6f} ± {xgb_std_rmse:.6f} | MAE={xgb_mean_mae:.6f} ± {xgb_std_mae:.6f}")
    print(f"[7.3-lite][XGB] Best params: {xgb_best_params}")

    y_pred_te = xgb_best.predict(X_te_df.values)
    y_pred_tr = xgb_best.predict(X_tr_df.values)
    xgb_rmse_te, xgb_mae_te, xgb_mape_te = _metrics(y_te_s.values, y_pred_te)

    print(f"[7.3-lite][XGB][TEST] RMSE={xgb_rmse_te:.6f} | MAE={xgb_mae_te:.6f} | MAPE={xgb_mape_te:.6f}")

    pd.DataFrame({"date":X_te_df.index,"y_true":y_te_s.values,"y_pred":y_pred_te}) \
        .to_csv(os.path.join(PRED_DIR,"part7_xgb_test_predictions.csv"), index=False)
    pd.DataFrame({"date":X_te_df.index,"residual":y_te_s.values - y_pred_te}) \
        .to_csv(os.path.join(PRED_DIR,"part7_xgb_test_residuals.csv"), index=False)

    plt.figure(figsize=(12,4))
    plt.plot(X_te_df.index, y_te_s.values, label="y_true")
    plt.plot(X_te_df.index, y_pred_te,   label="y_pred")
    plt.title("Part 7 - XGB: Test vs Prediction")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR,"part7_xgb_test_vs_pred.png"), dpi=150); plt.close()

    plt.figure(figsize=(6,4))
    plt.hist(y_te_s.values - y_pred_te, bins=50)
    plt.title("Part 7 - XGB: Residuals (Test)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR,"part7_xgb_residuals_hist.png"), dpi=150); plt.close()

    xgb_imp = getattr(xgb_best, "feature_importances_", None)
    if xgb_imp is not None:
        imp_df = pd.DataFrame({"feature":feature_names,"importance":xgb_imp}) \
                   .sort_values("importance", ascending=False)
        imp_df.to_csv(os.path.join(IMP_DIR,"part7_xgb_feature_importance.csv"), index=False)
        top_k = min(15, len(imp_df))
        plt.figure(figsize=(8, max(4, 0.35*top_k)))
        plt.barh(imp_df.head(top_k)["feature"][::-1], imp_df.head(top_k)["importance"][::-1])
        plt.title("Part 7 - XGB: Feature Importance (Top)")
        plt.tight_layout()
        plt.savefig(os.path.join(IMP_DIR,"part7_xgb_feature_importance.png"), dpi=150); plt.close()

    joblib.dump(xgb_best, os.path.join(RESULTS_ML_ADV,"part7_xgb_best.joblib"))
    update_leaderboard({"model":"ML_XGB","rmse":xgb_rmse_te,"mae":xgb_mae_te,"mape_pct":xgb_mape_te*100.0})
    print("[7.3-lite] XGBoost complete.")

except Exception as e:
    print(f"[7.3-lite] XGBoost skipped → {e}")

# ------------------- LightGBM (lite grid) -------------------
try:
    import lightgbm as lgb

    lgb_est = lgb.LGBMRegressor(
        random_state=42, n_jobs=1, objective="regression",
        force_row_wise=True,           # plus stable sur petits jeux
        feature_pre_filter=False,      # évite le pré-filtrage agressif
        verbosity=-1
    )

    lgb_grid = {
        "n_estimators":     [300, 600],
        "learning_rate":    [0.06, 0.03],
        "num_leaves":       [31],          # taille raisonnable
        "feature_fraction": [0.8, 1.0],
        "bagging_fraction": [0.8, 1.0],
        "min_data_in_leaf": [25],
        "min_gain_to_split":[0.0],
    }
    scoring = {"rmse":"neg_root_mean_squared_error", "mae":"neg_mean_absolute_error"}

    lgb_gcv = GridSearchCV(
        estimator=lgb_est, param_grid=lgb_grid, scoring=scoring, refit="rmse",
        cv=tscv, n_jobs=gs_n_jobs, verbose=0, return_train_score=False
    )
    lgb_gcv.fit(X_tr_df.values, y_tr_s.values)

    lgb_best = lgb_gcv.best_estimator_
    lgb_best_params = lgb_gcv.best_params_
    lgb_idx = lgb_gcv.best_index_

    lgb_mean_rmse = -lgb_gcv.cv_results_["mean_test_rmse"][lgb_idx]
    lgb_std_rmse  =  lgb_gcv.cv_results_["std_test_rmse"][lgb_idx]
    lgb_mean_mae  = -lgb_gcv.cv_results_["mean_test_mae"][lgb_idx]
    lgb_std_mae   =  lgb_gcv.cv_results_["std_test_mae"][lgb_idx]

    pd.DataFrame([{
        "model":"ML_LGBM","mean_rmse":lgb_mean_rmse,"std_rmse":lgb_std_rmse,
        "mean_mae":lgb_mean_mae,"std_mae":lgb_std_mae,
        "best_params":json.dumps(lgb_best_params)
    }]).to_csv(os.path.join(CV_DIR,"part7_lgbm_cv_metrics.csv"), index=False)
    pd.DataFrame(lgb_gcv.cv_results_).to_csv(os.path.join(CV_DIR,"part7_lgbm_cv_full.csv"), index=False)

    print(f"[7.3-lite][LGBM][CV] RMSE={lgb_mean_rmse:.6f} ± {lgb_std_rmse:.6f} | MAE={lgb_mean_mae:.6f} ± {lgb_std_mae:.6f}")
    print(f"[7.3-lite][LGBM] Best params: {lgb_best_params}")

    y_pred_te = lgb_best.predict(X_te_df.values)
    y_pred_tr = lgb_best.predict(X_tr_df.values)
    lgb_rmse_te, lgb_mae_te, lgb_mape_te = _metrics(y_te_s.values, y_pred_te)

    print(f"[7.3-lite][LGBM][TEST] RMSE={lgb_rmse_te:.6f} | MAE={lgb_mae_te:.6f} | MAPE={lgb_mape_te:.6f}")

    pd.DataFrame({"date":X_te_df.index,"y_true":y_te_s.values,"y_pred":y_pred_te}) \
        .to_csv(os.path.join(PRED_DIR,"part7_lgbm_test_predictions.csv"), index=False)
    pd.DataFrame({"date":X_te_df.index,"residual":y_te_s.values - y_pred_te}) \
        .to_csv(os.path.join(PRED_DIR,"part7_lgbm_test_residuals.csv"), index=False)

    plt.figure(figsize=(12,4))
    plt.plot(X_te_df.index, y_te_s.values, label="y_true")
    plt.plot(X_te_df.index, y_pred_te,   label="y_pred")
    plt.title("Part 7 - LGBM: Test vs Prediction")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR,"part7_lgbm_test_vs_pred.png"), dpi=150); plt.close()

    plt.figure(figsize=(6,4))
    plt.hist(y_te_s.values - y_pred_te, bins=50)
    plt.title("Part 7 - LGBM: Residuals (Test)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR,"part7_lgbm_residuals_hist.png"), dpi=150); plt.close()

    lgb_imp = getattr(lgb_best, "feature_importances_", None)
    if lgb_imp is not None:
        imp_df = pd.DataFrame({"feature":feature_names,"importance":lgb_imp}) \
                   .sort_values("importance", ascending=False)
        imp_df.to_csv(os.path.join(IMP_DIR,"part7_lgbm_feature_importance.csv"), index=False)
        top_k = min(15, len(imp_df))
        plt.figure(figsize=(8, max(4, 0.35*top_k)))
        plt.barh(imp_df.head(top_k)["feature"][::-1], imp_df.head(top_k)["importance"][::-1])
        plt.title("Part 7 - LGBM: Feature Importance (Top)")
        plt.tight_layout()
        plt.savefig(os.path.join(IMP_DIR,"part7_lgbm_feature_importance.png"), dpi=150); plt.close()

    joblib.dump(lgb_best, os.path.join(RESULTS_ML_ADV,"part7_lgbm_best.joblib"))
    update_leaderboard({"model":"ML_LGBM","rmse":lgb_rmse_te,"mae":lgb_mae_te,"mape_pct":lgb_mape_te*100.0})
    print("[7.3-lite] LightGBM complete.")

except Exception as e:
    print(f"[7.3-lite] LightGBM skipped → {e}")
    
    # =========================================================
# 7.4 - Stacking & Weighted Blending (RF + GBDT)
# =========================================================

import os, json, joblib, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.ensemble import StackingRegressor
from copy import deepcopy

# --- Output dirs (reuse Part 7 dirs) ---
FIG_DIR = os.path.join(RESULTS_ML_ADV, "figs")
PRED_DIR = os.path.join(RESULTS_ML_ADV, "preds")
CV_DIR   = os.path.join(RESULTS_ML_ADV, "cv")
for d in [FIG_DIR, PRED_DIR, CV_DIR]:
    os.makedirs(d, exist_ok=True)

# --- Helper: load X/y if not in memory ---
try:
    X_train_df, y_train_s, X_test_df, y_test_s
except NameError:
    # fallback loader
    try:
        TARGET_CANDIDATES
    except NameError:
        TARGET_CANDIDATES = ["target_next_return", "target", "log_return", "r_t", "y"]

    def _read_df(path):
        df = pd.read_csv(path, parse_dates=["date"])
        return df.set_index("date").sort_index()

    def _load_Xy():
        X_tr = _read_df(os.path.join(DATA_PROC, "eur_chf_features_train.csv"))
        X_te = _read_df(os.path.join(DATA_PROC, "eur_chf_features_test.csv"))
        y_tr_df = _read_df(os.path.join(DATA_PROC, "eur_chf_train_y.csv"))
        y_te_df = _read_df(os.path.join(DATA_PROC, "eur_chf_test_y.csv"))
        ycol = None
        for c in TARGET_CANDIDATES:
            if c in y_tr_df.columns and c in y_te_df.columns:
                ycol = c; break
        if ycol is None:
            raise ValueError(f"No target column found in y files. Columns train={list(y_tr_df.columns)}, test={list(y_te_df.columns)}")
        y_tr = y_tr_df[ycol].rename("target")
        y_te = y_te_df[ycol].rename("target")
        for c in TARGET_CANDIDATES:
            if c in X_tr.columns: X_tr = X_tr.drop(columns=[c])
            if c in X_te.columns: X_te = X_te.drop(columns=[c])
        X_tr, y_tr = X_tr.align(y_tr, join="inner", axis=0)
        X_te, y_te = X_te.align(y_te, join="inner", axis=0)
        print(f"[7.4] X_train:{X_tr.shape}  X_test:{X_te.shape}  | y_train:{y_tr.shape}  y_test:{y_te.shape}")
        return X_tr, y_tr, X_te, y_te

    X_train_df, y_train_s, X_test_df, y_test_s = _load_Xy()

feature_names = X_train_df.columns.tolist()
tscv = TimeSeriesSplit(n_splits=5)

def _metrics(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    eps  = 1e-12
    mape = float(np.mean(np.abs((y_true + eps) - (y_pred + eps)) / (np.abs(y_true) + eps)))
    return rmse, mae, mape

# --- Load best base models from 7.1 / 7.2 ---
RF_PATH  = os.path.join(RESULTS_ML_ADV, "part7_rf_best.joblib")
GB_PATH  = os.path.join(RESULTS_ML_ADV, "part7_gbdt_best.joblib")
if not (os.path.exists(RF_PATH) and os.path.exists(GB_PATH)):
    raise FileNotFoundError("Missing base models. Ensure 7.1 and 7.2 have run (part7_rf_best.joblib / part7_gbdt_best.joblib).")

rf_best = joblib.load(RF_PATH)
gb_best = joblib.load(GB_PATH)

# =========================================================
# 7.4.A - Stacking (final estimator = Ridge)
# =========================================================
stack_est = StackingRegressor(
    estimators=[("rf", deepcopy(rf_best)), ("gbdt", deepcopy(gb_best))],
    final_estimator=Ridge(alpha=1.0, random_state=42),
    passthrough=False, n_jobs=-1
)

# CV on train (TimeSeriesSplit)
cv_preds, cv_truth = [], []
for fold, (tr_idx, va_idx) in enumerate(tscv.split(X_train_df)):
    X_tr, X_va = X_train_df.iloc[tr_idx].values, X_train_df.iloc[va_idx].values
    y_tr, y_va = y_train_s.iloc[tr_idx].values, y_train_s.iloc[va_idx].values

    # fresh clones each fold to avoid leakage of fitted state
    model_fold = StackingRegressor(
        estimators=[("rf", deepcopy(rf_best)), ("gbdt", deepcopy(gb_best))],
        final_estimator=Ridge(alpha=1.0, random_state=42),
        passthrough=False, n_jobs=-1
    )
    model_fold.fit(X_tr, y_tr)
    cv_preds.append(model_fold.predict(X_va))
    cv_truth.append(y_va)

cv_preds = np.concatenate(cv_preds)
cv_truth = np.concatenate(cv_truth)
stack_cv_rmse, stack_cv_mae, stack_cv_mape = _metrics(cv_truth, cv_preds)

# Fit on full train and evaluate on test
stack_est.fit(X_train_df.values, y_train_s.values)
y_pred_stack_tr = stack_est.predict(X_train_df.values)
y_pred_stack_te = stack_est.predict(X_test_df.values)

stack_tr_rmse, stack_tr_mae, stack_tr_mape = _metrics(y_train_s.values, y_pred_stack_tr)
stack_te_rmse, stack_te_mae, stack_te_mape = _metrics(y_test_s.values,  y_pred_stack_te)

# Save artifacts
pd.DataFrame([{
    "model": "ML_STACK",
    "cv_rmse": stack_cv_rmse, "cv_mae": stack_cv_mae, "cv_mape": stack_cv_mape
}]).to_csv(os.path.join(CV_DIR, "part7_stack_cv_metrics.csv"), index=False)

pd.DataFrame({"date": X_test_df.index, "y_true": y_test_s.values, "y_pred": y_pred_stack_te}) \
  .to_csv(os.path.join(PRED_DIR, "part7_stack_test_predictions.csv"), index=False)
pd.DataFrame({"date": X_test_df.index, "residual": y_test_s.values - y_pred_stack_te}) \
  .to_csv(os.path.join(PRED_DIR, "part7_stack_test_residuals.csv"), index=False)

plt.figure(figsize=(12,4))
plt.plot(X_test_df.index, y_test_s.values, label="y_true")
plt.plot(X_test_df.index, y_pred_stack_te, label="y_pred")
plt.title("Part 7 - STACK: Test vs Prediction")
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "part7_stack_test_vs_pred.png"), dpi=150); plt.close()

plt.figure(figsize=(6,4))
plt.hist(y_test_s.values - y_pred_stack_te, bins=50)
plt.title("Part 7 - STACK: Residuals (Test)")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "part7_stack_residuals_hist.png"), dpi=150); plt.close()

joblib.dump(stack_est, os.path.join(RESULTS_ML_ADV, "part7_stack_best.joblib"))

update_leaderboard({"model": "ML_STACK", "rmse": stack_te_rmse, "mae": stack_te_mae, "mape_pct": stack_te_mape*100.0})
print(f"[7.4][STACK] CV_RMSE={stack_cv_rmse:.6f} | TEST_RMSE={stack_te_rmse:.6f} | TEST_MAE={stack_te_mae:.6f}")

# =========================================================
# 7.4.B - Weighted Blend (RF & GBDT), weights via CV on train
# =========================================================
weights = [0.2, 0.35, 0.5, 0.65, 0.8]  # w for RF; (1-w) for GBDT

# Build out-of-fold preds for RF & GBDT on train (TimeSeriesSplit)
oof_rf, oof_gb, oof_y = [], [], []
for fold, (tr_idx, va_idx) in enumerate(tscv.split(X_train_df)):
    X_tr, X_va = X_train_df.iloc[tr_idx].values, X_train_df.iloc[va_idx].values
    y_tr, y_va = y_train_s.iloc[tr_idx].values, y_train_s.iloc[va_idx].values

    rf_fold = deepcopy(rf_best)
    gb_fold = deepcopy(gb_best)
    rf_fold.fit(X_tr, y_tr)
    gb_fold.fit(X_tr, y_tr)

    oof_rf.append(rf_fold.predict(X_va))
    oof_gb.append(gb_fold.predict(X_va))
    oof_y.append(y_va)

oof_rf = np.concatenate(oof_rf)
oof_gb = np.concatenate(oof_gb)
oof_y  = np.concatenate(oof_y)

# Grid search for best weight on OOF preds
best_w, best_rmse = None, np.inf
for w in weights:
    blend = w*oof_rf + (1.0 - w)*oof_gb
    rmse, mae, _ = _metrics(oof_y, blend)
    if rmse < best_rmse:
        best_rmse = rmse
        best_w = w

# Fit base models on full train, make test blend
rf_full = deepcopy(rf_best).fit(X_train_df.values, y_train_s.values)
gb_full = deepcopy(gb_best).fit(X_train_df.values, y_train_s.values)

pred_rf_te = rf_full.predict(X_test_df.values)
pred_gb_te = gb_full.predict(X_test_df.values)
pred_blend_te = best_w*pred_rf_te + (1.0 - best_w)*pred_gb_te

blend_te_rmse, blend_te_mae, blend_te_mape = _metrics(y_test_s.values, pred_blend_te)

# Save artifacts
pd.DataFrame([{
    "model": "ML_BLEND",
    "best_weight_rf": best_w,
    "cv_oof_rmse": best_rmse
}]).to_csv(os.path.join(CV_DIR, "part7_blend_cv_metrics.csv"), index=False)

pd.DataFrame({"date": X_test_df.index, "y_true": y_test_s.values, "y_pred": pred_blend_te}) \
  .to_csv(os.path.join(PRED_DIR, "part7_blend_test_predictions.csv"), index=False)
pd.DataFrame({"date": X_test_df.index, "residual": y_test_s.values - pred_blend_te}) \
  .to_csv(os.path.join(PRED_DIR, "part7_blend_test_residuals.csv"), index=False)

plt.figure(figsize=(12,4))
plt.plot(X_test_df.index, y_test_s.values, label="y_true")
plt.plot(X_test_df.index, pred_blend_te, label="y_pred")
plt.title("Part 7 - BLEND: Test vs Prediction")
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "part7_blend_test_vs_pred.png"), dpi=150); plt.close()

plt.figure(figsize=(6,4))
plt.hist(y_test_s.values - pred_blend_te, bins=50)
plt.title("Part 7 - BLEND: Residuals (Test)")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "part7_blend_residuals_hist.png"), dpi=150); plt.close()

update_leaderboard({"model": "ML_BLEND", "rmse": blend_te_rmse, "mae": blend_te_mae, "mape_pct": blend_te_mape*100.0})
print(f"[7.4][BLEND] best_w_RF={best_w:.2f} | TEST_RMSE={blend_te_rmse:.6f} | TEST_MAE={blend_te_mae:.6f}")

# =========================================================
# 7.5 - Diagnostics avancés (qualité & robustesse)
# =========================================================
import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from copy import deepcopy
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.inspection import permutation_importance

# (optionnel) PDP/ICE
try:
    from sklearn.inspection import PartialDependenceDisplay
    HAS_PDP = True
except Exception:
    HAS_PDP = False

# (optionnel) ACF depuis statsmodels, sinon fallback numpy
try:
    from statsmodels.graphics.tsaplots import plot_acf
    HAS_SM = True
except Exception:
    HAS_SM = False

# ---------- Répertoires ----------
try:
    PROJECT_ROOT
except NameError:
    PROJECT_ROOT = "/Users/eliemenassa/Desktop/Projet Forecasting"

DATA_PROC       = os.path.join(PROJECT_ROOT, "data", "processed")
RESULTS_ML_ADV  = os.path.join(PROJECT_ROOT, "results", "ml_advanced")
FIG_DIR         = os.path.join(RESULTS_ML_ADV, "figs")
PRED_DIR        = os.path.join(RESULTS_ML_ADV, "preds")
CV_DIR          = os.path.join(RESULTS_ML_ADV, "cv")
IMP_DIR         = os.path.join(RESULTS_ML_ADV, "imp")
for d in [RESULTS_ML_ADV, FIG_DIR, PRED_DIR, CV_DIR, IMP_DIR]:
    os.makedirs(d, exist_ok=True)

# ---------- Utilitaires ----------
def _read_df(path):
    df = pd.read_csv(path, parse_dates=["date"])
    return df.set_index("date").sort_index()

def _metrics(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    return rmse, mae

def _load_Xy():
    TARGET_CANDIDATES = ["target_next_return", "target", "log_return", "r_t", "y"]
    X_tr = _read_df(os.path.join(DATA_PROC, "eur_chf_features_train.csv"))
    X_te = _read_df(os.path.join(DATA_PROC, "eur_chf_features_test.csv"))
    y_tr_df = _read_df(os.path.join(DATA_PROC, "eur_chf_train_y.csv"))
    y_te_df = _read_df(os.path.join(DATA_PROC, "eur_chf_test_y.csv"))

    # cible
    ycol = None
    for c in TARGET_CANDIDATES:
        if c in y_tr_df.columns and c in y_te_df.columns:
            ycol = c; break
    if ycol is None:
        raise ValueError(f"No target column found in y files. train={list(y_tr_df.columns)} test={list(y_te_df.columns)}")

    y_tr = y_tr_df[ycol].rename("target")
    y_te = y_te_df[ycol].rename("target")

    # pas de fuite
    for c in TARGET_CANDIDATES:
        if c in X_tr.columns: X_tr = X_tr.drop(columns=[c])
        if c in X_te.columns: X_te = X_te.drop(columns=[c])

    X_tr, y_tr = X_tr.align(y_tr, join="inner", axis=0)
    X_te, y_te = X_te.align(y_te, join="inner", axis=0)

    print(f"[7.5] X_train:{X_tr.shape}  X_test:{X_te.shape} | y_train:{y_tr.shape}  y_test:{y_te.shape}")
    return X_tr, y_tr, X_te, y_te

# Reuse si déjà en mémoire, sinon recharge
if all(name in globals() for name in ["X_train_df","y_train_s","X_test_df","y_test_s"]):
    X_tr_df, y_tr_s, X_te_df, y_te_s = X_train_df, y_train_s, X_test_df, y_test_s
else:
    X_tr_df, y_tr_s, X_te_df, y_te_s = _load_Xy()

feature_names = X_tr_df.columns.tolist()
tscv = TimeSeriesSplit(n_splits=5)

# =========================================================
# 7.5.1 — Résidus : histogramme + ACF (PATCH complet)
# =========================================================
import warnings

# Redéfinir la table des fichiers de prédictions si absente
try:
    pred_files
except NameError:
    pred_files = {
        "rf"    : "part7_rf_test_predictions.csv",
        "gbdt"  : "part7_gbdt_test_predictions.csv",
        "xgb"   : "part7_xgb_test_predictions.csv",
        "lgbm"  : "part7_lgbm_test_predictions.csv",
        "stack" : "part7_stack_test_predictions.csv",
        "blend" : "part7_blend_test_predictions.csv",
    }

# S'assurer qu'on a la vérité test (y_te_s) disponible
try:
    y_te_s
except NameError:
    # charge via le helper défini plus haut en 7.5
    _, _, _, y_te_s = _load_Xy()

# Boucle de génération des figures (histogramme + ACF) SANS collision de colonnes
for key, fname in pred_files.items():
    fpath = os.path.join(PRED_DIR, fname)
    if not os.path.exists(fpath):
        continue

    dfp = pd.read_csv(fpath, parse_dates=["date"]).set_index("date").sort_index()

    # Assure la présence de y_pred
    if "y_pred" not in dfp.columns:
        warnings.warn(f"[7.5] missing y_pred in {fname} – skipping residuals ACF for {key}")
        continue

    # PATCH anti-collision : on aligne y_true directement sur l'index des prédictions
    dfp["y_true"] = y_te_s.reindex(dfp.index)

    # Garde uniquement les dates communes valides
    dfp = dfp.dropna(subset=["y_pred", "y_true"])

    resid = (dfp["y_true"] - dfp["y_pred"]).astype(float)

    # Plot hist + ACF
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Histogramme
    axes[0].hist(resid.values, bins=50)
    axes[0].set_title(f"{key.upper()} — Residuals (Test)")
    axes[0].set_xlabel("Residual"); axes[0].set_ylabel("Count")

    # ACF des résidus
    if 'HAS_SM' in globals() and HAS_SM:
        plot_acf(resid.values, ax=axes[1], lags=40, title=f"{key.upper()} — Residual ACF (Test)")
    else:
        r = resid.values - resid.values.mean()
        acf = np.correlate(r, r, mode="full")[len(r)-1:]
        acf = acf / acf[0]
        lags = min(40, len(acf)-1)
        axes[1].stem(range(lags+1), acf[:lags+1], use_line_collection=True)
        axes[1].set_title(f"{key.upper()} — Residual ACF (Test)")
        axes[1].set_xlabel("Lag"); axes[1].set_ylabel("ACF")

    plt.tight_layout()
    out = os.path.join(FIG_DIR, f"part7_{key}_resid_acf.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"[7.5] Saved residual hist+ACF → {out}")


# =========================================================
# 7.5.2 — Erreurs par régimes de volatilité (terciles)
# - Vol = 'roll_std_21' si dispo dans X_test ; sinon reconstruit depuis returns
# - Sortie: CSV + figure (RMSE par tertile, par modèle)
# =========================================================
def _get_test_volatility():
    if "roll_std_21" in X_te_df.columns:
        vs = X_te_df["roll_std_21"].copy()
        vs.name = "roll_std_21"
        return vs.dropna()
    # fallback depuis eur_chf_returns.csv
    ret_path = os.path.join(DATA_PROC, "eur_chf_returns.csv")
    if not os.path.exists(ret_path):
        raise FileNotFoundError("eur_chf_returns.csv not found to rebuild roll_std_21")
    dfr = pd.read_csv(ret_path, parse_dates=["date"]).set_index("date").sort_index()
    # s'assurer qu'on a log_return
    if "log_return" not in dfr.columns:
        if "price" in dfr.columns:
            dfr["log_return"] = np.log(dfr["price"]).diff()
        else:
            raise KeyError("No 'log_return' or 'price' in eur_chf_returns.csv for volatility fallback")
    dfr["roll_std_21"] = dfr["log_return"].rolling(21, min_periods=5).std(ddof=0)
    return dfr["roll_std_21"].reindex(y_te_s.index).dropna()

vol_series = _get_test_volatility()
# garde les dates communes (sécurité)
common_idx = y_te_s.index.intersection(vol_series.index)

def _by_volatility(model_key, pred_path):
    if not os.path.exists(pred_path):
        return None
    dfp = pd.read_csv(pred_path, parse_dates=["date"]).set_index("date").sort_index()
    # align
    df = pd.DataFrame({
        "y_true": y_te_s.reindex(common_idx),
        "y_pred": dfp["y_pred"].reindex(common_idx),
        "vol":   vol_series.reindex(common_idx),
    }).dropna()
    if df.empty:
        return None
    # terciles
    q = df["vol"].quantile([1/3, 2/3]).values
    bins = [-np.inf, q[0], q[1], np.inf]
    labels = ["Low", "Mid", "High"]
    df["vol_tercile"] = pd.cut(df["vol"], bins=bins, labels=labels, include_lowest=True)
    rows = []
    for t in labels:
        dft = df[df["vol_tercile"] == t]
        if len(dft) == 0:
            continue
        rmse, mae = _metrics(dft["y_true"].values, dft["y_pred"].values)
        rows.append({"model": model_key.upper(), "tercile": t, "n": len(dft), "rmse": rmse, "mae": mae})
    return pd.DataFrame(rows)

frames = []
for key, fname in pred_files.items():
    fpath = os.path.join(PRED_DIR, fname)
    dfres = _by_volatility(key, fpath)
    if dfres is not None:
        frames.append(dfres)

if frames:
    byvol = pd.concat(frames, ignore_index=True)
    out_csv = os.path.join(RESULTS_ML_ADV, "part7_error_by_volatility.csv")
    byvol.to_csv(out_csv, index=False)
    print(f"[7.5] Saved volatility error table → {out_csv}")

    # Figure: RMSE par tertile (grouped bar, modèles en couleurs)
    # pour lisibilité, on trie les modèles par RMSE moyen
    order_models = (byvol.groupby("model")["rmse"].mean().sort_values().index.tolist())
    order_terc  = ["Low", "Mid", "High"]

    pivot = byvol.pivot(index="model", columns="tercile", values="rmse").reindex(order_models)[order_terc]
    ax = pivot.plot(kind="bar", figsize=(10,5))
    ax.set_title("EUR/CHF — Test RMSE by Volatility Tercile")
    ax.set_xlabel("Model"); ax.set_ylabel("RMSE")
    plt.xticks(rotation=0); plt.tight_layout()
    out_fig = os.path.join(FIG_DIR, "part7_error_by_volatility.png")
    plt.savefig(out_fig, dpi=150); plt.close()
    print(f"[7.5] Saved volatility error figure → {out_fig}")
else:
    print("[7.5] No predictions found for volatility breakdown – skipped.")

# =========================================================
# 7.5.3 — Stabilité des importances (Permutation Importance par fold)
# - On le fait pour des modèles *tree-based* (ex: RF, GBDT) disponibles en mémoire
# - Sorties: CSV par feature & fold + plot (moyenne ± écart-type)
# =========================================================
MODELS_FOR_PI = {}

# Tenter d'utiliser rf_best / gb_best s'ils existent (cf. 7.1/7.2)
if "rf_best" in globals():
    MODELS_FOR_PI["rf"] = rf_best
if "gb_best" in globals():
    MODELS_FOR_PI["gbdt"] = gb_best

# Si rien en mémoire, on tente de recharger depuis joblib (facultatif)
import joblib
def _load_if_exists(path):
    try:
        if os.path.exists(path):
            return joblib.load(path)
    except Exception:
        pass
    return None

if not MODELS_FOR_PI:
    # adapter les chemins si vous avez sauvegardé ailleurs
    cand = {
        "rf":   os.path.join(RESULTS_ML_ADV, "part7_rf_best.joblib"),
        "gbdt": os.path.join(RESULTS_ML_ADV, "part7_gbdt_best.joblib"),
    }
    for k, p in cand.items():
        est = _load_if_exists(p)
        if est is not None:
            MODELS_FOR_PI[k] = est

from sklearn.inspection import permutation_importance as _perm_imp

def _perm_by_fold(model_key, estimator, n_repeats=10, random_state=42):
    """Permutation importance par fold (sur la *validation* du fold)."""
    rows = []
    for f, (tr_idx, va_idx) in enumerate(tscv.split(X_tr_df)):
        X_tr, X_va = X_tr_df.iloc[tr_idx].values, X_tr_df.iloc[va_idx].values
        y_tr, y_va = y_tr_s.iloc[tr_idx].values, y_tr_s.iloc[va_idx].values

        est = deepcopy(estimator)
        # certains est. ont n_jobs; éviter double parallélisation
        if hasattr(est, "n_jobs"):
            try: est.set_params(n_jobs=1)
            except Exception: pass

        est.fit(X_tr, y_tr)
        pi = _perm_imp(est, X_va, y_va, n_repeats=n_repeats, random_state=random_state, scoring="neg_mean_squared_error")
        importances = pi.importances_mean  # plus négatif = pire (car scoring = neg MSE)
        # On convertit en "gain" positif: -neg_mse = +mse drop ; mais on peut rester sur la valeur brute pour stabilité
        for feat, val in zip(feature_names, importances):
            rows.append({"model": model_key.upper(), "fold": f, "feature": feat, "importance": float(val)})
    return pd.DataFrame(rows)

if MODELS_FOR_PI:
    for mkey, est in MODELS_FOR_PI.items():
        dfpi = _perm_by_fold(mkey, est, n_repeats=10, random_state=42)
        out_csv = os.path.join(IMP_DIR, f"part7_{mkey}_permimp_byfold.csv")
        dfpi.to_csv(out_csv, index=False)
        print(f"[7.5] Saved permutation importance by fold → {out_csv}")

        # Plot stabilité: moyenne ± écart-type par feature
        agg = dfpi.groupby(["feature"])["importance"].agg(["mean", "std"]).sort_values("mean", ascending=False)
        top_k = min(15, len(agg))
        plt.figure(figsize=(8, max(4, 0.35*top_k)))
        plt.barh(agg.head(top_k).index[::-1], agg["mean"].head(top_k).values[::-1], xerr=agg["std"].head(top_k).values[::-1])
        plt.title(f"{mkey.upper()} — Permutation importance (mean ± std across folds)")
        plt.tight_layout()
        out_fig = os.path.join(FIG_DIR, f"part7_{mkey}_permimp_stability.png")
        plt.savefig(out_fig, dpi=150); plt.close()
        print(f"[7.5] Saved PI stability plot → {out_fig}")
else:
    print("[7.5] No tree-based estimator available in memory or on disk for permutation importance.")

# =========================================================
# 7.5.4 — (Optionnel) PDP/ICE sur 2–3 features clés (ex: lag_1, roll_std_21)
# =========================================================
def _safe_pdp(estimator, X, features, model_key="model", kind="average"):
    if not HAS_PDP:
        print("[7.5] PDP not available in this sklearn version — skipped.")
        return
    try:
        fig = plt.figure(figsize=(9, 4))
        if len(features) == 1:
            PartialDependenceDisplay.from_estimator(estimator, X, features=features, kind=kind)
        else:
            PartialDependenceDisplay.from_estimator(estimator, X, features=features[:2], kind=kind)
        plt.suptitle(f"{model_key.upper()} — PDP ({kind})")
        plt.tight_layout()
        out = os.path.join(FIG_DIR, f"pdp_{model_key}_{kind}.png")
        plt.savefig(out, dpi=150); plt.close()
        print(f"[7.5] Saved PDP → {out}")
    except Exception as e:
        print(f"[7.5] PDP failed for {model_key}: {e}")

# PDP sur RF si dispo (interprétation naturelle)
if "rf" in MODELS_FOR_PI:
    rf_est = deepcopy(MODELS_FOR_PI["rf"])
    if hasattr(rf_est, "n_jobs"):
        try: rf_est.set_params(n_jobs=1)
        except Exception: pass
    rf_est.fit(X_tr_df.values, y_tr_s.values)

    # features cibles : privilégier 'lag_1' et 'roll_std_21' si présents
    pdp_feats = []
    for cand in ["lag_1", "roll_std_21", "lag_5", "roll_mean_21"]:
        if cand in feature_names and cand not in pdp_feats:
            pdp_feats.append(cand)
    if not pdp_feats:
        pdp_feats = [feature_names[0]]

    _safe_pdp(rf_est, X_tr_df.values, [pdp_feats[0]], model_key="rf", kind="average")
    if len(pdp_feats) >= 2:
        _safe_pdp(rf_est, X_tr_df.values, pdp_feats[:2], model_key="rf_pair", kind="average")

print("[7.5] Diagnostics completed.")

# =========================================================
# 7.6 — Walk-Forward (rolling-origin) backtesting (expanding window)
# =========================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error

# --- Dossiers (au cas où) ---
RESULTS_ML_ADV = os.path.join(PROJECT_ROOT, "results", "ml_advanced")
CV_DIR  = os.path.join(RESULTS_ML_ADV, "cv")
FIG_DIR = os.path.join(RESULTS_ML_ADV, "figs")
os.makedirs(CV_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# --- Helper métriques (robuste) ---
def _metrics(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    return rmse, mae

# --- Chargement des features & cible unifiées ---
try:
    # Utilise le helper déjà défini en 7.x
    X_train_df, y_train_s, X_test_df, y_test_s = _load_Xy()
except NameError:
    # Fallback minimal : recharge depuis les CSV standards
    X_train_df = pd.read_csv(os.path.join(DATA_PROC, "eur_chf_features_train.csv"), parse_dates=["date"]).set_index("date").sort_index()
    X_test_df  = pd.read_csv(os.path.join(DATA_PROC, "eur_chf_features_test.csv"),  parse_dates=["date"]).set_index("date").sort_index()
    # Cible (essaie différents noms)
    y_tr = pd.read_csv(os.path.join(DATA_PROC, "eur_chf_train_y.csv"))
    y_te = pd.read_csv(os.path.join(DATA_PROC, "eur_chf_test_y.csv"))
    ycol = "target" if "target" in y_tr.columns else ( "log_return_t+1" if "log_return_t+1" in y_tr.columns else y_tr.columns[-1] )
    y_train_s = pd.Series(y_tr[ycol].values, index=X_train_df.index, name="target")
    y_test_s  = pd.Series(y_te[ycol].values, index=X_test_df.index,  name="target")

# Intersection des colonnes (sécurité)
common_cols = sorted(list(set(X_train_df.columns) & set(X_test_df.columns)))
X_train_df = X_train_df[common_cols].copy()
X_test_df  = X_test_df[common_cols].copy()

# Concatène train+test pour un walk-forward "calendaire"
X_all = pd.concat([X_train_df, X_test_df], axis=0).sort_index()
y_all = pd.concat([y_train_s,  y_test_s],  axis=0).sort_index()

# --- Fabrique des modèles (clones frais à chaque bloc) ---
def get_rf_factory():
    # essaie variable en mémoire sinon joblib
    try:
        model = deepcopy(rf_best)
    except NameError:
        import joblib
        model = joblib.load(os.path.join(RESULTS_ML_ADV, "part7_rf_best.joblib"))
    def factory():
        return deepcopy(model)
    return factory

def get_gbdt_factory():
    try:
        model = deepcopy(gb_best)
    except NameError:
        import joblib
        model = joblib.load(os.path.join(RESULTS_ML_ADV, "part7_gbdt_best.joblib"))
    def factory():
        return deepcopy(model)
    return factory

def get_xgb_factory():
    try:
        model = deepcopy(xgb_best)
        def factory():
            return deepcopy(model)
        return factory
    except NameError:
        # essaie de charger, sinon None
        try:
            import joblib
            model = joblib.load(os.path.join(RESULTS_ML_ADV, "part7_xgb_best.joblib"))
            def factory():
                return deepcopy(model)
            return factory
        except Exception:
            return None

def get_lgbm_factory():
    try:
        model = deepcopy(lgb_best)
        def factory():
            return deepcopy(model)
        return factory
    except NameError:
        # essaie de charger, sinon None
        try:
            import joblib
            model = joblib.load(os.path.join(RESULTS_ML_ADV, "part7_lgbm_best.joblib"))
            def factory():
                return deepcopy(model)
            return factory
        except Exception:
            return None

def get_stack_factory():
    # STACK = StackingRegressor(rf, gbdt, final Ridge)
    # nécessite rf_best & gb_best (ou chargement)
    try:
        base_rf = deepcopy(rf_best)
    except NameError:
        import joblib
        base_rf = joblib.load(os.path.join(RESULTS_ML_ADV, "part7_rf_best.joblib"))
    try:
        base_gb = deepcopy(gb_best)
    except NameError:
        import joblib
        base_gb = joblib.load(os.path.join(RESULTS_ML_ADV, "part7_gbdt_best.joblib"))

    from sklearn.ensemble import StackingRegressor
    def factory():
        return StackingRegressor(
            estimators=[("rf", deepcopy(base_rf)), ("gbdt", deepcopy(base_gb))],
            final_estimator=Ridge(alpha=1.0, random_state=42),
            passthrough=False, n_jobs=-1
        )
    return factory

# --- Walk-forward calendaire (fenêtre "expanding") ---
def walk_forward_yearly(estimator_factory, X_df, y_s, start_test_year=None, min_train_years=3):
    """
    Train jusqu'à (année-1), teste sur l'année en cours. 
    start_test_year: si None, commence à (min_year + min_train_years)
    """
    idx = X_df.index
    years = np.array(idx.year)
    unique_years = np.unique(years)

    if start_test_year is None:
        start_test_year = int(unique_years.min()) + min_train_years

    rows = []
    for y in unique_years:
        if y < start_test_year:
            continue
        train_mask = (years < y)   # tout jusqu'à l'année précédente
        test_mask  = (years == y)  # année courante

        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue  # sécurité

        model = estimator_factory()  # clone neuf

        X_tr, y_tr = X_df[train_mask].values, y_s[train_mask].values
        X_te, y_te = X_df[test_mask].values,  y_s[test_mask].values

        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        rmse, mae = _metrics(y_te, y_pred)

        rows.append({
            "year": int(y),
            "train_start": X_df.index[train_mask][0].date().isoformat(),
            "train_end":   X_df.index[train_mask][-1].date().isoformat(),
            "test_start":  X_df.index[test_mask][0].date().isoformat(),
            "test_end":    X_df.index[test_mask][-1].date().isoformat(),
            "n_test": int(test_mask.sum()),
            "rmse": rmse,
            "mae": mae
        })
    return pd.DataFrame(rows)

# --- Liste des modèles à backtester ---
model_factories = {
    "rf":    get_rf_factory(),
    "gbdt":  get_gbdt_factory(),
    "xgb":   get_xgb_factory(),   # peut être None si pas dispo
    "lgbm":  get_lgbm_factory(),  # peut être None si pas dispo
    "stack": get_stack_factory(),
}

# Nettoyage (retire None)
model_factories = {k: v for k, v in model_factories.items() if v is not None}

# --- Exécution & sauvegarde des artefacts ---
for key, factory in model_factories.items():
    print(f"[7.6][WF] running yearly expanding backtest for: {key.upper()}")

    df_metrics = walk_forward_yearly(factory, X_all, y_all, start_test_year=2020, min_train_years=4)
    # Sauvegarde CSV
    csv_path = os.path.join(CV_DIR, f"part7_{key}_rolling_metrics.csv")
    df_metrics.to_csv(csv_path, index=False)
    print(f"[7.6][WF] saved → {csv_path}")

    # Figure RMSE dans le temps
    plt.figure(figsize=(10, 4))
    plt.plot(df_metrics["year"], df_metrics["rmse"], marker="o")
    plt.title(f"Part 7 — {key.upper()} rolling-origin RMSE (yearly)")
    plt.xlabel("Year"); plt.ylabel("RMSE"); plt.grid(True, alpha=0.3)
    fig_path = os.path.join(FIG_DIR, f"part7_{key}_rolling_rmse.png")
    plt.tight_layout(); plt.savefig(fig_path, dpi=150); plt.close()
    print(f"[7.6][WF] figure → {fig_path}")

# =========================================================
# Part 7.7 — Final comparison (ML advanced vs references)
# =========================================================
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---------- Paths (adapt if needed) ----------
PROJECT_ROOT = "/Users/eliemenassa/Desktop/Projet Forecasting"
DATA_PROC     = os.path.join(PROJECT_ROOT, "data", "processed")
RES_BASE      = os.path.join(PROJECT_ROOT, "results", "baseline")
RES_STATS     = os.path.join(PROJECT_ROOT, "results", "stats")
RES_ML_BASE   = os.path.join(PROJECT_ROOT, "results", "ml_baseline")
RES_ML_ADV    = os.path.join(PROJECT_ROOT, "results", "ml_advanced")
PRED_DIR      = os.path.join(RES_ML_ADV, "preds")
FIG_DIR       = os.path.join(RES_ML_ADV, "figs")
os.makedirs(RES_ML_ADV, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

SUMMARY_CSV   = os.path.join(RES_ML_ADV, "part7_summary_metrics.csv")
LEADERBOARD   = os.path.join(PROJECT_ROOT, "models", "leaderboard.csv")
os.makedirs(os.path.dirname(LEADERBOARD), exist_ok=True)

# ---------- Helpers ----------
def rmse(y, yhat):
    return float(np.sqrt(np.mean((y - yhat)**2)))

def mae(y, yhat):
    return float(np.mean(np.abs(y - yhat)))

def mape(y, yhat, eps=1e-12):
    # returns~0 → MAPE is not meaningful; keep for consistency
    y = np.asarray(y); yhat = np.asarray(yhat)
    return float(np.mean(np.abs((yhat - y) / (np.abs(y) + eps))))

def compute_metrics(y_true_s, y_pred_s):
    """y_* are aligned pandas Series on the same index."""
    return {
        "rmse": rmse(y_true_s.values, y_pred_s.values),
        "mae":  mae(y_true_s.values, y_pred_s.values),
        "mape_pct": 100.0 * mape(y_true_s.values, y_pred_s.values),
    }

def _load_test_target():
    """Robust loader for y_test (2023–2025), tolerant to various index/column formats."""
    ypaths = [
        os.path.join(DATA_PROC, "eur_chf_test_y.csv"),
        os.path.join(DATA_PROC, "eur_chf_test.csv"),
    ]
    y = None
    for p in ypaths:
        if not os.path.exists(p):
            continue
        try:
            df = pd.read_csv(p, parse_dates=["date"]).set_index("date").sort_index()
        except Exception:
            df = pd.read_csv(p)
            # detect date-like column
            date_col = None
            for c in df.columns:
                lc = c.lower()
                if lc in {"date", "datetime", "timestamp"} or "unnamed" in lc or "index" in lc:
                    try:
                        pd.to_datetime(df[c])
                        date_col = c
                        break
                    except Exception:
                        pass
            if date_col is not None:
                df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
                df = df.set_index(date_col).sort_index()
            else:
                # final fallback: try to parse index
                try:
                    df.index = pd.to_datetime(df.index, errors="coerce")
                except Exception:
                    pass

        # candidate target names
        for c in ["target_next_return", "target", "y", "y_true", "log_return_next", "log_return"]:
            if c in df.columns:
                y = df[c].rename("y_true")
                break
        if y is not None:
            break

    if y is None:
        raise FileNotFoundError("Unable to load y_test: compatible file/column not found in data/processed.")
    # keep the common test horizon used in Parts 4–7
    return y.loc["2023-01-01":]

def _safe_read_pred(path):
    """Robust loader for prediction CSVs with varied date/index formats and column names."""
    if not os.path.exists(path):
        return None
    # try modern format
    try:
        df = pd.read_csv(path, parse_dates=["date"])
        df = df.set_index("date").sort_index()
    except Exception:
        # fallback inference
        df = pd.read_csv(path)
        date_col = None
        for c in df.columns:
            lc = c.lower()
            if lc in {"date", "datetime", "timestamp"} or "unnamed" in lc or "index" in lc:
                try:
                    pd.to_datetime(df[c])
                    date_col = c
                    break
                except Exception:
                    pass
        if date_col is not None:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.set_index(date_col).sort_index()
        else:
            try:
                df.index = pd.to_datetime(df.index, errors="coerce")
            except Exception:
                pass

    # prediction column candidates
    candidates = ["y_pred", "yhat", "pred", "prediction", "forecast"]
    pred_col = next((c for c in candidates if c in df.columns), None)
    if pred_col is None:
        # default to last numeric column if nothing explicit
        nums = df.select_dtypes(include=[np.number]).columns.tolist()
        if not nums:
            return None
        pred_col = nums[-1]
    s = df[pred_col].rename("y_pred")
    # align time window if datetime index
    if isinstance(s.index, pd.DatetimeIndex):
        s = s.loc["2023-01-01":]
    return s

def align_on_index(y_true_s, y_pred_s):
    """Inner-join alignment on index if datetime-like; else align by tail length."""
    if isinstance(y_true_s.index, pd.DatetimeIndex) and isinstance(y_pred_s.index, pd.DatetimeIndex):
        df = pd.concat([y_true_s, y_pred_s], axis=1, join="inner").dropna()
        return df.iloc[:,0], df.iloc[:,1]
    # non-datetime: align by length (use last N points)
    n = min(len(y_true_s), len(y_pred_s))
    return y_true_s.iloc[-n:], y_pred_s.iloc[-n:]

def upsert_leaderboard(rows_for_lb):
    """Insert or update Part-7 model rows in models/leaderboard.csv."""
    cols = ["model","rmse","mae","mape_pct"]
    new_df = pd.DataFrame(rows_for_lb, columns=cols)
    if os.path.exists(LEADERBOARD):
        lb = pd.read_csv(LEADERBOARD)
        # upsert
        for _, r in new_df.iterrows():
            mask = lb["model"] == r["model"]
            if mask.any():
                lb.loc[mask, ["rmse","mae","mape_pct"]] = [r["rmse"], r["mae"], r["mape_pct"]]
            else:
                lb = pd.concat([lb, pd.DataFrame([r])], ignore_index=True)
    else:
        lb = new_df
    lb.to_csv(LEADERBOARD, index=False)
    print(f"[leaderboard] upserted → {LEADERBOARD}")

# ---------- Load target (test) ----------
y_test = _load_test_target()

# ---------- Prediction files to compare ----------
candidates = {
    # Part 4 (baseline)
    "Baseline_SES (Part 4)": os.path.join(RES_BASE,    "ses_predictions.csv"),
    # Part 5 (stats)
    "ARIMA (Part 5)"       : os.path.join(RES_STATS,   "arima_predictions.csv"),
    "HW_SES (Part 5)"      : os.path.join(RES_STATS,   "hw_ses_predictions.csv"),
    # Part 6 (linear ML)
    "Linear (Part 6)"      : os.path.join(RES_ML_BASE, "part6_linear_predictions.csv"),
    # Part 7 (advanced ML)
    "RF (Part 7)"          : os.path.join(PRED_DIR,    "part7_rf_test_predictions.csv"),
    "GBDT (Part 7)"        : os.path.join(PRED_DIR,    "part7_gbdt_test_predictions.csv"),
    "XGB (Part 7)"         : os.path.join(PRED_DIR,    "part7_xgb_test_predictions.csv"),
    "LGBM (Part 7)"        : os.path.join(PRED_DIR,    "part7_lgbm_test_predictions.csv"),
    "STACK (Part 7)"       : os.path.join(PRED_DIR,    "part7_stack_test_predictions.csv"),
    "BLEND (Part 7)"       : os.path.join(PRED_DIR,    "part7_blend_test_predictions.csv"),
}

rows = []
residuals_for_plot = {}
pred_series_for_plot = {}

for label, fpath in candidates.items():
    s_pred = _safe_read_pred(fpath)
    if s_pred is None or len(s_pred) == 0:
        print(f"[7.7] WARN: missing/empty predictions → {label} ({fpath})")
        continue

    y_al, yhat_al = align_on_index(y_test.rename("y_true"), s_pred)
    if len(y_al) == 0:
        print(f"[7.7] WARN: no overlapping dates → {label}")
        continue

    m = compute_metrics(y_al, yhat_al)
    rows.append({"label": label, **m})
    # store plots data
    residuals_for_plot[label] = (y_al - yhat_al).rename(label)
    pred_series_for_plot[label] = yhat_al.rename(label)

# ---------- Save summary CSV ----------
if not rows:
    raise RuntimeError("No model predictions were found; nothing to summarize.")
df_sum = pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
df_sum.to_csv(SUMMARY_CSV, index=False)
print(f"[7.7] Summary saved → {SUMMARY_CSV}")
print(df_sum)

# ---------- RMSE bar chart (all models) ----------
plt.figure(figsize=(10,5))
plt.barh(df_sum["label"], df_sum["rmse"])
plt.gca().invert_yaxis()
plt.xlabel("RMSE"); plt.title("Part 7 — Test RMSE comparison (all models)")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "part7_rmse_comparison.png"), dpi=150)
plt.close()

# ---------- Overlay top-3 forecasts vs y_test ----------
top3 = df_sum.head(3)["label"].tolist()
common_index = y_test.index
plt.figure(figsize=(14,4))
plt.plot(y_test.index, y_test.values, label="y_true", linewidth=1.2)
for lbl in top3:
    yhat = pred_series_for_plot[lbl]
    # align again to y_test index for a clean overlay
    yt, yh = align_on_index(y_test, yhat)
    plt.plot(yt.index, yh.values, label=lbl, linewidth=1.0)
plt.title("Part 7 — Top-3 models: y_true vs predictions (test)")
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "part7_top3_forecasts_vs_test.png"), dpi=150)
plt.close()

# ---------- Error distributions (top-3) ----------
fig, axes = plt.subplots(1, 3, figsize=(14,4), sharex=False, sharey=True)
for ax, lbl in zip(axes, top3):
    resid = residuals_for_plot[lbl]
    ax.hist(resid.dropna().values, bins=50)
    ax.set_title(lbl); ax.set_xlabel("Residual"); ax.set_ylabel("Count")
fig.suptitle("Part 7 — Residual distributions (Top-3 models)", y=1.03)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "part7_error_distributions.png"), dpi=150, bbox_inches="tight")
plt.close()

# ---------- Leaderboard upsert (only Part-7 models) ----------
lb_rows = []
code_map = {
    "RF (Part 7)"   : "ML_RF",
    "GBDT (Part 7)" : "ML_GBDT",
    "XGB (Part 7)"  : "ML_XGB",
    "LGBM (Part 7)" : "ML_LGBM",
    "STACK (Part 7)": "ML_STACK",
    "BLEND (Part 7)": "ML_BLEND",
}
for _, r in df_sum.iterrows():
    if r["label"] in code_map:
        lb_rows.append({
            "model": code_map[r["label"]],
            "rmse": r["rmse"],
            "mae": r["mae"],
            "mape_pct": r["mape_pct"],
        })
if lb_rows:
    upsert_leaderboard(lb_rows)
else:
    print("[leaderboard] nothing to upsert for Part-7 models.")

print("[7.7] Figures saved:")
print(" -", os.path.join(FIG_DIR, "part7_rmse_comparison.png"))
print(" -", os.path.join(FIG_DIR, "part7_top3_forecasts_vs_test.png"))
print(" -", os.path.join(FIG_DIR, "part7_error_distributions.png"))
