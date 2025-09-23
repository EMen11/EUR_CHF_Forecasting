# =========================================================
# Part 6.1 - Features (lag1, lag5, roll_mean_5, roll_std_20)
# Version unique et robuste (root forcé + debug + fallback noms)
# =========================================================
import os
from pathlib import Path
import pandas as pd

# ---------- 6.0 SETUP (ROOT FORCÉ, AUCUNE REDÉFINITION APRÈS !) ----------
PROJECT_ROOT = Path("/Users/eliemenassa/Desktop/Projet Forecasting").resolve()
DATA_PROC    = PROJECT_ROOT / "data" / "processed"
RESULTS_ML   = PROJECT_ROOT / "results" / "ml_baseline"
DATA_PROC.mkdir(parents=True, exist_ok=True)
RESULTS_ML.mkdir(parents=True, exist_ok=True)

def banner(txt):
    print("\n" + "="*56)
    print(txt)
    print("="*56)

banner("6.0 - Setup forcé")
print(f"PROJECT_ROOT: {PROJECT_ROOT}")
print(f"DATA_PROC   : {DATA_PROC}")
print(f"RESULTS_ML  : {RESULTS_ML}")

# ---------- DEBUG: lister le dossier pour vérifier les fichiers ----------
banner("Listing de data/processed")
if DATA_PROC.exists():
    for p in sorted(DATA_PROC.iterdir()):
        print("-", p.name)
else:
    raise FileNotFoundError(f"Dossier absent: {DATA_PROC}")

# ---------- Helper: trouver les chemins train/test de façon flexible ----------
def find_file(candidates):
    for name in candidates:
        p = DATA_PROC / name
        if p.exists():
            return p
    # Essayer aussi les variantes parquet si csv introuvable
    for name in candidates:
        base = name.rsplit(".", 1)[0]
        p = DATA_PROC / f"{base}.parquet"
        if p.exists():
            return p
    return None

train_candidates = [
    "eur_chf_train.csv", "EUR_CHF_train.csv", "eur_chf_train_scaled.csv"
]
test_candidates  = [
    "eur_chf_test.csv",  "EUR_CHF_test.csv",  "eur_chf_test_scaled.csv"
]

train_path = find_file(train_candidates)
test_path  = find_file(test_candidates)

if train_path is None or test_path is None:
    raise FileNotFoundError(
        "Impossible de trouver les fichiers train/test dans data/processed.\n"
        f"Train tried: {train_candidates}\n"
        f"Test tried : {test_candidates}\n"
        "Regarde le listing ci-dessus et renomme si besoin."
    )

banner("Fichiers trouvés")
print(f"Train: {train_path.name}")
print(f"Test : {test_path.name}")

# ---------- Lecture & harmonisation ----------
def read_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    # normaliser date/index
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").set_index("date")
    return df

train = read_any(train_path)
test  = read_any(test_path)

# s'assurer d'avoir log_return
def ensure_log_return(df: pd.DataFrame) -> pd.DataFrame:
    if "log_return" not in df.columns:
        cands = [c for c in df.columns if c.lower() == "log_return"]
        if cands:
            df = df.rename(columns={cands[0]: "log_return"})
        else:
            raise ValueError(
                f"'log_return' introuvable dans {list(df.columns)[:10]}..."
            )
    return df

train = ensure_log_return(train)
test  = ensure_log_return(test)

# ---------- Concat pour calculer les features sans fuite ----------
full = pd.concat([train[["log_return"]], test[["log_return"]]], axis=0).sort_index()

full["lag1"]        = full["log_return"].shift(1)
full["lag5"]        = full["log_return"].shift(5)
full["roll_mean_5"] = full["log_return"].rolling(window=5,  min_periods=5).mean()
full["roll_std_20"] = full["log_return"].rolling(window=20, min_periods=20).std(ddof=0)

# cible = retour t+1
full["target_next_return"] = full["log_return"].shift(-1)

# ---------- Re-split dates (identiques à tes chapitres) ----------
train_start = "2015-01-02"
train_end   = "2022-12-30"
test_start  = "2023-01-02"
test_end    = "2025-09-16"

cols = ["lag1","lag5","roll_mean_5","roll_std_20","target_next_return"]
full_model = full[cols].dropna()

train_feat = full_model.loc[train_start:train_end].copy()
test_feat  = full_model.loc[test_start:test_end].copy()

banner("Tailles après features")
print(f"Train: {train_feat.shape} | Test: {test_feat.shape}")

assert len(train_feat) > 0, "Train vide: vérifie la fenêtre et les dates."
assert len(test_feat)  > 0, "Test vide: vérifie la fenêtre et les dates."

# ---------- Sauvegardes ----------
feat_train_path = DATA_PROC / "eur_chf_features_train.csv"
feat_test_path  = DATA_PROC / "eur_chf_features_test.csv"
train_feat.to_csv(feat_train_path)
test_feat.to_csv(feat_test_path)

banner("Fichiers sauvegardés")
print(f"- {feat_train_path}")
print(f"- {feat_test_path}")

# Notes:
# - Un seul bloc de setup: rien ne redéfinit PROJECT_ROOT/ DATA_PROC après.
# - Listing initial = diagnostic immédiat si le nom diffère.
# - Fallback parquet et variantes de casse si CSV manquant.

# =========================================================
# 6.2 - LinearRegression + TimeSeriesSplit (baseline ML)
# =========================================================
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ---------- 6.0 Setup ----------
PROJECT_ROOT = Path("/Users/eliemenassa/Desktop/Projet Forecasting").resolve()
DATA_PROC    = PROJECT_ROOT / "data" / "processed"
RESULTS_ML   = PROJECT_ROOT / "results" / "ml_baseline"
RESULTS_ML.mkdir(parents=True, exist_ok=True)

def banner(txt):
    print("\n" + "="*56)
    print(txt)
    print("="*56)

banner("6.2 - LinearRegression + TimeSeriesSplit")

# ---------- Lecture des features (créées en 6.1) ----------
feat_train_path = DATA_PROC / "eur_chf_features_train.csv"
feat_test_path  = DATA_PROC / "eur_chf_features_test.csv"
if not feat_train_path.exists() or not feat_test_path.exists():
    raise FileNotFoundError("Exécuter 6.1 d'abord pour générer les features train/test.")

train = pd.read_csv(feat_train_path, parse_dates=["date"]).set_index("date")
test  = pd.read_csv(feat_test_path,  parse_dates=["date"]).set_index("date")

FEATURES = ["lag1", "lag5", "roll_mean_5", "roll_std_20"]
TARGET   = "target_next_return"

X_train, y_train = train[FEATURES].values, train[TARGET].values
X_test,  y_test  = test[FEATURES].values,  test[TARGET].values

# ---------- Pipeline : StandardScaler + LinearRegression ----------
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("linreg", LinearRegression())
])

# ---------- Validation croisée (TimeSeriesSplit) ----------
tscv = TimeSeriesSplit(n_splits=5)
cv_rmse = -cross_val_score(pipe, X_train, y_train, cv=tscv, scoring="neg_root_mean_squared_error")
cv_mae  = -cross_val_score(pipe, X_train, y_train, cv=tscv, scoring="neg_mean_absolute_error")

banner("CV (TimeSeriesSplit) sur TRAIN")
print(f"RMSE (mean ± std): {cv_rmse.mean():.6f} ± {cv_rmse.std():.6f}")
print(f"MAE  (mean ± std): {cv_mae.mean():.6f} ± {cv_mae.std():.6f}")

# ---------- Fit final sur tout le TRAIN, évaluation sur TEST ----------
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

# ✅ Correction : RMSE calculé avec sqrt() pour compatibilité
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)

banner("Métriques sur TEST")
print(f"RMSE: {rmse:.6f}")
print(f"MAE : {mae :.6f}")
print(f"R²  : {r2  :.6f}")

# ---------- Sauvegardes : métriques, leaderboard, prédictions ----------
metrics = {
    "model": "Linear_Regression",
    "family": "ML_Baseline",
    "split": "test",
    "rmse": float(rmse),
    "mae": float(mae),
    "r2": float(r2),
    "cv_rmse_mean": float(cv_rmse.mean()),
    "cv_rmse_std":  float(cv_rmse.std()),
    "cv_mae_mean":  float(cv_mae.mean()),
    "cv_mae_std":   float(cv_mae.std()),
}
metrics_df = pd.DataFrame([metrics])
metrics_path = RESULTS_ML / "part6_linear_metrics.csv"
metrics_df.to_csv(metrics_path, index=False)

leaderboard_path = DATA_PROC / "leaderboard.csv"
lb_row = pd.DataFrame([{
    "model": "Linear_Regression",
    "test_rmse": rmse,
    "test_mae": mae,
    "test_r2": r2
}])
if leaderboard_path.exists():
    lb = pd.read_csv(leaderboard_path)
    lb = lb[lb["model"] != "Linear_Regression"]
    lb = pd.concat([lb, lb_row], ignore_index=True)
else:
    lb = lb_row
lb.to_csv(leaderboard_path, index=False)

preds = test.copy()
preds["y_true"] = y_test
preds["y_pred"] = y_pred
preds["residual"] = preds["y_true"] - preds["y_pred"]
preds_path = RESULTS_ML / "part6_linear_predictions.csv"
preds.to_csv(preds_path)

# ---------- Figures ----------
# 1) y_true vs y_pred
plt.figure(figsize=(10,4))
preds[["y_true", "y_pred"]].plot(ax=plt.gca())
plt.title("Part 6 – Linear Regression: y_true vs y_pred (test)")
plt.xlabel("Date"); plt.ylabel("Return")
plt.tight_layout()
fig1_path = RESULTS_ML / "part6_linear_test_vs_pred.png"
plt.savefig(fig1_path, dpi=150)
plt.close()

# 2) histogramme des résidus
plt.figure(figsize=(6,4))
plt.hist(preds["residual"].dropna(), bins=40)
plt.title("Part 6 – Linear Regression: Residuals (test)")
plt.xlabel("Residual"); plt.ylabel("Count")
plt.tight_layout()
fig2_path = RESULTS_ML / "part6_linear_residuals_hist.png"
plt.savefig(fig2_path, dpi=150)
plt.close()

# 3) coefficients (standardisés)
linreg = pipe.named_steps["linreg"]
coef   = linreg.coef_
coef_df = pd.DataFrame({"feature": FEATURES, "coef_standardized": coef})
coef_path = RESULTS_ML / "part6_linear_coefficients.csv"
coef_df.to_csv(coef_path, index=False)

banner("Artifacts sauvegardés")
print(f"- metrics:       {metrics_path}")
print(f"- leaderboard:   {leaderboard_path}")
print(f"- predictions:   {preds_path}")
print(f"- fig (series):  {fig1_path}")
print(f"- fig (resid):   {fig2_path}")
print(f"- coefficients:  {coef_path}")

# =========================================================
# 6.3 - Validation: TimeSeriesSplit vs "CV classique"
# Objectif: montrer l'impact du respect de la temporalité
# =========================================================
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# ---------- Setup (mêmes chemins que 6.1 / 6.2) ----------
PROJECT_ROOT = Path("/Users/eliemenassa/Desktop/Projet Forecasting").resolve()
DATA_PROC    = PROJECT_ROOT / "data" / "processed"
RESULTS_ML   = PROJECT_ROOT / "results" / "ml_baseline"
RESULTS_ML.mkdir(parents=True, exist_ok=True)

def banner(txt):
    print("\n" + "="*56)
    print(txt)
    print("="*56)

banner("6.3 - Validation: TimeSeriesSplit vs CV classique")

# ---------- Charger les features (train uniquement) ----------
feat_train_path = DATA_PROC / "eur_chf_features_train.csv"
if not feat_train_path.exists():
    raise FileNotFoundError("Features train introuvables. Exécute d'abord la partie 6.1.")

train = pd.read_csv(feat_train_path, parse_dates=["date"]).set_index("date")
FEATURES = ["lag1", "lag5", "roll_mean_5", "roll_std_20"]
TARGET   = "target_next_return"
X_train, y_train = train[FEATURES].values, train[TARGET].values

# ---------- Pipeline baseline ----------
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("linreg", LinearRegression())
])

# ---------- Définition des schémas de CV ----------
K = 5
cv_classic = KFold(n_splits=K, shuffle=True, random_state=42)  # ❌ mélange passé/futur
cv_tscv    = TimeSeriesSplit(n_splits=K)                        # ✅ respecte la chronologie

# ---------- Scores (on évalue RMSE / MAE via scoring négatif) ----------
# Nota: cross_val_score retourne des scores NEGATIFS pour les pertes -> on les inverse
rmse_classic = -cross_val_score(pipe, X_train, y_train, cv=cv_classic, scoring="neg_root_mean_squared_error")
mae_classic  = -cross_val_score(pipe, X_train, y_train, cv=cv_classic,  scoring="neg_mean_absolute_error")

rmse_tscv = -cross_val_score(pipe, X_train, y_train, cv=cv_tscv, scoring="neg_root_mean_squared_error")
mae_tscv  = -cross_val_score(pipe, X_train, y_train, cv=cv_tscv,  scoring="neg_mean_absolute_error")

banner("Résultats CV (moyenne ± écart-type)")
print(f"[CV classique]  RMSE: {rmse_classic.mean():.6f} ± {rmse_classic.std():.6f} | "
      f"MAE: {mae_classic.mean():.6f} ± {mae_classic.std():.6f}")
print(f"[TimeSeriesSplit] RMSE: {rmse_tscv.mean():.6f} ± {rmse_tscv.std():.6f} | "
      f"MAE: {mae_tscv.mean():.6f} ± {mae_tscv.std():.6f}")

# ---------- Tableau comparatif + sauvegarde ----------
cv_summary = pd.DataFrame({
    "scheme": ["CV_classic_shuffle", "TimeSeriesSplit"],
    "rmse_mean": [rmse_classic.mean(), rmse_tscv.mean()],
    "rmse_std":  [rmse_classic.std(),  rmse_tscv.std()],
    "mae_mean":  [mae_classic.mean(),  mae_tscv.mean()],
    "mae_std":   [mae_classic.std(),   mae_tscv.std()],
    "k_folds":   [K, K]
})
cv_path = RESULTS_ML / "part6_cv_comparison.csv"
cv_summary.to_csv(cv_path, index=False)

# ---------- Visualisation simple (bar chart) ----------
labels = ["CV classique\n(shuffle)", "TimeSeriesSplit"]
rmse_vals = [rmse_classic.mean(), rmse_tscv.mean()]
mae_vals  = [mae_classic.mean(),  mae_tscv.mean()]

plt.figure(figsize=(8,4))
x = np.arange(len(labels))
width = 0.35
plt.bar(x - width/2, rmse_vals, width, label="RMSE")
plt.bar(x + width/2, mae_vals,  width, label="MAE")
plt.xticks(x, labels)
plt.title("Part 6.3 – CV classique vs TimeSeriesSplit (moyennes sur train)")
plt.ylabel("Error")
plt.legend()
plt.tight_layout()
fig_path = RESULTS_ML / "part6_cv_comparison.png"
plt.savefig(fig_path, dpi=150)
plt.close()

banner("Artifacts sauvegardés")
print(f"- Résumé CSV : {cv_path}")
print(f"- Figure     : {fig_path}")

# Notes :
# - CV classique (shuffle=True) mélange passé et futur -> optimiste/biaisé pour des séries temporelles.
# - TimeSeriesSplit conserve un ordre chronologique (train = passé, val = futur) -> estimation plus réaliste.
# - On utilise les mêmes features et le même pipeline pour comparer à iso-périmètre.


# =========================================================
# 6.4 - Résultats : comparaison Linear Regression vs SES (fix rapide)
# =========================================================
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path("/Users/eliemenassa/Desktop/Projet Forecasting").resolve()
RESULTS_ML   = PROJECT_ROOT / "results" / "ml_baseline"
RESULTS_ML.mkdir(parents=True, exist_ok=True)

# --- 1) Charger les métriques Linear Regression (Part 6.2) ---
lin_metrics_path = RESULTS_ML / "part6_linear_metrics.csv"
lin = pd.read_csv(lin_metrics_path).iloc[0]
lin_rmse = float(lin["rmse"])
lin_mae  = float(lin["mae"])

# --- 2) Forcer les métriques SES (issues de la Partie 4) ---
# Valeurs confirmées dans Part 4.3/4.4/4.5:
ses_rmse = 0.003326
ses_mae  = 0.002511

print("=== 6.4 - Comparaison SES vs Linear (TEST) ===")
print(f"SES     -> RMSE={ses_rmse:.6f} | MAE={ses_mae:.6f}")
print(f"Linear  -> RMSE={lin_rmse:.6f} | MAE={lin_mae:.6f}")

# --- 3) Tableau comparatif + deltas (%) ---
comp = pd.DataFrame({
    "model": ["SES", "Linear_Regression"],
    "rmse":  [ses_rmse, lin_rmse],
    "mae":   [ses_mae,  lin_mae]
})
comp_path = RESULTS_ML / "part6_vs_ses_comparison.csv"
comp.to_csv(comp_path, index=False)

delta_rmse_pct = (lin_rmse - ses_rmse) / ses_rmse * 100.0
delta_mae_pct  = (lin_mae  - ses_mae ) / ses_mae  * 100.0
print(f"Δ RMSE vs SES: {delta_rmse_pct:+.2f}%")
print(f"Δ MAE  vs SES: {delta_mae_pct:+.2f}%")

# --- 4) Figure comparatif ---
labels = ["SES", "Linear"]
rmse_vals = [ses_rmse, lin_rmse]
mae_vals  = [ses_mae,  lin_mae]

plt.figure(figsize=(8,4))
x = np.arange(len(labels)); w = 0.35
plt.bar(x - w/2, rmse_vals, w, label="RMSE")
plt.bar(x + w/2, mae_vals,  w, label="MAE")
plt.xticks(x, labels)
plt.ylabel("Error")
plt.title("Part 6.4 – SES vs Linear Regression (Test)")
plt.legend()
plt.tight_layout()
fig_path = RESULTS_ML / "part6_vs_ses_comparison.png"
plt.savefig(fig_path, dpi=150)
plt.close()

# --- 5) Bilan console ---
best_rmse = "Linear" if lin_rmse < ses_rmse else "SES"
best_mae  = "Linear" if lin_mae  < ses_mae  else "SES"
print(f"Meilleur RMSE : {best_rmse} ({min(lin_rmse, ses_rmse):.6f})")
print(f"Meilleur MAE  : {best_mae}  ({min(lin_mae,  ses_mae ):.6f})")
print(f"CSV   -> {comp_path}")
print(f"FIG   -> {fig_path}")

