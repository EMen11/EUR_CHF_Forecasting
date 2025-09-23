# %% =========================================================
# 3.0 - EVALUATION UTILITIES (no-leak, backtesting, metrics, plots)
# =============================================================
# This section centralizes shared tooling for all models (baselines, ARIMA, ML, etc.)
# - file paths & safe data loading
# - metrics: RMSE/MAE (+ optional directional accuracy)
# - time-series cross-validation helpers (no shuffling)
# - walk-forward evaluation on the Test period (with periodic refits)
# - residual diagnostics (Ljung-Box), plotting helpers
# Everything is designed to avoid data leakage.

# =========================
# Imports (clean & grouped)
# =========================

# Standard library
import os
import json
from pathlib import Path

# Core scientific stack
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# scikit-learn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error

# statsmodels
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf, pacf, adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
try:
    from statsmodels.stats.diagnostic import acorr_ljungbox
except Exception:
    acorr_ljungbox = None  # guarded use if not available

# Persistence
import joblib


# --- Banner helper (fallback if not already available) ---
def _default_banner(text: str):
    bar = "=" * len(text)
    print(f"\n{bar}\n{text}\n{bar}")

try:
    # If you defined banner() earlier (e.g., Part2), we reuse it:
    from utils.helpers import banner  # noqa: F401
except Exception:
    banner = _default_banner  # fallback to a local simple banner
    
from sklearn.ensemble import RandomForestRegressor

# XGBoost (optionnel)
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

# SHAP (optionnel)
try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False    


# --- Project paths ---
PROJECT_ROOT = os.getcwd()
DATA_PROC = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
PRED_DIR = os.path.join(DATA_PROC, "preds")
FIG_DIR = os.path.join(REPORTS_DIR, "figures")

# Create folders if missing
for d in [MODELS_DIR, REPORTS_DIR, PRED_DIR, FIG_DIR]:
    os.makedirs(d, exist_ok=True)

banner("3.0 - EVALUATION UTILITIES")


# =========================================================
# 3.0.A - Safe loading helpers (train/test features and target) — ROBUST
# =========================================================
def _read_with_flexible_date(path: str):
    """
    Read a CSV and reliably get a DatetimeIndex.
    Tries, in order:
      1) parse_dates=['Date']
      2) parse_dates=['date']
      3) index_col=0 and parse that as dates
    Returns: DataFrame indexed by DatetimeIndex, with all original columns.
    """
    # Try 'Date'
    try:
        df = pd.read_csv(path, parse_dates=['Date'])
        if 'Date' in df.columns:
            df = df.set_index('Date')
        return df
    except Exception:
        pass

    # Try 'date'
    try:
        df = pd.read_csv(path, parse_dates=['date'])
        if 'date' in df.columns:
            df = df.set_index('date')
        return df
    except Exception:
        pass

    # Fallback: assume first column is date/index
    # (common when saved with reset_index or default index)
    df = pd.read_csv(path)
    # If first column looks like an index column (e.g. 'Unnamed: 0' or similar), use it
    first_col = df.columns[0]
    try:
        df[first_col] = pd.to_datetime(df[first_col], errors='raise')
        df = df.set_index(first_col)
    except Exception:
        # As a last resort, try to infer datetime from index after setting it
        df = df.set_index(first_col)
        try:
            df.index = pd.to_datetime(df.index, errors='raise')
        except Exception as e:
            raise ValueError(
                f"Could not parse any date column in {path}. "
                f"Tried 'Date', 'date', and first column '{first_col}'."
            ) from e
    return df


def load_train_test_features(
    data_dir: str = DATA_PROC,
    X_train_file: str = "eur_chf_train_X_scaled.csv",
    X_test_file: str = "eur_chf_test_X_scaled.csv",
    y_train_file: str = "eur_chf_train_y.csv",
    y_test_file: str = "eur_chf_test_y.csv",
    target_col: str = "y_next",
):
    """
    Load pre-built features and targets created in Part 2 (robust to date column naming).
    Returns:
        X_train, y_train, X_test, y_test  (all with DatetimeIndex)
    """
    # Read with flexible date parsing
    X_train = _read_with_flexible_date(os.path.join(data_dir, X_train_file))
    X_test  = _read_with_flexible_date(os.path.join(data_dir, X_test_file))
    y_train = _read_with_flexible_date(os.path.join(data_dir, y_train_file))
    y_test  = _read_with_flexible_date(os.path.join(data_dir, y_test_file))

    # Keep only numeric columns for X (defensive)
    X_train = X_train.select_dtypes(include=[np.number])
    X_test  = X_test.select_dtypes(include=[np.number])

    # Extract target as Series; allow a fallback if target_col differs
    if target_col not in y_train.columns or target_col not in y_test.columns:
        # Try common alternatives
        alt_names = [c for c in ["target", "y", "y_target", "next_return", "y_next"] if c in y_train.columns]
        if not alt_names:
            raise KeyError(
                f"Target column '{target_col}' not found in y_train/y_test. "
                f"Available in y_train: {list(y_train.columns)[:10]} ..."
            )
        target_col = alt_names[0]

    y_train = y_train[target_col].astype(float)
    y_test  = y_test[target_col].astype(float)

    # Sanity checks
    if not X_train.index.is_monotonic_increasing: X_train = X_train.sort_index()
    if not X_test.index.is_monotonic_increasing:  X_test  = X_test.sort_index()
    if not y_train.index.is_monotonic_increasing: y_train = y_train.sort_index()
    if not y_test.index.is_monotonic_increasing:  y_test  = y_test.sort_index()

    # Align exact dates between X and y (inner join) to avoid hidden misalignments
    X_train, y_train = X_train.align(y_train, join="inner", axis=0)
    X_test,  y_test  = X_test.align(y_test,   join="inner", axis=0)

    # Final checks
    assert y_train.index.equals(X_train.index), "Train X and y must share the same dates"
    assert y_test.index.equals(X_test.index),   "Test  X and y must share the same dates"
    return X_train, y_train, X_test, y_test



# =========================================================
# 3.0.B - Metrics (RMSE/MAE) + optional Directional Accuracy
# =========================================================
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(mean_absolute_error(y_true, y_pred))


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Percentage of times the sign of the prediction matches the sign of the actual.
    Useful for returns series when the level signal is small.
    """
    true_sign = np.sign(y_true)
    pred_sign = np.sign(y_pred)
    return float((true_sign == pred_sign).mean())


def evaluate_forecast(
    y_true: pd.Series,
    y_pred: pd.Series,
    model_name: str = "Model",
    extra: dict | None = None
):
    """
    Compute core metrics and return a flat dict for easy CSV/JSON logging.
    y_true / y_pred must be aligned by index (dates).
    """
    # Align (defensive)
    aligned = y_true.align(y_pred, join="inner")
    yt, yp = aligned[0].values, aligned[1].values

    out = {
        "model": model_name,
        "n": int(len(yt)),
        "rmse": rmse(yt, yp),
        "mae": mae(yt, yp),
        "directional_acc": directional_accuracy(yt, yp),
    }
    if extra:
        out.update(extra)
    return out



# =========================================================
# 3.0.C - Residual diagnostics (Ljung-Box) & quick plots
# =========================================================
def ljung_box_pvalue(residuals: pd.Series, lags: int = 20) -> float | None:
    """
    Ljung-Box test for autocorrelation in residuals.
    Returns the p-value for the max lag; None if statsmodels is missing.
    Interpretation: p-value > 0.05 => cannot reject "no autocorrelation".
    """
    if acorr_ljungbox is None:
        return None
    res = acorr_ljungbox(residuals, lags=[lags], return_df=True)
    # last row p-value
    return float(res["lb_pvalue"].iloc[-1])

def plot_forecast_vs_actuals(y_true, y_pred, title=""):
    plt.figure(figsize=(14,4))          # <- nouvelle figure
    ax = plt.gca()
    y_true.plot(ax=ax, label="Actual")
    y_pred.plot(ax=ax, label="Forecast")
    ax.legend()
    ax.set_title(title)
    plt.show()

def plot_residuals(residuals: pd.Series, title: str = "Residuals"):
    """Quick residual line plot and histogram."""
    fig, ax = plt.subplots(2, 1, figsize=(10, 6))
    residuals.plot(ax=ax[0])
    ax[0].set_title(title)
    ax[0].grid(True)
    ax[1].hist(residuals.dropna().values, bins=40)
    ax[1].set_title("Residuals histogram")
    plt.tight_layout()
    plt.show()


# =========================================================
# 3.0.D - TimeSeriesSplit helpers (expanding window)
# =========================================================
def make_expanding_tscv(n_splits: int = 5, min_train_size: int | None = None):
    """
    Create a TimeSeriesSplit with expanding windows.
    - n_splits: number of splits
    - min_train_size: ensures initial training window is not too small (optional)
    Note: scikit-learn's TimeSeriesSplit already uses consecutive folds in order.
    """
    tss = TimeSeriesSplit(n_splits=n_splits)
    tss.min_train_size = min_train_size  # for your own reference/logs
    return tss


# =========================================================
# 3.0.E - Walk-forward evaluation (Test set) with periodic refits
# =========================================================
def walk_forward_predict(
    model_factory,
    X_train_full: pd.DataFrame,
    y_train_full: pd.Series,
    X_test: pd.DataFrame,
    refit_every: int = 63,
    verbose: bool = True,
):
    """
    Generic walk-forward:
    - model_factory: a callable that returns a *fresh* model instance (to avoid state carryover).
                     Example: lambda: RandomForestRegressor(n_estimators=500, random_state=0)
                     or for statsmodels, a wrapper class with fit/predict methods.
    - X_train_full, y_train_full: the whole Train period (no test info!)
    - X_test: the Test period features to predict sequentially
    - refit_every: refit frequency in business days (e.g., 63 ~ 3 months)
    Returns:
        pd.Series of predictions indexed by X_test.index
    """
    preds = []
    # We start from the END of the train set, then roll forward across test in chunks
    train_X = X_train_full.copy()
    train_y = y_train_full.copy()

    dates = list(X_test.index)
    total = len(dates)

    for start in range(0, total, refit_every):
        end = min(start + refit_every, total)
        chunk_index = dates[start:end]
        X_chunk = X_test.loc[chunk_index]

        # Fresh model each refit
        model = model_factory()

        # Fit strictly on data available *before* the chunk
        if verbose:
            print(f"[walk-forward] Fitting on train up to {train_X.index[-1].date()} | predicting {chunk_index[0].date()} → {chunk_index[-1].date()}")

        model.fit(train_X.values, train_y.values)

        # Predict the current chunk
        y_chunk_pred = model.predict(X_chunk.values)
        preds.append(pd.Series(y_chunk_pred, index=chunk_index))

        # Roll the training window forward by appending the true y from this chunk
        # (We assume in realistic setting we observe true y after each step.)
        # Here we extend train with the *true* labels from the same dates.
        if chunk_index[-1] in train_X.index:
            # just in case, avoid duplicate append
            pass
        else:
            # Append both X and y truth up to 'end' of the chunk
            train_X = pd.concat([train_X, X_chunk], axis=0)
            # Important: we need the true y for these dates; they are not passed here,
            # so the caller should compute residuals separately or pass y_test and align outside.
            # To keep this util generic, we do not mutate y here (no leakage).
            # We'll compute residuals outside with y_test.

            # If you prefer an *online* update using true y available progressively,
            # you can modify this function to accept y_test and append it here.

            # For safety, we *do not* update train_y in this generic version.
            # This makes the refit happen on the original train only (conservative).
            # If you want rolling refits using accumulated truth, use the variant below.

            # ---- Variant (enable if you want accumulating truth) ----
            # train_y = pd.concat([train_y, y_test.loc[chunk_index]], axis=0)

            pass

    y_pred = pd.concat(preds).sort_index()
    return y_pred


def walk_forward_predict_with_truth_update(
    model_factory,
    X_train_full: pd.DataFrame,
    y_train_full: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    refit_every: int = 63,
    verbose: bool = True,
):
    """
    Same as walk_forward_predict but *accumulates* observed truth after each chunk,
    so each refit benefits from the most recent (test) observations (realistic in production).
    """
    preds = []
    train_X = X_train_full.copy()
    train_y = y_train_full.copy()

    dates = list(X_test.index)
    total = len(dates)

    for start in range(0, total, refit_every):
        end = min(start + refit_every, total)
        chunk_index = dates[start:end]
        X_chunk = X_test.loc[chunk_index]

        model = model_factory()
        if verbose:
            print(f"[walk-forward+] Fit up to {train_X.index[-1].date()} | predict {chunk_index[0].date()} → {chunk_index[-1].date()}")
        model.fit(train_X.values, train_y.values)

        y_chunk_pred = model.predict(X_chunk.values)
        preds.append(pd.Series(y_chunk_pred, index=chunk_index))

        # Now that we predicted this chunk, we assume we observe its truth and add it to train
        train_X = pd.concat([train_X, X_chunk], axis=0)
        train_y = pd.concat([train_y, y_test.loc[chunk_index]], axis=0)

    y_pred = pd.concat(preds).sort_index()
    return y_pred


# =========================================================
# 3.0.F - Lightweight logging (CSV/JSON) for metrics & preds
# =========================================================
def save_metrics(metrics_dict: dict, out_csv: str, out_json: str | None = None):
    """
    Append metrics to a CSV 'leaderboard' file; also dumps JSON if requested.
    """
    df_row = pd.DataFrame([metrics_dict])
    # Append or create
    if os.path.exists(out_csv):
        old = pd.read_csv(out_csv)
        new = pd.concat([old, df_row], ignore_index=True)
    else:
        new = df_row
    new.to_csv(out_csv, index=False)

    if out_json:
        with open(out_json, "w") as f:
            json.dump(metrics_dict, f, indent=2)


def save_predictions(y_pred: pd.Series, out_path: str):
    """
    Save predictions as CSV with a Date column.
    """
    out = y_pred.copy().to_frame("y_pred")
    out.index.name = "Date"
    out.reset_index().to_csv(out_path, index=False)


# =========================================================
# 3.0.G - Quick smoke test (optional)
# =========================================================
if __name__ == "__main__":
    # Try loading features to ensure file paths are correct.
    try:
        # On précise la bonne colonne cible
        X_tr, y_tr, X_te, y_te = load_train_test_features(target_col="target_next_return")
        print(f"[check] Train: {X_tr.shape}, Test: {X_te.shape}")
        print(f"[check] Date ranges: train {X_tr.index[0].date()} → {X_tr.index[-1].date()} | test {X_te.index[0].date()} → {X_te.index[-1].date()}")
    except FileNotFoundError as e:
        print("[check] Skipping load check (files not found yet):", e)
        
# %% =========================================================
# 3.1 - BASELINES (Naive, Zero, SMA, SES)
# =========================================================
banner("3.1 - BASELINES")



# Optionnel : lissage exponentiel simple (SES)
try:
    from statsmodels.tsa.holtwinters import SimpleExpSmoothing
except Exception:
    SimpleExpSmoothing = None

# --- Dossiers de sortie pour les baselines ---
BASE_PRED_DIR = os.path.join(PRED_DIR, "baselines")
BASE_FIG_DIR = os.path.join(FIG_DIR, "baselines")
os.makedirs(BASE_PRED_DIR, exist_ok=True)
os.makedirs(BASE_FIG_DIR, exist_ok=True)

# --- 1) Charger X/y (features + cible) déjà préparés en Partie 2 ---
# IMPORTANT : préciser le nom réel de ta cible (validé plus tôt)
X_tr, y_tr, X_te, y_te = load_train_test_features(target_col="target_next_return")

# On vérifie que la feature 'r_t' (retour courant) est dispo pour les baselines
if "r_t" not in X_tr.columns or "r_t" not in X_te.columns:
    raise KeyError("Feature 'r_t' is required for baselines (naive/SMA/SES) but not found in X_*.")

# ==================================================================
# A. ZERO-RETURN baseline: ŷ_{t+1} = 0
# ==================================================================
y_pred_zero = pd.Series(0.0, index=y_te.index, name="y_pred_zero")
metrics_zero = evaluate_forecast(y_te, y_pred_zero, model_name="Baseline_ZERO")
print("[ZERO] ", metrics_zero)

# Sauvegardes
save_predictions(y_pred_zero, os.path.join(BASE_PRED_DIR, "baseline_zero.csv"))
save_metrics(metrics_zero, os.path.join(DATA_PROC, "leaderboard.csv"))

# Visualisation
plot_forecast_vs_actuals(y_te, y_pred_zero, title="Baseline ZERO – Prévisions vs Réel (Test)")

# ==================================================================
# B. NAIVE-RETURN baseline: ŷ_{t+1} = r_t
#    (prédit le retour de demain par le retour d'aujourd'hui)
# ==================================================================
y_pred_naive = X_te["r_t"].copy()
y_pred_naive.name = "y_pred_naive"
metrics_naive = evaluate_forecast(y_te, y_pred_naive, model_name="Baseline_NAIVE")
print("[NAIVE]", metrics_naive)

save_predictions(y_pred_naive, os.path.join(BASE_PRED_DIR, "baseline_naive.csv"))
save_metrics(metrics_naive, os.path.join(DATA_PROC, "leaderboard.csv"))
plot_forecast_vs_actuals(y_te, y_pred_naive, title="Baseline NAIVE – Prévisions vs Réel (Test)")

# ==================================================================
# C. SMA-k baseline: ŷ_{t+1} = moyenne mobile_k(r_·) évaluée à t
#    (k ∈ {5, 21} typiquement ; aucune fuite car calculée au temps t)
# ==================================================================
def sma_forecast(X_train: pd.DataFrame, X_test: pd.DataFrame, k: int) -> pd.Series:
    """
    Construit la prédiction SMA-k pour y_{t+1} en utilisant r_t historique.
    Pour éviter toute fuite, on calcule la SMA sur la concaténation (train+test),
    puis on récupère les valeurs aux dates du test (la SMA à t n'utilise que ≤ t).
    """
    r_all = pd.concat([X_train["r_t"], X_test["r_t"]], axis=0).sort_index()
    sma_all = r_all.rolling(k, min_periods=k).mean()  # moyenne mobile sur k jours
    y_pred = sma_all.loc[X_test.index].copy()
    y_pred.name = f"y_pred_sma{k}"
    return y_pred

for k in [5, 21]:
    y_pred_sma = sma_forecast(X_tr, X_te, k=k)
    # Les premières dates peuvent être NaN si on n'a pas assez d'historique au tout début du train.
    # Sur le Test, on a suffisamment d'historique (car Train est long), mais on protège:
    y_pred_sma = y_pred_sma.fillna(0.0)

    metrics_sma = evaluate_forecast(y_te, y_pred_sma, model_name=f"Baseline_SMA{k}")
    print(f"[SMA{k}]", metrics_sma)

    save_predictions(y_pred_sma, os.path.join(BASE_PRED_DIR, f"baseline_sma{k}.csv"))
    save_metrics(metrics_sma, os.path.join(DATA_PROC, "leaderboard.csv"))
    plot_forecast_vs_actuals(y_te, y_pred_sma, title=f"Baseline SMA{k} – Prévisions vs Réel (Test)")

# ==================================================================
# D. SES (Simple Exponential Smoothing): ŷ_{t+1} = niveau_t
#    - On ajuste alpha sur le train.
#    - Prédiction pour la 1re date du Test = niveau à la fin du Train.
#    - Puis on met à jour le niveau séquentiellement avec le vrai r_t du Test.
# ==================================================================
def ses_walk_forward_returns(r_train: pd.Series, r_test: pd.Series, alpha: float | None = None) -> pd.Series:
    """
    Crée des prévisions one-step-ahead via SES pour des retours:
    - Si alpha est None, on l'estime sur r_train avec statsmodels.
    - La prévision au temps t (pour y_{t+1}) = 'niveau' après avoir observé r_{≤t}.
    - On met à jour le niveau séquentiellement en parcourant le Test.
    """
    # Estimation (alpha) sur le train
    if alpha is None:
        if SimpleExpSmoothing is None:
            raise ImportError("statsmodels est requis pour estimer alpha (SimpleExpSmoothing).")
        ses_model = SimpleExpSmoothing(r_train, initialization_method="estimated")
        ses_fit = ses_model.fit(optimized=True)
        alpha = ses_fit.params.get("smoothing_level", 0.2)  # fallback au cas où

    # Niveau à la fin du train (EMA avec alpha)
    level_train_last = r_train.ewm(alpha=alpha, adjust=False).mean().iloc[-1]

    # Parcours du Test en mode walk-forward (aucune fuite)
    preds = []
    level = float(level_train_last)
    for t in r_test.index:
        # prévision pour (t+1) = niveau courant (après avoir vu ≤ t-1, au départ après train)
        preds.append((t, level))
        # on observe r_t (véridique) et on met à jour le niveau
        r_t = float(r_test.loc[t])
        level = alpha * r_t + (1.0 - alpha) * level

    y_pred = pd.Series([p[1] for p in preds], index=[p[0] for p in preds], name="y_pred_ses")
    return y_pred, alpha

# Exécuter SES si possible
if SimpleExpSmoothing is not None:
    r_tr = X_tr["r_t"].astype(float)
    r_te = X_te["r_t"].astype(float)

    y_pred_ses, alpha_hat = ses_walk_forward_returns(r_tr, r_te, alpha=None)
    print(f"[SES] smoothing_level (alpha) estimé sur le train: {alpha_hat:.4f}")

    metrics_ses = evaluate_forecast(y_te, y_pred_ses, model_name="Baseline_SES", extra={"alpha": alpha_hat})
    print("[SES] ", metrics_ses)

    save_predictions(y_pred_ses, os.path.join(BASE_PRED_DIR, "baseline_ses.csv"))
    save_metrics(metrics_ses, os.path.join(DATA_PROC, "leaderboard.csv"))
    plot_forecast_vs_actuals(y_te, y_pred_ses, title="Baseline SES – Prévisions vs Réel (Test)")
else:
    print("[SES] statsmodels non disponible : baseline SES ignorée.")

# ==================================================================
# E. (Option) Diagnostics rapides sur résidus (autocorrélation)
# ==================================================================
for name, y_pred in [
    ("Baseline_ZERO", y_pred_zero),
    ("Baseline_NAIVE", y_pred_naive),
]:
    resid = (y_te - y_pred).rename("residual")
    pval = ljung_box_pvalue(resid, lags=20)
    print(f"[{name}] Ljung–Box p-value (lag 20): {pval}")


# %% =========================================================
# 3.1 - BASELINES (Naive, Zero, SMA, SES) — NON-SCALÉS
# =========================================================
banner("3.1 - BASELINES (unscaled returns)")



# SES (optionnel)
try:
    from statsmodels.tsa.holtwinters import SimpleExpSmoothing
except Exception:
    SimpleExpSmoothing = None

# --- Dossiers de sortie ---
BASE_PRED_DIR = os.path.join(PRED_DIR, "baselines")
BASE_FIG_DIR  = os.path.join(FIG_DIR,  "baselines")
os.makedirs(BASE_PRED_DIR, exist_ok=True)
os.makedirs(BASE_FIG_DIR,  exist_ok=True)

# --- Charger y (pour connaître les index Train/Test) ---
# (on réutilise le loader robuste de 3.0 et on précise la cible)
X_tr, y_tr, X_te, y_te = load_train_test_features(target_col="target_next_return")

# --- Charger la série de retours NON-SCALÉS depuis Part 2 ---
# On utilise le helper _read_with_flexible_date défini en 3.0
ret_path = os.path.join(DATA_PROC, "eur_chf_returns.csv")
ret_df   = _read_with_flexible_date(ret_path)

# Nom de colonne attendu pour les retours non-scalés

ret_col_candidates = ["r_t", "log_return", "return", "r"]

ret_col = None
for c in ret_col_candidates:
    if c in ret_df.columns:
        ret_col = c
        break
if ret_col is None:
    raise KeyError(f"Impossible de trouver la colonne des retours dans {ret_path} "
                   f"(cherché {ret_col_candidates}).")

# Série complète de retours non-scalés
r_all = ret_df[ret_col].astype(float).sort_index()

# Aligner la série des retours sur les index Train/Test de y
r_tr = r_all.loc[y_tr.index]
r_te = r_all.loc[y_te.index]

# Sanity check : mêmes index
assert r_tr.index.equals(y_tr.index)
assert r_te.index.equals(y_te.index)

# ==================================================================
# A. ZERO-RETURN : ŷ_{t+1} = 0
# ==================================================================
y_pred_zero = pd.Series(0.0, index=y_te.index, name="y_pred_zero")
metrics_zero = evaluate_forecast(y_te, y_pred_zero, model_name="Baseline_ZERO")
print("[ZERO] ", metrics_zero)
save_predictions(y_pred_zero, os.path.join(BASE_PRED_DIR, "baseline_zero.csv"))
save_metrics(metrics_zero, os.path.join(DATA_PROC, "leaderboard.csv"))
plot_forecast_vs_actuals(y_te, y_pred_zero, title="Baseline ZERO – Prévisions vs Réel (Test)")

# ==================================================================
# B. NAIVE-RETURN : ŷ_{t+1} = r_t  (r_t NON-SCALÉ)
# ==================================================================
y_pred_naive = r_te.copy()
y_pred_naive.name = "y_pred_naive"
metrics_naive = evaluate_forecast(y_te, y_pred_naive, model_name="Baseline_NAIVE")
print("[NAIVE]", metrics_naive)
save_predictions(y_pred_naive, os.path.join(BASE_PRED_DIR, "baseline_naive_unscaled.csv"))
save_metrics(metrics_naive, os.path.join(DATA_PROC, "leaderboard.csv"))
plot_forecast_vs_actuals(y_te, y_pred_naive, title="Baseline NAIVE (unscaled) – Prévisions vs Réel (Test)")

# ==================================================================
# C. SMA-k : ŷ_{t+1} = moyenne_mobile_k(r_·) évaluée à t (NON-SCALÉ)
# ==================================================================
def sma_forecast_unscaled(r_train: pd.Series, r_test: pd.Series, k: int) -> pd.Series:
    """
    SMA-k calculée sur la concaténation (train+test), puis on prend les valeurs aux dates du test.
    Aucune fuite : la SMA à t n'utilise que l'historique ≤ t.
    """
    r_concat = pd.concat([r_train, r_test], axis=0).sort_index()
    sma_all = r_concat.rolling(k, min_periods=k).mean()
    y_pred = sma_all.loc[r_test.index].copy()
    y_pred.name = f"y_pred_sma{k}"
    return y_pred

for k in [5, 21]:
    y_pred_sma = sma_forecast_unscaled(r_tr, r_te, k=k).fillna(0.0)
    metrics_sma = evaluate_forecast(y_te, y_pred_sma, model_name=f"Baseline_SMA{k}")
    print(f"[SMA{k}]", metrics_sma)
    save_predictions(y_pred_sma, os.path.join(BASE_PRED_DIR, f"baseline_sma{k}_unscaled.csv"))
    save_metrics(metrics_sma, os.path.join(DATA_PROC, "leaderboard.csv"))
    plot_forecast_vs_actuals(y_te, y_pred_sma, title=f"Baseline SMA{k} (unscaled) – Prévisions vs Réel (Test)")

# ==================================================================
# D. SES (Simple Exponential Smoothing) — strictly on UN-SCALED returns
# ==================================================================
from typing import Tuple

def ses_one_step_walk_forward(
    r_train: pd.Series,
    r_test: pd.Series,
    alpha: float | None = None
) -> Tuple[pd.Series, float]:
    """
    One-step-ahead forecasts for returns using SES (no leakage):
    - Fit (estimate alpha if needed) on r_train (unscaled).
    - Produce rolling one-step-ahead forecasts over the test by updating level sequentially.
    Returns:
        y_pred (Series aligned to r_test.index), alpha_hat (float)
    """
    # --- 0) Safety checks on scale (returns should be small, not prices) ---
    # If we see magnitudes >> 0.1, warn: that's probably prices or wrongly scaled data.
    if r_train.abs().max() > 0.1 or r_test.abs().max() > 0.1:
        print("[SES WARN] Input magnitudes look large (>0.1). Are these PRICES rather than RETURNS?")

    # --- 1) Estimate alpha on TRAIN if not provided ---
    if alpha is None:
        if SimpleExpSmoothing is None:
            raise ImportError("statsmodels is required for SES (SimpleExpSmoothing).")
        # Fit SES on the TRAIN returns only
        ses_model = SimpleExpSmoothing(r_train.astype(float), initialization_method="estimated")
        ses_fit = ses_model.fit(optimized=True)
        alpha_hat = float(ses_fit.params.get("smoothing_level", 0.2))
    else:
        alpha_hat = float(alpha)

    # --- 2) Initialize level at end of TRAIN using EMA with alpha_hat ---
    # On returns, this should be a small number near 0.
    level = float(r_train.ewm(alpha=alpha_hat, adjust=False).mean().iloc[-1])

    # --- 3) Walk-forward on TEST (no leakage) ---
    preds = []
    for t in r_test.index:
        preds.append((t, level))  # forecast for t+1 is current level
        # observe r_t and update
        r_t = float(r_test.loc[t])
        level = alpha_hat * r_t + (1.0 - alpha_hat) * level

    y_pred = pd.Series([p[1] for p in preds], index=[p[0] for p in preds], name="y_pred_ses")
    return y_pred, alpha_hat


# === Use the UN-SCALED returns we already built above (r_tr, r_te) ===
# (r_tr / r_te came from eur_chf_returns.csv column 'r_t' and were aligned to y_tr / y_te)

y_pred_ses, alpha_hat = ses_one_step_walk_forward(r_tr, r_te, alpha=None)
print(f"[SES] smoothing_level (alpha) estimated on train: {alpha_hat:.6g}")

metrics_ses = evaluate_forecast(y_te, y_pred_ses, model_name="Baseline_SES_unscaled", extra={"alpha": alpha_hat})
print("[SES] ", metrics_ses)

save_predictions(y_pred_ses, os.path.join(BASE_PRED_DIR, "baseline_ses_unscaled.csv"))
save_metrics(metrics_ses, os.path.join(DATA_PROC, "leaderboard.csv"))
plot_forecast_vs_actuals(y_te, y_pred_ses, title="Baseline SES (unscaled) – Prévisions vs Réel (Test)")


# ==================================================================
# E. Diagnostics Ljung–Box sur résidus (option)
# ==================================================================
for name, y_pred in [
    ("Baseline_ZERO",  y_pred_zero),
    ("Baseline_NAIVE", y_pred_naive),
]:
    resid = (y_te - y_pred).rename("residual")
    pval = ljung_box_pvalue(resid, lags=20)
    print(f"[{name}] Ljung–Box p-value (lag 20): {pval}")

# =========================================================
# 3.2 - ARIMA / ARIMAX (Classical Statistical Models)
# =========================================================



# --- Project paths ---
PROJECT_ROOT = "/Users/eliemenassa/Desktop/Projet Forecasting"
DATA_PROC    = os.path.join(PROJECT_ROOT, "data", "processed")
REPORTS_FIGS = os.path.join(PROJECT_ROOT, "reports", "figures", "arima")
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models", "arima")
PREDS_DIR    = os.path.join(DATA_PROC, "preds", "arima")

os.makedirs(REPORTS_FIGS, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PREDS_DIR, exist_ok=True)

# --- Utility banner ---
def banner(title: str):
    print("\n" + "="*len(title))
    print(title)
    print("="*len(title))

# =========================================================
# 3.2.A - Load returns (train/test)
# =========================================================
banner("3.2.A - Load returns")

ret_path = os.path.join(DATA_PROC, "eur_chf_returns.csv")
df = pd.read_csv(ret_path, parse_dates=["date"], index_col="date")

# Chronological split (as in Part 2)
split_date = "2023-01-01"
train = df.loc[:split_date]["r_t"].dropna()
test  = df.loc[split_date:]["r_t"].dropna()

print(f"Train: {train.index[0].date()} → {train.index[-1].date()} | {len(train)} obs")
print(f"Test : {test.index[0].date()} → {test.index[-1].date()} | {len(test)} obs")

# =========================================================
# 3.2.B - ACF / PACF diagnostics
# =========================================================
banner("3.2.B - ACF / PACF diagnostics")

fig, ax = plt.subplots(2,1, figsize=(12,6))
plot_acf(train, lags=40, ax=ax[0])
plot_pacf(train, lags=40, ax=ax[1])
fig.suptitle("ACF & PACF of Train Returns")
fig.tight_layout()
plt.savefig(os.path.join(REPORTS_FIGS, "acf_pacf_train.png"))

# =========================================================
# 3.2.C - Fit candidate ARIMA models
# =========================================================
banner("3.2.C - Fit candidate ARIMA models")

# Try small orders (p,q) since returns are near-stationary (d=0)
orders = [(1,0,1), (1,0,2), (2,0,1), (2,0,2), (5,0,1)]

results = []
for order in orders:
    try:
        model = sm.tsa.ARIMA(train, order=order)
        fit = model.fit()
        aic, bic = fit.aic, fit.bic
        results.append((order, aic, bic))
        print(f"ARIMA{order} -> AIC={aic:.2f}, BIC={bic:.2f}")
        # Save model summary
        with open(os.path.join(MODELS_DIR, f"arima_{order}.txt"), "w") as f:
            f.write(str(fit.summary()))
    except Exception as e:
        print(f"Failed for {order}: {e}")

# Pick best by AIC
best_order = min(results, key=lambda x: x[1])[0]
print(f"Selected best ARIMA order by AIC: {best_order}")

# =========================================================
# 3.2.D - Residual diagnostics
# =========================================================
banner("3.2.D - Residual diagnostics")

best_model = sm.tsa.ARIMA(train, order=best_order).fit()
resid = best_model.resid

fig, ax = plt.subplots(3,1, figsize=(12,8))
ax[0].plot(resid)
ax[0].set_title("Residuals")
plot_acf(resid, lags=40, ax=ax[1])
plot_pacf(resid, lags=40, ax=ax[2])
fig.tight_layout()
plt.savefig(os.path.join(REPORTS_FIGS, "residuals_diagnostics.png"))

# Ljung-Box test
lb_test = acorr_ljungbox(resid, lags=[20], return_df=True)

print("Ljung-Box (lag=20):")
print(lb_test)

# =========================================================
# 3.2.E - Forecast on test set
# =========================================================
banner("3.2.E - Forecast on test set")

forecast = best_model.get_forecast(steps=len(test))
pred_mean = forecast.predicted_mean
pred_ci   = forecast.conf_int()

# Save predictions
pred_df = pd.DataFrame({
    "y_true": test,
    "y_pred": pred_mean
}, index=test.index)
pred_df.to_csv(os.path.join(PREDS_DIR, "arima_predictions.csv"))

# Plot forecast vs actual
fig, ax = plt.subplots(figsize=(14,5))
test.plot(ax=ax, label="Actual")
pred_mean.plot(ax=ax, label="ARIMA forecast")
ax.fill_between(pred_ci.index, pred_ci.iloc[:,0], pred_ci.iloc[:,1], color="gray", alpha=0.3)
ax.legend()
ax.set_title(f"ARIMA{best_order} - Forecast vs Actual")
plt.savefig(os.path.join(REPORTS_FIGS, "arima_forecast_vs_actual.png"))

print(pred_df.head())


# %% =========================================================
# 3.3 - AR(1)-GARCH(1,1): point forecast + risk (volatility)
# =========================================================
banner("3.3 - AR(1)-GARCH(1,1) – Returns & Volatility")



# arch pour GARCH
try:
    from arch import arch_model
    HAS_ARCH = True
except Exception:
    HAS_ARCH = False
    print("[3.3] Package 'arch' introuvable. Installe-le via: pip install arch")

GARCH_MODEL_DIR = os.path.join(MODELS_DIR, "garch")
GARCH_PRED_DIR  = os.path.join(PRED_DIR,   "garch")
GARCH_FIG_DIR   = os.path.join(FIG_DIR,    "garch")
os.makedirs(GARCH_MODEL_DIR, exist_ok=True)
os.makedirs(GARCH_PRED_DIR,  exist_ok=True)
os.makedirs(GARCH_FIG_DIR,   exist_ok=True)

if HAS_ARCH:
    # ---------------------------
    # 1) Charger les retours (non scalés)
    # ---------------------------
    ret_path = os.path.join(DATA_PROC, "eur_chf_returns.csv")
    df_ret = _read_with_flexible_date(ret_path)

    ret_col = None
    for c in ["r_t", "log_return", "return", "r"]:
        if c in df_ret.columns:
            ret_col = c
            break
    if ret_col is None:
        raise KeyError(f"[3.3] Colonne returns introuvable dans {ret_path}. Colonnes: {list(df_ret.columns)}")

    r_all = df_ret[ret_col].astype(float).sort_index()

    split_date = pd.Timestamp("2023-01-01")
    r_train = r_all.loc[r_all.index < split_date]
    r_test  = r_all.loc[r_all.index >= split_date]

    print("Train:", r_train.index[0].date(), "→", r_train.index[-1].date(), "|", len(r_train), "obs")
    print("Test :", r_test.index[0].date(),  "→", r_test.index[-1].date(),  "|", len(r_test),  "obs")

    # ---------------------------
    # 2) Fit AR(1)-GARCH(1,1)
    # ---------------------------
    am = arch_model(
        r_train,
        mean="ARX", lags=1,
        vol="GARCH", p=1, q=1,
        dist="t",
        rescale=True  # aide l'optimisation (r petits)
    )
    res = am.fit(disp="off", update_freq=0)

    with open(os.path.join(GARCH_MODEL_DIR, "ar1_garch11_summary.txt"), "w") as f:
        f.write(res.summary().as_text())

    # ---------------------------
    # 3) Prévisions out-of-sample (walk-forward robuste)
    # ---------------------------
    # On refit toutes les 63 séances (≈ trimestre), comme pour RF, pour rester propre.
    refit_every = 63

    mu_list = []
    s2_list = []
    idx_list = []

    # On itère sur des blocs consécutifs du test
    test_dates = r_test.index
    start_pos = 0
    while start_pos < len(test_dates):
        end_pos = min(start_pos + refit_every, len(test_dates))
        # Fenêtre d'entraînement = tout jusqu’à la veille du bloc courant
        last_train_date = test_dates[start_pos] - pd.tseries.offsets.BDay(1)
        r_train_blk = r_all.loc[:last_train_date]

        # Fit AR(1)-GARCH(1,1) sur la fenêtre courante
        am_blk = arch_model(
            r_train_blk,
            mean="ARX", lags=1,
            vol="GARCH", p=1, q=1,
            dist="t",
            rescale=True
        )
        res_blk = am_blk.fit(disp="off", update_freq=0)

        # Pour chaque jour du bloc, on fait une prévision 1-step ahead
        blk_dates = test_dates[start_pos:end_pos]
        # Ici, comme les arbres, on garde le même fit et on lit la prévision 1-step
        # en recalculant un forecast au début du bloc, puis on répète la même
        # (les paramètres GARCH évoluent peu d’un jour à l’autre à ce pas).
        fcast_blk = res_blk.forecast(horizon=1, start=blk_dates[0], align="target")
        mu_blk    = fcast_blk.mean["h.1"].reindex(blk_dates)
        s2_blk    = fcast_blk.variance["h.1"].reindex(blk_dates)

        # Si (selon version arch) align="target" donne du vide, on retente "origin"
        if mu_blk.dropna().empty or s2_blk.dropna().empty:
            fcast_blk = res_blk.forecast(horizon=1, start=blk_dates[0], align="origin")
            mu_blk    = fcast_blk.mean["h.1"].reindex(blk_dates)
            s2_blk    = fcast_blk.variance["h.1"].reindex(blk_dates)

        # Ultime sécurité : si encore vide (rare), on boucle jour par jour
        if mu_blk.dropna().empty or s2_blk.dropna().empty:
            mu_blk = pd.Series(index=blk_dates, dtype=float)
            s2_blk = pd.Series(index=blk_dates, dtype=float)
            for d in blk_dates:
                # entraînement jusqu’à la veille de d
                r_train_day = r_all.loc[:(d - pd.tseries.offsets.BDay(1))]
                am_day = arch_model(
                    r_train_day, mean="ARX", lags=1,
                    vol="GARCH", p=1, q=1,
                    dist="t", rescale=True
                )
                res_day = am_day.fit(disp="off", update_freq=0)
                f_day = res_day.forecast(horizon=1)
                mu_blk.loc[d] = f_day.mean.iloc[-1, 0]
                s2_blk.loc[d] = f_day.variance.iloc[-1, 0]

        mu_list.append(mu_blk)
        s2_list.append(s2_blk)
        idx_list.append(blk_dates)

        start_pos = end_pos

    mu_hat = pd.concat(mu_list).sort_index()
    sigma2_hat = pd.concat(s2_list).sort_index()
    sigma_hat = np.sqrt(sigma2_hat)

    # ---------------------------
    # 4) Aligner & évaluer (drop NaN)
    # ---------------------------
    aligned = pd.concat([r_test, mu_hat, sigma_hat, sigma2_hat], axis=1)
    aligned.columns = ["y_true", "y_pred", "sigma", "sigma2"]
    aligned = aligned.dropna()
    if aligned.empty:
        raise ValueError("[3.3] Walk-forward: aucune date alignée (vérifie l'index business day).")

    y_true_aligned = aligned["y_true"]
    y_pred_aligned = aligned["y_pred"]

    garch_point_metrics = evaluate_forecast(
        y_true=y_true_aligned,
        y_pred=y_pred_aligned,
        model_name="AR1_GARCH11_point",
        extra={"dist": "t", "mean": "AR(1)", "vol": "GARCH(1,1)", "wf_refit": refit_every}
    )
    print("[3.3] Point-forecast metrics:", garch_point_metrics)
    save_metrics(garch_point_metrics, os.path.join(DATA_PROC, "leaderboard.csv"))

    # Vol metrics
    def mse(a, b):
        a = np.asarray(a); b = np.asarray(b)
        return float(np.mean((a - b) ** 2))

    def qlike(r2, sig2):
        sig2 = np.clip(np.asarray(sig2), 1e-12, None)
        return float(np.mean(np.log(sig2) + np.asarray(r2) / sig2))

    r2_test_aligned = (y_true_aligned ** 2).values
    vol_metrics = {
        "model": "AR1_GARCH11_vol",
        "n": int(len(y_true_aligned)),
        "mse_r2_vs_sigma2": mse(r2_test_aligned, aligned["sigma2"].values),
        "mse_absr_vs_sigma": mse(np.abs(y_true_aligned.values), aligned["sigma"].values),
        "qlike": qlike(r2_test_aligned, aligned["sigma2"].values),
        "wf_refit": refit_every
    }
    print("[3.3] Volatility metrics:", vol_metrics)
    save_metrics(vol_metrics, os.path.join(DATA_PROC, "leaderboard.csv"))

    # Sauvegarde + plots
    aligned.to_csv(os.path.join(GARCH_PRED_DIR, "ar1_garch11_preds.csv"), index_label="date")
    plot_forecast_vs_actuals(
        y_true=y_true_aligned,
        y_pred=y_pred_aligned,
        title="AR(1)-GARCH(1,1) – Forecast vs Actual (Returns) [walk-forward]"
    )

    plt.figure(figsize=(12,4))
    plt.plot(aligned.index, aligned["sigma"].values, label="Predicted σ(t+1)")
    plt.plot(aligned.index, np.abs(aligned["y_true"].values), label="Realized |r_t|", alpha=0.6)
    plt.title("AR(1)-GARCH(1,1) – Predicted volatility vs Realized |r| [walk-forward]")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(GARCH_FIG_DIR, "garch_vol_vs_absr.png")); plt.show()

    plt.figure(figsize=(12,4))
    plt.plot(aligned.index, aligned["sigma2"].values, label="Predicted σ²(t+1)")
    plt.plot(aligned.index, (aligned["y_true"].values ** 2), label="Realized r_t²", alpha=0.6)
    plt.title("AR(1)-GARCH(1,1) – Predicted variance vs Realized r² [walk-forward]")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(GARCH_FIG_DIR, "garch_var_vs_r2.png")); plt.show()
    
 # %% =========================================================
# 3.5 - TREE-BASED MODELS (RandomForest, XGBoost if available)
# =========================================================
banner("3.5 - TREE-BASED MODELS (RF, XGB)")


# XGBoost (optionnel)
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

# SHAP (optionnel)
try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

# ---------------------------
# Dossiers de sortie
# ---------------------------
TREES_MODEL_DIR = os.path.join(MODELS_DIR, "ml_trees")
TREES_PRED_DIR  = os.path.join(PRED_DIR,   "ml_trees")
TREES_FIG_DIR   = os.path.join(FIG_DIR,    "ml_trees")
os.makedirs(TREES_MODEL_DIR, exist_ok=True)
os.makedirs(TREES_PRED_DIR,  exist_ok=True)
os.makedirs(TREES_FIG_DIR,   exist_ok=True)

# ---------------------------
# Chargement des données
# ---------------------------
# On veut les FEATURES NON SCALÉES (pour arbres)
# Fichiers produits en partie 2 : eur_chf_train_features.csv / eur_chf_test_features.csv
feat_train_path = os.path.join(DATA_PROC, "eur_chf_train_features.csv")
feat_test_path  = os.path.join(DATA_PROC, "eur_chf_test_features.csv")
y_train_path    = os.path.join(DATA_PROC, "eur_chf_train_y.csv")
y_test_path     = os.path.join(DATA_PROC, "eur_chf_test_y.csv")

X_tr_raw = _read_with_flexible_date(feat_train_path)
X_te_raw = _read_with_flexible_date(feat_test_path)
y_tr_df  = _read_with_flexible_date(y_train_path)
y_te_df  = _read_with_flexible_date(y_test_path)

# Détecter la cible (nous savons que tu utilises 'target_next_return')
target_col = "target_next_return"
if target_col not in y_tr_df.columns or target_col not in y_te_df.columns:
    # fallback (robuste)
    for c in ["y_next", "next_return", "target", "y"]:
        if c in y_tr_df.columns and c in y_te_df.columns:
            target_col = c
            break

y_tr = y_tr_df[target_col].astype(float)
y_te = y_te_df[target_col].astype(float)

# Features: garder uniquement numériques et s'assurer de l'alignement
X_tr = X_tr_raw.select_dtypes(include=[np.number]).copy()
X_te = X_te_raw.select_dtypes(include=[np.number]).copy()

# Important: on retire la colonne cible si elle est dedans par erreur
if target_col in X_tr.columns: X_tr = X_tr.drop(columns=[target_col])
if target_col in X_te.columns: X_te = X_te.drop(columns=[target_col])

# Sanity checks & alignement
X_tr, y_tr = X_tr.align(y_tr, join="inner", axis=0)
X_te, y_te = X_te.align(y_te, join="inner", axis=0)
assert X_tr.index.equals(y_tr.index) and X_te.index.equals(y_te.index), "Dates X/y désalignées"

feature_cols = X_tr.columns.tolist()
print(f"[3.5] Features used ({len(feature_cols)}): {feature_cols}")

# ---------------------------
# Aide: CV chrono (expanding)
# ---------------------------
def ts_cv_rmse(model, X, y, n_splits=5, min_train_size=None, verbose=False):
    """
    TimeSeriesSplit (fenêtre croissante). Retourne la moyenne du RMSE sur les folds.
    """
    tss = TimeSeriesSplit(n_splits=n_splits)
    rmses = []
    for i, (tr_idx, va_idx) in enumerate(tss.split(X)):
        X_tr_cv, X_va_cv = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr_cv, y_va_cv = y.iloc[tr_idx], y.iloc[va_idx]
        if min_train_size and len(X_tr_cv) < min_train_size:
            continue
        m = model
        m.fit(X_tr_cv.values, y_tr_cv.values)
        y_hat = m.predict(X_va_cv.values)
        fold_rmse = rmse(y_va_cv.values, y_hat)
        rmses.append(fold_rmse)
        if verbose:
            print(f"  - Fold {i+1}: RMSE={fold_rmse:.6f}")
    return float(np.mean(rmses)) if rmses else np.inf

# ---------------------------
# RandomForest: tuning léger
# ---------------------------
banner("3.5.A - RandomForest (tuning léger)")

rf_grid = [
    {"n_estimators": 400, "max_depth": None, "min_samples_leaf": 1, "max_features": "sqrt", "random_state": 42, "n_jobs": -1},
    {"n_estimators": 800, "max_depth": None, "min_samples_leaf": 1, "max_features": "sqrt", "random_state": 42, "n_jobs": -1},
    {"n_estimators": 600, "max_depth": 10,   "min_samples_leaf": 3, "max_features": "sqrt", "random_state": 42, "n_jobs": -1},
    {"n_estimators": 600, "max_depth": 5,    "min_samples_leaf": 5, "max_features": "sqrt", "random_state": 42, "n_jobs": -1},
]

best_rf_cfg, best_rf_cv = None, np.inf
for cfg in rf_grid:
    rf = RandomForestRegressor(**cfg)
    cv_rmse = ts_cv_rmse(rf, X_tr, y_tr, n_splits=5, verbose=False)
    print(f"RF {cfg} -> CV_RMSE={cv_rmse:.6f}")
    if cv_rmse < best_rf_cv:
        best_rf_cv = cv_rmse
        best_rf_cfg = cfg

print(f"[RF] Selected config: {best_rf_cfg} with CV_RMSE={best_rf_cv:.6f}")

# Walk-forward sur Test (refit périodique)
rf_factory = lambda: RandomForestRegressor(**best_rf_cfg)
y_rf_pred = walk_forward_predict_with_truth_update(
    model_factory=rf_factory,
    X_train_full=X_tr,
    y_train_full=y_tr,
    X_test=X_te,
    y_test=y_te,
    refit_every=63,
    verbose=True,
)
rf_metrics = evaluate_forecast(y_te, y_rf_pred, model_name="RF_WF63", extra={"cv_rmse": best_rf_cv})
print("[RF] ", rf_metrics)

save_predictions(y_rf_pred, os.path.join(TREES_PRED_DIR, "rf_wf63.csv"))
save_metrics(rf_metrics, os.path.join(DATA_PROC, "leaderboard.csv"))
plot_forecast_vs_actuals(y_te, y_rf_pred, title="RandomForest (walk-forward) – Prévisions vs Réel")

# Feature importances (barplot top 12)
rf = rf_factory()
rf.fit(X_tr.values, y_tr.values)
importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
topk = importances.head(12)
plt.figure(figsize=(10,4))
topk.plot(kind="barh")
plt.gca().invert_yaxis()
plt.title("RandomForest – Top feature importances")
plt.tight_layout()
plt.savefig(os.path.join(TREES_FIG_DIR, "rf_feature_importances.png"))
plt.show()

# ---------------------------
# XGBoost (optionnel) : tuning léger
# ---------------------------
if HAS_XGB:
    banner("3.5.B - XGBoost (tuning léger)")
    xgb_grid = [
        {"n_estimators": 400, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8,
         "reg_lambda": 1.0, "random_state": 42, "tree_method": "hist", "verbosity": 0},
        {"n_estimators": 800, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8,
         "reg_lambda": 1.0, "random_state": 42, "tree_method": "hist", "verbosity": 0},
        {"n_estimators": 600, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8,
         "reg_lambda": 1.0, "random_state": 42, "tree_method": "hist", "verbosity": 0},
    ]
    best_xgb_cfg, best_xgb_cv = None, np.inf
    for cfg in xgb_grid:
        xgb = XGBRegressor(**cfg)
        cv_rmse = ts_cv_rmse(xgb, X_tr, y_tr, n_splits=5, verbose=False)
        print(f"XGB {cfg} -> CV_RMSE={cv_rmse:.6f}")
        if cv_rmse < best_xgb_cv:
            best_xgb_cv = cv_rmse
            best_xgb_cfg = cfg
    print(f"[XGB] Selected config: {best_xgb_cfg} with CV_RMSE={best_xgb_cv:.6f}")

    xgb_factory = lambda: XGBRegressor(**best_xgb_cfg)
    y_xgb_pred = walk_forward_predict_with_truth_update(
        model_factory=xgb_factory,
        X_train_full=X_tr,
        y_train_full=y_tr,
        X_test=X_te,
        y_test=y_te,
        refit_every=63,
        verbose=True,
    )
    xgb_metrics = evaluate_forecast(y_te, y_xgb_pred, model_name="XGB_WF63", extra={"cv_rmse": best_xgb_cv})
    print("[XGB]", xgb_metrics)

    save_predictions(y_xgb_pred, os.path.join(TREES_PRED_DIR, "xgb_wf63.csv"))
    save_metrics(xgb_metrics, os.path.join(DATA_PROC, "leaderboard.csv"))
    plot_forecast_vs_actuals(y_te, y_xgb_pred, title="XGBoost (walk-forward) – Prévisions vs Réel")

    # Importances (gain)
    xgb = xgb_factory()
    xgb.fit(X_tr.values, y_tr.values)
    try:
        booster = xgb.get_booster()
        gain_dict = booster.get_score(importance_type="gain")
        # Mapper les noms de colonnes (xgb les appelle f0,f1,...) à feature_cols
        # Ordre: XGB prend les colonnes dans l'ordre de X_tr.values
        fmap = {f"f{i}": col for i, col in enumerate(feature_cols)}
        gain_series = pd.Series({fmap.get(k, k): v for k, v in gain_dict.items()}).sort_values(ascending=False)
    except Exception:
        # fallback: feature_importances_
        gain_series = pd.Series(xgb.feature_importances_, index=feature_cols).sort_values(ascending=False)

    topk = gain_series.head(12)
    plt.figure(figsize=(10,4))
    topk.plot(kind="barh")
    plt.gca().invert_yaxis()
    plt.title("XGBoost – Top feature importances (gain)")
    plt.tight_layout()
    plt.savefig(os.path.join(TREES_FIG_DIR, "xgb_feature_importances.png"))
    plt.show()

    # SHAP (optionnel, si lib présente)
    if HAS_SHAP:
        try:
            explainer = shap.TreeExplainer(xgb)
            # échantillon léger pour l’illustration
            sample_idx = np.random.RandomState(42).choice(len(X_tr), size=min(500, len(X_tr)), replace=False)
            shap_values = explainer.shap_values(X_tr.iloc[sample_idx])
            shap.summary_plot(shap_values, X_tr.iloc[sample_idx], show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(TREES_FIG_DIR, "xgb_shap_summary.png"))
            plt.show()
        except Exception as e:
            print("[SHAP] skipped:", e)
else:
    print("[XGB] xgboost non disponible — on passe.")
   


# ================================
# 3.4 - Linear ML (Ridge / Lasso)
# ================================
# --- BEFORE 3.4 / juste avant de construire X_train, y_train ---
# Harmoniser la colonne cible (certains fichiers ont 'r_t' ou 'log_return')
possible_targets = ["r_t", "log_return", "target", "y", "y_true"]

def harmonize_target(df):
    present = [c for c in possible_targets if c in df.columns]
    if not present:
        raise ValueError(
            f"Aucune colonne cible trouvée dans df.columns={list(df.columns)}. "
            f"Attendu l’une de {possible_targets}."
        )
    tcol = present[0]
    if tcol != "target":
        df = df.rename(columns={tcol: "target"})
    return df

# Si ce n'est pas déjà fait plus haut, assure-toi que df_tr/df_te sont chargés ici
# df_tr = pd.read_csv(os.path.join(DATA_PROC, "eur_chf_features_train.csv"))
# df_te = pd.read_csv(os.path.join(DATA_PROC, "eur_chf_features_test.csv"))

df_tr = harmonize_target(df_tr)
df_te = harmonize_target(df_te)

# Optionnel : gérer l’index date si présent
for df in (df_tr, df_te):
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
    df.sort_index(inplace=True)

# Définir les features = toutes colonnes sauf 'target' (et 'date' si restée)
feature_cols = [c for c in df_tr.columns if c not in ["target"]]
# (si tu préfères whitelister)
# feature_cols = ['lag_1','lag_5','lag_10','lag_21',
#                 'roll_mean_5','roll_std_5','roll_mean_21','roll_std_21',
#                 'roll_mean_63','roll_std_63','r_t']  # si dispo, 'r_t' sera déjà renommée 'target'

# Nettoyage NA cohérent train/test
df_tr = df_tr.dropna(subset=["target"] + feature_cols)
df_te = df_te.dropna(subset=["target"] + feature_cols)

X_train, y_train = df_tr[feature_cols].values, df_tr["target"].values
X_test,  y_test  = df_te[feature_cols].values, df_te["target"].values
print(f"[3.4] Shapes → X_train:{X_train.shape} X_test:{X_test.shape}  |  y_train:{y_train.shape} y_test:{y_test.shape}")



banner("3.4 - Linear ML (Ridge, Lasso) with Pipeline + TimeSeriesSplit")



# --- Paths ---
MODELS_DIR  = os.path.join(PROJECT_ROOT, "models")
FIG_DIR     = os.path.join(PROJECT_ROOT, "reports", "figures")
PREDS_DIR   = os.path.join(PROJECT_ROOT, "data", "processed", "preds")
METRICS_DIR = os.path.join(MODELS_DIR, "metrics")
DATA_PROC   = os.path.join(PROJECT_ROOT, "data", "processed")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(PREDS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

# ------------------------------------------------------------------
# 3.4.A - Chargement robuste des features (ou reconstruction)
# ------------------------------------------------------------------
def _read_features_safe(path: str) -> pd.DataFrame | None:
    """Read a feature CSV and set the datetime index to 'date' if present."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p, parse_dates=["date"])
        if "date" in df.columns:
            df = df.set_index("date")
        elif df.columns[0].lower().startswith("unnamed"):
            df = df.set_index(df.columns[0])
            df.index = pd.to_datetime(df.index, errors="coerce")
        return df.sort_index()
    except Exception as e:
        print(f"[WARN] Failed to read {p.name}: {e}")
        return None

def _read_target_series_safe(path: str):
    """Read y CSV, detect target column among ['target','log_return','r_t','y'] and return a Series named 'target' indexed by date."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p, parse_dates=["date"]).set_index("date").sort_index()
        for c in ["target", "log_return", "r_t", "y"]:
            if c in df.columns:
                return df[c].rename("target")
        print(f"[WARN] No target-like column found in {p.name}. Columns={list(df.columns)}")
        return None
    except Exception as e:
        print(f"[WARN] Failed to read target from {p.name}: {e}")
        return None

# Essaye plusieurs fichiers de features
candidates_train = [
    os.path.join(DATA_PROC, "eur_chf_features_train.csv"),
    os.path.join(DATA_PROC, "eur_chf_features_train_unscaled.csv"),
    os.path.join(DATA_PROC, "eur_chf_features_train_scaled.csv"),
]
candidates_test = [
    os.path.join(DATA_PROC, "eur_chf_features_test.csv"),
    os.path.join(DATA_PROC, "eur_chf_features_test_unscaled.csv"),
    os.path.join(DATA_PROC, "eur_chf_features_test_scaled.csv"),
]

df_tr, df_te = None, None
for path in candidates_train:
    df_tr = _read_features_safe(path)
    if df_tr is not None:
        print(f"[3.4] Loaded TRAIN features from: {path}")
        break

for path in candidates_test:
    df_te = _read_features_safe(path)
    if df_te is not None:
        print(f"[3.4] Loaded TEST features from: {path}")
        break

rebuilt = False
# Si pas de fichiers → on reconstruit à partir de eur_chf_returns.csv
if (df_tr is None) or (df_te is None):
    print("[3.4] Feature files not found. Rebuilding from eur_chf_returns.csv ...")
    returns_path = os.path.join(DATA_PROC, "eur_chf_returns.csv")
    if not os.path.exists(returns_path):
        raise FileNotFoundError("eur_chf_returns.csv not found in DATA_PROC.")

    df0 = pd.read_csv(returns_path, parse_dates=["date"]).set_index("date").sort_index()

    # Patch : s'assurer qu'une colonne log_return existe
    if "log_return" not in df0.columns:
        if "price" in df0.columns:
            df0["log_return"] = np.log(df0["price"]).diff()
        elif "return" in df0.columns:
            df0.rename(columns={"return": "log_return"}, inplace=True)
        else:
            raise KeyError("No 'log_return', 'price', or 'return' in eur_chf_returns.csv")

    # Split chronologique (ajuste si nécessaire)
    split_date = pd.Timestamp("2023-01-01")
    df_tr_raw = df0.loc[: split_date - pd.Timedelta(days=1)].copy()
    df_te_raw = df0.loc[split_date:].copy()

    def build_features(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        for L in [1, 2, 3, 5, 10]:
            out[f"lag_{L}"] = out["log_return"].shift(L)
        for W in [5, 10, 20]:
            out[f"roll_mean_{W}"] = out["log_return"].rolling(W, min_periods=1).mean()
            out[f"roll_std_{W}"]  = out["log_return"].rolling(W, min_periods=1).std(ddof=0)
        out["target"] = out["log_return"].shift(-1)
        return out.dropna()

    df_tr = build_features(df_tr_raw)
    df_te = build_features(df_te_raw)
    rebuilt = True

    # Sauvegarde pour réutilisation
    df_tr.reset_index().to_csv(os.path.join(DATA_PROC, "eur_chf_features_train.csv"), index=False)
    df_te.reset_index().to_csv(os.path.join(DATA_PROC, "eur_chf_features_test.csv"), index=False)
    print("[3.4] Features rebuilt and saved.")

# ------------------------------------------------------------------
# 3.4.B - Chargement robuste de y (cible) + alignement
# ------------------------------------------------------------------
# On privilégie les fichiers dédiés y; si absents, on infère depuis les features
y_tr = _read_target_series_safe(os.path.join(DATA_PROC, "eur_chf_train_y.csv"))
y_te = _read_target_series_safe(os.path.join(DATA_PROC, "eur_chf_test_y.csv"))

def _infer_target_from_features(df_feat: pd.DataFrame) -> pd.Series | None:
    for c in ["target", "log_return", "r_t", "y"]:
        if c in df_feat.columns:
            return df_feat[c].rename("target")
    return None

if y_tr is None or y_te is None:
    print("[3.4] Dedicated y files not found or missing target column; inferring from features.")
    y_tr = y_tr if y_tr is not None else _infer_target_from_features(df_tr)
    y_te = y_te if y_te is not None else _infer_target_from_features(df_te)
    if y_tr is None or y_te is None:
        raise ValueError("Impossible de déterminer la colonne cible (target/log_return/r_t/y).")

# Important: retirer toute éventuelle colonne cible des features
for c in ["target", "log_return", "r_t", "y"]:
    if c in df_tr.columns: df_tr = df_tr.drop(columns=[c])
    if c in df_te.columns: df_te = df_te.drop(columns=[c])

# Alignement par date (inner join) + dropna
df_train_full = df_tr.join(y_tr, how="inner")
df_test_full  = df_te.join(y_te, how="inner")

# Nettoyage NA (sécurité)
df_train_full = df_train_full.dropna()
df_test_full  = df_test_full.dropna()

feature_cols = [c for c in df_train_full.columns if c != "target"]

X_train, y_train = df_train_full[feature_cols].values, df_train_full["target"].values
X_test,  y_test  = df_test_full[feature_cols].values,  df_test_full["target"].values

print(f"[3.4] Feature cols ({len(feature_cols)}): {feature_cols}")
print(f"[3.4] Shapes → X_train:{X_train.shape} X_test:{X_test.shape} | y_train:{y_train.shape} y_test:{y_test.shape}")

# ------------------------------------------------------------------
# 3.4.C - Fonctions utilitaires
# ------------------------------------------------------------------
def compute_metrics(y_true, y_pred, prefix="test"):
    mse  = mean_squared_error(y_true, y_pred)   # MSE
    rmse = float(np.sqrt(mse))                  # RMSE
    mae  = mean_absolute_error(y_true, y_pred)
    eps  = 1e-12
    mape = float(np.mean(np.abs((y_true + eps) - (y_pred + eps)) / (np.abs(y_true) + eps)))
    return {
        f"{prefix}_RMSE": rmse,
        f"{prefix}_MAE": mae,
        f"{prefix}_MAPE": mape
    }

def log_metrics(model_name, metrics: dict):
    csv_path = os.path.join(METRICS_DIR, "metrics_log.csv")
    row = {"model": model_name, **metrics}
    pd.DataFrame([row]).to_csv(csv_path, mode="a", header=not os.path.exists(csv_path), index=False)
    with open(os.path.join(METRICS_DIR, f"{model_name}_metrics.json"), "w") as f:
        json.dump(row, f, indent=2)

def save_predictions(model_name, index_train, index_test, yhat_tr, yhat_te):
    pd.DataFrame({"date": index_train, "yhat": yhat_tr}).set_index("date") \
        .to_csv(os.path.join(PREDS_DIR, f"{model_name}_train.csv"))
    pd.DataFrame({"date": index_test, "yhat": yhat_te}).set_index("date") \
        .to_csv(os.path.join(PREDS_DIR, f"{model_name}_test.csv"))

def plot_coefficients(model_name, estimator, feature_names):
    coefs = getattr(estimator, "coef_", None)
    if coefs is None: return
    coef_df = pd.DataFrame({"feature": feature_names, "coef": coefs}).sort_values("coef")
    plt.figure(figsize=(8, 5))
    plt.barh(coef_df["feature"], coef_df["coef"])
    plt.title(f"{model_name} - Coefficients")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"{model_name}_coefficients.png"), dpi=150)
    plt.close()

# ------------------------------------------------------------------
# 3.4.D - Entraînement Ridge & Lasso
# ------------------------------------------------------------------
tscv = TimeSeriesSplit(n_splits=5)

linear_specs = {
    "Ridge": {"estimator": Ridge(random_state=42),
              "param_grid": {"model__alpha": np.logspace(-4, 3, 12)}},
    "Lasso": {"estimator": Lasso(random_state=42, max_iter=10000),
              "param_grid": {"model__alpha": np.logspace(-4, 1, 10)}},
}

for name, spec in linear_specs.items():
    banner(f"3.4 - Training {name}")
    pipe = Pipeline([("scaler", StandardScaler()), ("model", spec["estimator"])])
    grid = GridSearchCV(pipe, spec["param_grid"],
                        scoring="neg_root_mean_squared_error",
                        cv=tscv, n_jobs=-1, refit=True)
    grid.fit(X_train, y_train)
    best_est = grid.best_estimator_
    best_alpha = grid.best_params_.get("model__alpha", None)

    # Predictions + metrics
    yhat_train = best_est.predict(X_train)
    yhat_test  = best_est.predict(X_test)
    m_tr = compute_metrics(y_train, yhat_train, prefix="train")
    m_te = compute_metrics(y_test, yhat_test, prefix="test")
    metrics_all = {**m_tr, **m_te, "best_alpha": best_alpha}
    log_metrics(f"Linear_{name}", metrics_all)

    # Save outputs
    save_predictions(f"Linear_{name}", df_train_full.index, df_test_full.index, yhat_train, yhat_test)
    joblib.dump(best_est, os.path.join(MODELS_DIR, f"Linear_{name}.joblib"))
    plot_coefficients(f"Linear_{name}", best_est.named_steps["model"], feature_cols)

    print(f"[{name}] Best alpha: {best_alpha}")
    print(f"[{name}] Test RMSE={m_te['test_RMSE']:.6f} | MAE={m_te['test_MAE']:.6f} | MAPE={m_te['test_MAPE']:.6f}")

    
 # ==========================================
# 3.7 - Model Comparison & Leaderboard
# ==========================================
banner("3.7 - Model Comparison & Leaderboard")

# This section assumes each previous model (Naive, ARIMA, GARCH, RF, Linear) called log_metrics()
# and saved test_* metrics into models/metrics/metrics_log.csv.
# If earlier sections wrote to different paths, adapt METRICS_DIR.

metrics_log_path = os.path.join(METRICS_DIR, "metrics_log.csv")
if not os.path.exists(metrics_log_path):
    raise FileNotFoundError("metrics_log.csv not found. Ensure each model logs metrics via log_metrics().")

metrics_df = pd.read_csv(metrics_log_path)

# Keep latest run per model (if multiple)
metrics_df["row_id"] = np.arange(len(metrics_df))
metrics_latest = metrics_df.sort_values("row_id").groupby("model", as_index=False).tail(1)

# Leaderboard = sort by Test RMSE ascending
sort_col = "test_RMSE"
if sort_col not in metrics_latest.columns:
    # Backward compatibility: derive RMSE if only MSE exists (not the case here, but just in case)
    raise KeyError(f"Expected column '{sort_col}' not found in metrics.")

leaderboard = metrics_latest.sort_values(sort_col, ascending=True).reset_index(drop=True)
leaderboard_path = os.path.join(MODELS_DIR, "leaderboard.csv")
leaderboard.to_csv(leaderboard_path, index=False)

print("Leaderboard saved:", leaderboard_path)
print(leaderboard[["model", "test_RMSE", "test_MAE", "test_MAPE"]])

# --- Plot: bar chart of Test RMSE ---
plt.figure(figsize=(9, 5))
plt.bar(leaderboard["model"], leaderboard["test_RMSE"])
plt.title("Model Comparison - Test RMSE (lower is better)")
plt.xticks(rotation=20, ha="right")
plt.ylabel("RMSE")
plt.tight_layout()
fig1 = os.path.join(FIG_DIR, "leaderboard_test_rmse.png")
plt.savefig(fig1, dpi=150)
plt.close()

# --- Plot: scatter Bias vs Variance proxy (Train vs Test RMSE) ---
if "train_RMSE" in leaderboard.columns:
    plt.figure(figsize=(7, 5))
    plt.scatter(leaderboard["train_RMSE"], leaderboard["test_RMSE"])
    for i, m in enumerate(leaderboard["model"]):
        plt.annotate(m, (leaderboard["train_RMSE"][i], leaderboard["test_RMSE"][i]), fontsize=8, xytext=(5,5), textcoords="offset points")
    plt.xlabel("Train RMSE")
    plt.ylabel("Test RMSE")
    plt.title("Generalization (Train vs Test RMSE)")
    plt.tight_layout()
    fig2 = os.path.join(FIG_DIR, "leaderboard_generalization.png")
    plt.savefig(fig2, dpi=150)
    plt.close()

# --- Residual check plot, if you want to inspect the best model quickly ---
best_model = leaderboard.iloc[0]["model"]
print(f"Best model by Test RMSE: {best_model}")

# Load preds to compare against true test
best_pred_path = os.path.join(PREDS_DIR, f"{best_model}_test.csv")
if os.path.exists(best_pred_path):
    y_true = df_te[target_col].rename("y_true")
    y_hat  = pd.read_csv(best_pred_path, parse_dates=["date"]).set_index("date")["yhat"].rename("yhat")
    comp   = pd.concat([y_true, y_hat], axis=1).dropna()

    # Time plot
    plt.figure(figsize=(10, 4))
    plt.plot(comp.index, comp["y_true"], label="True (Test)")
    plt.plot(comp.index, comp["yhat"], label=f"Pred ({best_model})")
    plt.title(f"Test Predictions vs Actuals - {best_model}")
    plt.legend()
    plt.tight_layout()
    fig3 = os.path.join(FIG_DIR, f"{best_model}_test_pred_vs_actual.png")
    plt.savefig(fig3, dpi=150)
    plt.close()

    # Residuals histogram
    res = comp["y_true"] - comp["yhat"]
    plt.figure(figsize=(6,4))
    plt.hist(res, bins=40)
    plt.title(f"Residuals - {best_model}")
    plt.tight_layout()
    fig4 = os.path.join(FIG_DIR, f"{best_model}_residuals_hist.png")
    plt.savefig(fig4, dpi=150)
    plt.close()
# =========================================================
# 3.8 - Interpretation & Applied Finance (console only)
# =========================================================
banner("3.8 - Interpretation & Applied Finance (EN + FR)")

import re

# -- Helper: find leaderboard with robust paths
lb_paths = [
    os.path.join(MODELS_DIR, "leaderboard.csv"),
    os.path.join(DATA_PROC,   "leaderboard.csv"),
    os.path.join(PROJECT_ROOT, "models", "leaderboard.csv"),
]
lb_path_found = None
for p in lb_paths:
    if os.path.exists(p):
        lb_path_found = p
        break

if lb_path_found is None:
    print("[3.8] No leaderboard.csv found. Skipping interpretation.")
else:
    lb = pd.read_csv(lb_path_found)

    # -- Normalize column names (lowercase) for robustness
    lb.columns = [c.strip().lower() for c in lb.columns]

    # -- Find key columns (model, rmse, mae, mape) flexibly
    def pick_col(cands):
        for c in cands:
            if c in lb.columns:
                return c
        return None

    col_model = pick_col(["model", "name"])
    col_rmse  = pick_col(["test_rmse", "rmse"])
    col_mae   = pick_col(["test_mae", "mae"])
    col_mape  = pick_col(["test_mape", "mape"])

    if col_model is None or col_rmse is None:
        print("[3.8] Leaderboard is missing required columns (model/rmse). Columns:", lb.columns.tolist())
    else:
        # -- Sort by RMSE asc to get winner
        lb_sorted = lb.sort_values(by=col_rmse, ascending=True).reset_index(drop=True)
        winner = lb_sorted.loc[0, col_model]
        w_rmse = float(lb_sorted.loc[0, col_rmse])
        w_mae  = float(lb_sorted.loc[0, col_mae]) if col_mae in lb_sorted.columns else float("nan")
        w_mape = float(lb_sorted.loc[0, col_mape]) if col_mape in lb_sorted.columns else float("nan")

        # -- Try to find a baseline (Naive/Zero/SMA*)
        baseline_pattern = re.compile(r"(naive|baseline_naive|baseline_zero|zero|sma\d*)", re.IGNORECASE)
        lb_baselines = lb_sorted[lb_sorted[col_model].str.contains(baseline_pattern, na=False)]
        delta_txt = "Baseline comparison not available."
        if not lb_baselines.empty:
            # Prioritize Naive > Zero > SMA*
            def baseline_priority(name: str) -> int:
                n = name.lower()
                if "naive" in n: return 0
                if "zero"  in n: return 1
                if "sma"   in n: return 2
                return 3

            lb_baselines = lb_baselines.copy()
            lb_baselines["prio"] = lb_baselines[col_model].apply(baseline_priority)
            base_row = lb_baselines.sort_values(by=["prio", col_rmse], ascending=[True, True]).iloc[0]
            b_name = str(base_row[col_model])
            b_rmse = float(base_row[col_rmse])
            imp = (b_rmse - w_rmse) / b_rmse * 100.0
            delta_txt = f"Improvement vs baseline [{b_name}] (Test RMSE): {imp:.2f}%"

        # -- Optional: top-5 table (pretty)
        topk = min(5, len(lb_sorted))
        top_view = lb_sorted[[col_model, col_rmse] + ([col_mae] if col_mae else []) + ([col_mape] if col_mape else [])].head(topk)

        # ======================
        # Console report (EN)
        # ======================
        print("\n" + "="*70)
        print("Part 3 – Results & Interpretation (EUR/CHF)  [EN]")
        print("="*70)
        print(f"Winner: {winner}")
        print(f"Test metrics → RMSE={w_rmse:.6f} | MAE={w_mae:.6f}" + (f" | MAPE={w_mape:.6f}" if not np.isnan(w_mape) else ""))
        print(delta_txt)
        print("\nTop models (by Test RMSE):")
        print(top_view.to_string(index=False))

        print("\nKey takeaways:")
        print("- Short-horizon FX return forecasts are noisy; even small RMSE gains can help position sizing/risk limits.")
        print("- Linear Ridge tends to be strong with your features (lags, rolling stats).")
        print("- AR-GARCH is better for volatility (risk bands) than for return point forecasts.")
        print("- Tree-based models capture non-linearities but need careful time-aware validation.")

        print("\nNotes:")
        print("- MAPE on returns can explode (division by values ~0): prefer RMSE/MAE for returns.")
        print("- Always evaluate residuals and use walk-forward validation to avoid leakage.\n")

        # ======================
        # Console report (FR)
        # ======================
        print("="*70)
        print("Partie 3 – Résultats & Interprétation (EUR/CHF)  [FR]")
        print("="*70)
        print(f"Gagnant : {winner}")
        print(f"Métriques Test → RMSE={w_rmse:.6f} | MAE={w_mae:.6f}" + (f" | MAPE={w_mape:.6f}" if not np.isnan(w_mape) else ""))
        print(delta_txt)
        print("\nTop modèles (par RMSE Test) :")
        print(top_view.to_string(index=False))

        print("\nEnseignements clés :")
        print("- À court horizon, les rendements FX sont très bruiteux ; même de petits gains de RMSE aident pour le sizing/les limites de risque.")
        print("- Ridge linéaire fonctionne bien avec tes features (lags, stats roulantes).")
        print("- AR-GARCH sert surtout à la volatilité (bandes de risque), pas aux prévisions ponctuelles de retours.")
        print("- Les modèles à arbres capturent des non-linéarités mais exigent une validation temporelle soignée.")

        print("\nRemarques :")
        print("- Le MAPE sur des retours proches de 0 peut exploser ; privilégier RMSE/MAE.")
        print("- Toujours contrôler les résidus et utiliser du walk-forward pour éviter les fuites.\n")


   