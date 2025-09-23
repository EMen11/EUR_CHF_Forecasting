# =========================================================
# 4.0 - Setup (Baselines)  + robust datetime index + chrono fix
# =========================================================

# ---- Imports
import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Optional, List

warnings.filterwarnings("ignore")

# ---- Pretty banner
def banner(title: str, ch: str = "="):
    line = ch * max(len(title), 40)
    print("\n" + line)
    print(title)
    print(line)

banner("4.0 - Setup (Baselines)")

# ---- Directories (reuse if already defined in your session)
PROJECT_ROOT = globals().get("PROJECT_ROOT", os.getcwd())

DATA_PROC = globals().get(
    "DATA_PROC",
    os.path.join(PROJECT_ROOT, "data", "processed")
)

# Baseline results directory
RESULTS_BASE = os.path.join(PROJECT_ROOT, "results", "baseline")
FIGS_DIR     = os.path.join(RESULTS_BASE, "figs")
CACHE_DIR    = os.path.join(RESULTS_BASE, "cache")

for d in [RESULTS_BASE, FIGS_DIR, CACHE_DIR]:
    os.makedirs(d, exist_ok=True)

print("PROJECT_ROOT:", PROJECT_ROOT)
print("DATA_PROC   :", DATA_PROC)
print("RESULTS_BASE:", RESULTS_BASE)

# ---- Robust CSV/Parquet reader
def _try_read(path: Path) -> Optional[pd.DataFrame]:
    """Try reading CSV or Parquet depending on suffix; return None if fail."""
    try:
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path, low_memory=False)
            return df
        elif path.suffix.lower() in (".parquet", ".pq"):
            df = pd.read_parquet(path)
            return df
        else:
            return None
    except Exception:
        return None

def load_first_available(basenames: List[str], search_dir: str) -> pd.DataFrame:
    """
    Given base filenames (without extension), try in order:
    - <name>.csv, <name>.parquet
    Return the first DataFrame found; raise if none found.
    """
    candidates = []
    for base in basenames:
        candidates.append(Path(search_dir) / f"{base}.csv")
        candidates.append(Path(search_dir) / f"{base}.parquet")

    for p in candidates:
        if p.exists():
            df = _try_read(p)
            if df is not None and len(df) > 0:
                print(f"[load] Loaded: {p.name} ({len(df)} rows)")
                return df

    raise FileNotFoundError(
        f"None of the expected files found in {search_dir}. "
        f"Tried: {', '.join([c.name for c in candidates])}"
    )

# ---- Ensure a proper DatetimeIndex
def ensure_datetime_index(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """
    Make sure df has a proper DatetimeIndex based on a likely date column.
    Priority columns: 'Date','date','ds','timestamp','Datetime','datetime','DATE'
    Fallback: try to infer any column that parses to datetime for >=90% rows.
    """
    # if already a DatetimeIndex and sane, return
    if isinstance(df.index, pd.DatetimeIndex):
        dtmin = pd.to_datetime(df.index.min(), errors="coerce")
        if not pd.isna(dtmin) and dtmin.year > 2000:
            df = df.sort_index()
            print(f"[idx] {source_name}: DatetimeIndex ok → {df.index.min().date()}..{df.index.max().date()}")
            return df

    # candidates by name
    candidate_names = ["Date","date","ds","timestamp","Datetime","datetime","DATE"]
    for c in candidate_names:
        if c in df.columns:
            dt = pd.to_datetime(df[c], errors="coerce", utc=False)
            ok_ratio = dt.notna().mean()
            if ok_ratio >= 0.9:
                df = df.assign(_dt=dt).dropna(subset=["_dt"]).drop(columns=[c]).set_index("_dt").sort_index()
                df.index.name = "Date"
                print(f"[idx] {source_name}: using column '{c}' as DatetimeIndex "
                      f"({int(ok_ratio*100)}% parsed)")
                return df

    # generic inference over object-like columns
    obj_cols = [c for c in df.columns if df[c].dtype == "object" or "datetime" in str(df[c].dtype).lower()]
    for c in obj_cols:
        dt = pd.to_datetime(df[c], errors="coerce", utc=False)
        ok_ratio = dt.notna().mean()
        if ok_ratio >= 0.9:
            df = df.assign(_dt=dt).dropna(subset=["_dt"]).set_index("_dt").sort_index()
            df.index.name = "Date"
            print(f"[idx] {source_name}: inferred '{c}' as DatetimeIndex "
                  f"({int(ok_ratio*100)}% parsed)")
            return df

    # last resort: if there is an "Unnamed: 0" exported index that is a date-like string
    for c in df.columns:
        if "unnamed" in c.lower():
            dt = pd.to_datetime(df[c], errors="coerce", utc=False)
            if dt.notna().mean() >= 0.9:
                df = df.assign(_dt=dt).dropna(subset=["_dt"]).drop(columns=[c]).set_index("_dt").sort_index()
                df.index.name = "Date"
                print(f"[idx] {source_name}: using '{c}' (unnamed) as DatetimeIndex")
                return df

    # If we reach here, fail with diagnostic
    print(f"[err] {source_name}: could not infer a datetime index. Columns: {list(df.columns)}")
    raise ValueError(
        f"{source_name}: no usable date column found. "
        f"Add a 'Date' column or ensure one of {candidate_names} exists."
    )

def pick_target_col(df: pd.DataFrame) -> str:
    """
    Choose the target column for returns forecasting.
    Priority: 'log_return', 'return', 'y'
    """
    for c in ["log_return", "return", "y"]:
        if c in df.columns:
            return c
    if df.shape[1] == 1:
        return df.columns[0]
    raise ValueError(
        f"Could not find a target column among ['log_return','return','y'] in columns: {list(df.columns)}"
    )

def validate_splits(train: pd.DataFrame, test: pd.DataFrame, target_col: str):
    """Basic sanity checks for index, NaNs, overlap, and chronological order."""
    assert isinstance(train.index, pd.DatetimeIndex), "Train index must be DatetimeIndex."
    assert isinstance(test.index, pd.DatetimeIndex),  "Test index must be DatetimeIndex."

    assert train.index.is_monotonic_increasing, "Train index must be sorted ascending."
    assert test.index.is_monotonic_increasing,  "Test index must be sorted ascending."

    nan_train = train[target_col].isna().sum()
    nan_test  = test[target_col].isna().sum()
    if nan_train > 0 or nan_test > 0:
        print(f"[warn] NaNs detected in target → train:{nan_train}, test:{nan_test}. Dropping...")
        train.dropna(subset=[target_col], inplace=True)
        test.dropna(subset=[target_col], inplace=True)

    overlap = train.index.intersection(test.index)
    assert len(overlap) == 0, f"Train/Test date ranges overlap on {len(overlap)} timestamps."

    assert train.index.max() < test.index.min(), (
        f"Train max date ({train.index.max().date()}) should be < Test min date ({test.index.min().date()})."
    )

    print("[ok] Splits validated: indices are datetime, sorted, disjoint, and chronological.")

# ---- Load datasets (prefer the Part 2 outputs)
banner("4.0 - Loading Train/Test")

train_candidates = ["eur_chf_train", "eur_chf_features_train"]
test_candidates  = ["eur_chf_test",  "eur_chf_features_test"]

df_train_raw = load_first_available(train_candidates, DATA_PROC)
df_test_raw  = load_first_available(test_candidates,  DATA_PROC)

# ---- Normalize to DatetimeIndex (robust)
df_train = ensure_datetime_index(df_train_raw.copy(), "train")
df_test  = ensure_datetime_index(df_test_raw.copy(),  "test")

# ---- Choose target column (log returns expected)
target_col = pick_target_col(df_train)
print(f"[info] Target column selected: {target_col}")

if target_col not in df_test.columns:
    raise ValueError(f"Target '{target_col}' not found in test columns: {list(df_test.columns)}")

# ---- Keep only target for baselines (univariate)
y_train = df_train[[target_col]].copy()
y_test  = df_test[[target_col]].copy()

# =======================================================
# 4.0.A - Fix chrono split if needed (robuste)
# =======================================================
banner("4.0.A - Fix chrono split (robust)")

def _safe_dt_min(idx: pd.DatetimeIndex) -> pd.Timestamp:
    """Return a safe min date; if clearly wrong (<2000), raise."""
    dtmin = pd.to_datetime(idx.min(), errors="coerce")
    if pd.isna(dtmin) or dtmin.year < 2000:
        raise ValueError(f"[err] Suspicious test min date: {dtmin}. Check parsing/index.")
    return dtmin

true_test_min = _safe_dt_min(df_test.index)

overlap = y_train.index.intersection(y_test.index)
need_fix = len(overlap) > 0 or not (y_train.index.max() < y_test.index.min())

if need_fix:
    n_train_before, n_test_before = len(y_train), len(y_test)
    y_train = y_train.loc[y_train.index < true_test_min].copy()
    y_test  = y_test.loc[y_test.index >= true_test_min].copy()

    print(f"[fix] Applied chrono cut @ {true_test_min.date()}")
    print(f"[fix] Train: {n_train_before} -> {len(y_train)} rows")
    print(f"[fix] Test : {n_test_before} -> {len(y_test)} rows")

    assert len(y_test) > 0, "[err] Test set became empty after split fix."
    assert len(y_train) > 0, (
        "[err] Train set is empty after split fix. "
        "Probable cause: original train already started >= first test date.\n"
        "→ Vérifie les fichiers part2 (eur_chf_train/test) ou recalcule le split."
    )
else:
    print("[ok] No overlap detected. Chronology already clean.")

# 3) Cache versions corrigées
CLEAN_TAG = "chronofix"
y_train.to_csv(os.path.join(CACHE_DIR, f"y_train_{CLEAN_TAG}.csv"))
y_test.to_csv(os.path.join(CACHE_DIR,  f"y_test_{CLEAN_TAG}.csv"))

# 4) Re-validation stricte
validate_splits(y_train, y_test, target_col)

# ---- Quick summary & cache
summary = {
    "project_root": PROJECT_ROOT,
    "data_processed_dir": DATA_PROC,
    "results_baseline_dir": RESULTS_BASE,
    "target_col": target_col,
    "train": {
        "rows": int(len(y_train)),
        "start": y_train.index.min().strftime("%Y-%m-%d"),
        "end":   y_train.index.max().strftime("%Y-%m-%d"),
    },
    "test": {
        "rows": int(len(y_test)),
        "start": y_test.index.min().strftime("%Y-%m-%d"),
        "end":   y_test.index.max().strftime("%Y-%m-%d"),
    },
}

with open(os.path.join(RESULTS_BASE, "part4_setup_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

# Save tiny previews to help debugging
y_train.tail(3).to_csv(os.path.join(CACHE_DIR, "y_train_tail.csv"))
y_test.head(3).to_csv(os.path.join(CACHE_DIR, "y_test_head.csv"))

banner("4.0 - Setup complete")
print(json.dumps(summary, indent=2, ensure_ascii=False))

# =========================================================
# 4.1 - Baseline Naïve :  y_{t+1} = y_t
# =========================================================
banner("4.1 - Baseline Naïve")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path

# --- Helpers (metrics + leaderboard)
def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def mape_safe(y_true, y_pred, eps=1e-8):
    # Pour des returns ~0, la MAPE est peu pertinente; on l'affiche quand même en "best effort"
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)

def save_leaderboard(row_dict, models_dir=None, data_proc_dir=None):
    """
    Append/update leaderboard.csv où qu'il se trouve déjà (Partie 3).
    Si le fichier existe mais n'a pas toutes les colonnes attendues, on le met à niveau.
    """
    required_cols = ["model","part","train_start","train_end","test_start","test_end",
                     "test_rmse","test_mae","test_mape","notes"]

    # chemins possibles (comme en partie 3)
    from pathlib import Path
    candidates = []
    if models_dir:
        candidates.append(Path(models_dir) / "leaderboard.csv")
    if data_proc_dir:
        candidates.append(Path(data_proc_dir) / "leaderboard.csv")
    candidates.append(Path(PROJECT_ROOT) / "models" / "leaderboard.csv")
    candidates.append(Path(DATA_PROC) / "leaderboard.csv")

    # trouver le premier existant
    target = None
    for p in candidates:
        if p.exists():
            target = p
            break
    if target is None:
        target = Path(PROJECT_ROOT) / "models" / "leaderboard.csv"
        target.parent.mkdir(parents=True, exist_ok=True)

    # charger / créer
    if target.exists():
        lb = pd.read_csv(target)
        # Mise à niveau des colonnes manquantes
        missing = [c for c in required_cols if c not in lb.columns]
        if missing:
            for c in missing:
                # types par défaut raisonnables
                if c in ["test_rmse","test_mae","test_mape"]:
                    lb[c] = np.nan
                else:
                    lb[c] = ""
            # réordonner
            lb = lb[[c for c in required_cols if c in lb.columns] + 
                    [c for c in lb.columns if c not in required_cols]]
    else:
        lb = pd.DataFrame(columns=required_cols)

    # si le modèle existe déjà sur le même split, on remplace la ligne
    # (sécurité: assure que toutes les clés existent dans row_dict)
    for c in required_cols:
        if c not in row_dict:
            row_dict[c] = np.nan if c in ["test_rmse","test_mae","test_mape"] else ""

    mask_same = (lb["model"] == row_dict["model"]) & \
                (lb["train_start"] == row_dict["train_start"]) & \
                (lb["test_start"]  == row_dict["test_start"])
    lb = lb.loc[~mask_same].copy()
    lb = pd.concat([lb, pd.DataFrame([row_dict])], ignore_index=True)

    # Réordonner colonnes essentielles devant
    ordered = required_cols + [c for c in lb.columns if c not in required_cols]
    lb = lb[ordered]

    lb.to_csv(target, index=False)
    print(f"[save] Leaderboard updated → {target}")


# --- Build prediction: use last observed value (shift by 1 on concatenated series)
# y_all = train ∪ test, puis on récupère les prédictions alignées sur l'index de test
y_all = pd.concat([y_train, y_test], axis=0)
y_pred_test = y_all[target_col].shift(1).reindex(y_test.index)  # première prédiction test = dernière valeur de train

# --- Align & drop NaNs if any (should be none for test here)
pred_ok = pd.DataFrame({
    "y_true": y_test[target_col],
    "y_pred": y_pred_test
}).dropna()

# --- Metrics
rmse_val = rmse(pred_ok["y_true"], pred_ok["y_pred"])
mae_val  = mae(pred_ok["y_true"], pred_ok["y_pred"])
mape_val = mape_safe(pred_ok["y_true"], pred_ok["y_pred"])

print(f"[metrics] Naïve → RMSE={rmse_val:.6f} | MAE={mae_val:.6f} | MAPE%≈{mape_val:.2f}")

# --- Save predictions & residuals
pred_path = os.path.join(RESULTS_BASE, "naive_predictions.csv")
resid_path = os.path.join(RESULTS_BASE, "naive_residuals.csv")
pred_ok.to_csv(pred_path)
(pred_ok["y_true"] - pred_ok["y_pred"]).to_frame("residual").to_csv(resid_path)
print(f"[save] Predictions → {pred_path}")
print(f"[save] Residuals   → {resid_path}")

# --- Figure: y_true vs y_pred (test)
fig_path = os.path.join(FIGS_DIR, "naive_test_vs_pred.png")
plt.figure(figsize=(10,4))
plt.plot(pred_ok.index, pred_ok["y_true"], label="Test (y_true)")
plt.plot(pred_ok.index, pred_ok["y_pred"], label="Naïve forecast (y_pred)")
plt.title("Baseline Naïve — Test vs Prediction")
plt.xlabel("Date"); plt.ylabel("Log-returns")
plt.legend()
plt.tight_layout()
plt.savefig(fig_path, dpi=150)
plt.close()
print(f"[save] Figure → {fig_path}")

# --- Update leaderboard
row = {
    "model": "Baseline_Naive",
    "part": "4.1",
    "train_start": y_train.index.min().strftime("%Y-%m-%d"),
    "train_end":   y_train.index.max().strftime("%Y-%m-%d"),
    "test_start":  y_test.index.min().strftime("%Y-%m-%d"),
    "test_end":    y_test.index.max().strftime("%Y-%m-%d"),
    "test_rmse": rmse_val,
    "test_mae":  mae_val,
    "test_mape": mape_val,
    "notes": "Forecast y_{t+1}=y_t on log-returns; first test pred seeded by last train value."
}
save_leaderboard(row, models_dir=os.path.join(PROJECT_ROOT,"models"), data_proc_dir=DATA_PROC)

banner("4.1 - Naïve baseline complete")

# =========================================================
# 4.2 - Moving Average (k = 5, 10, 20)
# =========================================================
banner("4.2 - Moving Average")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Helpers (safe if already defined previously)
def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def mape_safe(y_true, y_pred, eps=1e-8):
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)

# NOTE: On suppose que save_leaderboard(row, ...) a été patchée (voir 4.1).
#       Si tu exécutes 4.2 directement après 4.1, elle est déjà disponible.

# --- Paramètres
windows = [5, 10, 20]

# --- Série concaténée pour calculer les moyennes glissantes sans fuite
y_all = pd.concat([y_train, y_test], axis=0)

metrics_rows = []

for k in windows:
    # 1) Prévision = moyenne des k derniers retours, décalée d’un jour
    #    => prédiction pour la date t = moyenne (t-1, t-2, ..., t-k)
    y_ma = y_all[target_col].rolling(window=k, min_periods=k).mean().shift(1)
    y_pred_test = y_ma.reindex(y_test.index)

    # 2) Alignement & drop NaN (au cas où premières dates)
    pred_ok = pd.DataFrame({
        "y_true": y_test[target_col],
        "y_pred": y_pred_test
    }).dropna()

    # 3) Métriques
    rmse_val = rmse(pred_ok["y_true"], pred_ok["y_pred"])
    mae_val  = mae(pred_ok["y_true"], pred_ok["y_pred"])
    mape_val = mape_safe(pred_ok["y_true"], pred_ok["y_pred"])

    print(f"[metrics] MA(k={k}) → RMSE={rmse_val:.6f} | MAE={mae_val:.6f} | MAPE%≈{mape_val:.2f}")

    # 4) Sauvegardes CSV
    pred_path  = os.path.join(RESULTS_BASE, f"ma_k{k}_predictions.csv")
    resid_path = os.path.join(RESULTS_BASE, f"ma_k{k}_residuals.csv")
    pred_ok.to_csv(pred_path)
    (pred_ok["y_true"] - pred_ok["y_pred"]).to_frame("residual").to_csv(resid_path)
    print(f"[save] k={k} Predictions → {pred_path}")
    print(f"[save] k={k} Residuals   → {resid_path}")

    # 5) Figure individuelle
    fig_path = os.path.join(FIGS_DIR, f"ma_k{k}_test_vs_pred.png")
    plt.figure(figsize=(10,4))
    plt.plot(pred_ok.index, pred_ok["y_true"], label="Test (y_true)")
    plt.plot(pred_ok.index, pred_ok["y_pred"], label=f"MA(k={k}) forecast")
    plt.title(f"Moving Average (k={k}) — Test vs Prediction")
    plt.xlabel("Date"); plt.ylabel("Log-returns")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"[save] k={k} Figure → {fig_path}")

    # 6) Leaderboard
    row = {
        "model": f"Baseline_MA_k{k}",
        "part": "4.2",
        "train_start": y_train.index.min().strftime("%Y-%m-%d"),
        "train_end":   y_train.index.max().strftime("%Y-%m-%d"),
        "test_start":  y_test.index.min().strftime("%Y-%m-%d"),
        "test_end":    y_test.index.max().strftime("%Y-%m-%d"),
        "test_rmse": rmse_val,
        "test_mae":  mae_val,
        "test_mape": mape_val,
        "notes": f"Rolling mean of last {k} days on log-returns, shifted by 1 (no look-ahead)."
    }
    save_leaderboard(row, models_dir=os.path.join(PROJECT_ROOT,"models"), data_proc_dir=DATA_PROC)

    # 7) Collecte pour tableau récap
    metrics_rows.append({"k": k, "rmse": rmse_val, "mae": mae_val, "mape": mape_val})

# --- Tableau récapitulatif et bar chart des RMSE
df_metrics = pd.DataFrame(metrics_rows).sort_values("k")
metrics_csv = os.path.join(RESULTS_BASE, "ma_summary_metrics.csv")
df_metrics.to_csv(metrics_csv, index=False)
print(f"[save] Summary metrics → {metrics_csv}")

# Bar chart des RMSE
bar_path = os.path.join(FIGS_DIR, "ma_rmse_comparison.png")
plt.figure(figsize=(7,4))
plt.bar(df_metrics["k"].astype(str), df_metrics["rmse"])
plt.title("Moving Average — RMSE by window size")
plt.xlabel("Window size k"); plt.ylabel("RMSE")
plt.tight_layout()
plt.savefig(bar_path, dpi=150)
plt.close()
print(f"[save] RMSE bar chart → {bar_path}")

banner("4.2 - Moving Average complete")

# =========================================================
# 4.3 - Simple Exponential Smoothing (SES)
# =========================================================
banner("4.3 - Simple Exponential Smoothing (SES)")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

# --- Helpers (si déjà définis, ces defs seront ignorées par Python)
def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def mape_safe(y_true, y_pred, eps=1e-8):
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)

# --- Vérifs rapides
assert target_col in y_train.columns and target_col in y_test.columns, "target_col absent."
assert isinstance(y_train.index, pd.DatetimeIndex) and isinstance(y_test.index, pd.DatetimeIndex)

# --- Fit SES sur TRAIN uniquement (alpha optimisé automatiquement)
endog_train = y_train[target_col].astype(float).copy()
endog_train = endog_train.replace([np.inf, -np.inf], np.nan).dropna()

model = SimpleExpSmoothing(endog_train, initialization_method="estimated")
fit = model.fit(optimized=True)  # smoothing_level (alpha) optimisé via likelihood

alpha = getattr(fit, "model", None)
alpha = fit.params.get("smoothing_level", np.nan)
print(f"[fit] SES optimized alpha ≈ {alpha:.6f}")

# --- Prévision out-of-sample sur tout le TEST (multi-step)
steps = len(y_test)
y_fcst = fit.forecast(steps=steps)
y_pred_test = pd.Series(y_fcst.values, index=y_test.index, name="y_pred")

# --- Alignement & métriques
pred_ok = pd.DataFrame({
    "y_true": y_test[target_col],
    "y_pred": y_pred_test
}).dropna()

rmse_val = rmse(pred_ok["y_true"], pred_ok["y_pred"])
mae_val  = mae(pred_ok["y_true"],  pred_ok["y_pred"])
mape_val = mape_safe(pred_ok["y_true"], pred_ok["y_pred"])

print(f"[metrics] SES → RMSE={rmse_val:.6f} | MAE={mae_val:.6f} | MAPE%≈{mape_val:.2f}")

# --- Sauvegardes CSV
pred_path  = os.path.join(RESULTS_BASE, "ses_predictions.csv")
resid_path = os.path.join(RESULTS_BASE, "ses_residuals.csv")
pred_ok.to_csv(pred_path)
(pred_ok["y_true"] - pred_ok["y_pred"]).to_frame("residual").to_csv(resid_path)
print(f"[save] Predictions → {pred_path}")
print(f"[save] Residuals   → {resid_path}")

# --- Figure: y_true vs y_pred (test)
fig_path = os.path.join(FIGS_DIR, "ses_test_vs_pred.png")
plt.show()
plt.figure(figsize=(10,4))
plt.plot(pred_ok.index, pred_ok["y_true"], label="Test (y_true)")
plt.plot(pred_ok.index, pred_ok["y_pred"], label="SES forecast")
plt.title("Simple Exponential Smoothing — Test vs Prediction")
plt.xlabel("Date"); plt.ylabel("Log-returns")
plt.legend()
plt.tight_layout()
plt.savefig(fig_path, dpi=150)
# plt.show()  # décommente si tu veux l'afficher dans Spyder
plt.close()
print(f"[save] Figure → {fig_path}")

# --- Leaderboard (nécessite la version patchée de save_leaderboard)
row = {
    "model": "Baseline_SES",
    "part": "4.3",
    "train_start": y_train.index.min().strftime("%Y-%m-%d"),
    "train_end":   y_train.index.max().strftime("%Y-%m-%d"),
    "test_start":  y_test.index.min().strftime("%Y-%m-%d"),
    "test_end":    y_test.index.max().strftime("%Y-%m-%d"),
    "test_rmse": rmse_val,
    "test_mae":  mae_val,
    "test_mape": mape_val,
    "notes": f"SimpleExpSmoothing (alpha optimized); fitted on train only; {steps} out-of-sample steps."
}
save_leaderboard(row, models_dir=os.path.join(PROJECT_ROOT,"models"), data_proc_dir=DATA_PROC)

banner("4.3 - SES complete")

# =========================================================
# 4.4 - Comparaison & Résultats (Baselines)
# =========================================================
banner("4.4 - Comparaison & Résultats")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# --- Helpers (au cas où)
def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def mape_safe(y_true, y_pred, eps=1e-8):
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)

# --- Chemins utiles
LB_CANDIDATES = [
    Path(PROJECT_ROOT) / "models" / "leaderboard.csv",
    Path(DATA_PROC) / "leaderboard.csv",
]
RES = Path(RESULTS_BASE)
FIGS = Path(FIGS_DIR)

# --- 1) Charger métriques depuis leaderboard (si dispo), sinon recalculer depuis CSV prédictions
def load_leaderboard_first():
    for p in LB_CANDIDATES:
        if p.exists():
            try:
                df = pd.read_csv(p)
                print(f"[load] Leaderboard found → {p}")
                return df
            except Exception:
                pass
    print("[info] Leaderboard not found or unreadable — will compute metrics from prediction CSVs.")
    return None

lb = load_leaderboard_first()

# Modèles attendus et fichiers de prédictions (pour fallback et pour résidus)
model_files = {
    "Baseline_Naive": RES / "naive_predictions.csv",
    "Baseline_MA_k5": RES / "ma_k5_predictions.csv",
    "Baseline_MA_k10": RES / "ma_k10_predictions.csv",
    "Baseline_MA_k20": RES / "ma_k20_predictions.csv",
    "Baseline_SES": RES / "ses_predictions.csv",
}
resid_files = {
    "Baseline_Naive": RES / "naive_residuals.csv",
    "Baseline_MA_k5": RES / "ma_k5_residuals.csv",
    "Baseline_MA_k10": RES / "ma_k10_residuals.csv",
    "Baseline_MA_k20": RES / "ma_k20_residuals.csv",
    "Baseline_SES": RES / "ses_residuals.csv",
}

# --- 2) Construire le tableau des métriques
rows = []
for model, pred_path in model_files.items():
    rmse_val = mae_val = mape_val = np.nan
    if lb is not None and "model" in lb.columns:
        # Priorité: lire du leaderboard si la ligne existe
        row_lb = lb.loc[lb["model"] == model]
        if len(row_lb) > 0:
            rmse_val = float(row_lb["test_rmse"].iloc[-1]) if "test_rmse" in row_lb.columns else np.nan
            mae_val  = float(row_lb["test_mae"].iloc[-1])  if "test_mae"  in row_lb.columns else np.nan
            mape_val = float(row_lb["test_mape"].iloc[-1]) if "test_mape" in row_lb.columns else np.nan

    # Fallback: recalc depuis CSV des prédictions
    if (np.isnan(rmse_val) or np.isnan(mae_val)) and pred_path.exists():
        dfp = pd.read_csv(pred_path, index_col=0)
        rmse_val = rmse(dfp["y_true"].values, dfp["y_pred"].values)
        mae_val  = mae(dfp["y_true"].values, dfp["y_pred"].values)
        mape_val = mape_safe(dfp["y_true"].values, dfp["y_pred"].values)

    rows.append({
        "model": model,
        "rmse": rmse_val,
        "mae": mae_val,
        "mape_pct": mape_val
    })

df_summary = pd.DataFrame(rows)
# Ordonner de façon lisible
order = ["Baseline_Naive", "Baseline_MA_k5", "Baseline_MA_k10", "Baseline_MA_k20", "Baseline_SES"]
df_summary["order"] = df_summary["model"].apply(lambda m: order.index(m) if m in order else 999)
df_summary = df_summary.sort_values("order").drop(columns="order")

# Sauvegarde tableau récap
summary_csv = RES / "part4_summary_metrics.csv"
df_summary.to_csv(summary_csv, index=False)
print(f"[save] Summary table → {summary_csv}")

# --- 3) Bar chart des RMSE (Naïve vs MA vs SES)
fig_rmse = FIGS / "part4_rmse_comparison.png"
plt.figure(figsize=(8,4))
plt.bar(df_summary["model"], df_summary["rmse"])
plt.title("Baselines — RMSE comparison")
plt.ylabel("RMSE"); plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(fig_rmse, dpi=150)
# plt.show()
plt.close()
print(f"[save] RMSE bar chart → {fig_rmse}")

# --- 4) Distributions des erreurs (histogrammes superposés)
# Choisir 2-3 modèles représentatifs pour la lisibilité
compare_models = ["Baseline_Naive", "Baseline_MA_k20", "Baseline_SES"]
colors = ["tab:gray", "tab:blue", "tab:green"]

fig_err = FIGS / "part4_error_distributions.png"
plt.figure(figsize=(8,4))
for model, col in zip(compare_models, colors):
    rf = resid_files.get(model)
    if rf is not None and rf.exists():
        r = pd.read_csv(rf, index_col=0)["residual"].dropna().values
        # hist normalisé (densité)
        plt.hist(r, bins=60, density=True, alpha=0.45, label=model, color=col)
    else:
        print(f"[warn] Residual file missing for {model}: {rf}")
plt.title("Residual distributions (Naïve vs MA(20) vs SES)")
plt.xlabel("Residual"); plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.savefig(fig_err, dpi=150)
# plt.show()
plt.close()
print(f"[save] Error distributions → {fig_err}")

banner("4.4 - Comparaison terminée")
print(df_summary)

