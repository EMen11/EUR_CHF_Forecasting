# Part2_1_load.py — Chargement & QA de la série EUR/CHF (prix)


 # --- Imports principaux ---
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Stats & ML ---
from scipy import stats
from sklearn.preprocessing import StandardScaler

# --- 0) Dossiers
PROJECT_ROOT = os.getcwd()
DATA_RAW = os.path.join(PROJECT_ROOT, "data", "raw")
RAW_FILE = os.path.join(DATA_RAW, "eur_chf_daily.csv")

def banner(title: str):
    print("\n" + "=" * 8 + f" {title} " + "=" * 8)

def read_eurchf_csv(path: str) -> pd.DataFrame:
    """Lecture robuste:
    - détecte la colonne date (date/Date/Unnamed: 0)
    - aplati un MultiIndex éventuel
    - garantit une colonne finale 'price' (float) + DatetimeIndex trié
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Fichier introuvable: {path}\n"
            f"Vérifie le Working Directory: {PROJECT_ROOT}\n"
            f"Contenu de data/raw: {os.listdir(DATA_RAW) if os.path.isdir(DATA_RAW) else 'dossier manquant'}"
        )

    # 1) lecture simple
    df = pd.read_csv(path)

    # 2) repérage colonne date
    date_col = None
    for cand in ["date", "Date", "DATE", "index", "Index", "Unnamed: 0"]:
        if cand in df.columns:
            date_col = cand
            break
    if date_col is None:
        # tente le 1er champ comme date si convertible
        first_col = df.columns[0]
        try:
            pd.to_datetime(df[first_col], errors="raise")
            date_col = first_col
        except Exception:
            pass
    if date_col is None:
        raise KeyError(f"Aucune colonne temporelle trouvée dans {list(df.columns)}")

    # 3) conversion en datetime + index
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    if df[date_col].isna().any():
        raise ValueError("Certaines dates sont invalides/NaT. Vérifie le CSV.")
    df = df.set_index(date_col).sort_index()
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # 4) aplatir un éventuel MultiIndex de colonnes
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join([str(x) for x in tup if str(x) != ""]).strip().lower()
            for tup in df.columns.values
        ]
    else:
        df.columns = [c.strip().lower() for c in df.columns]

    # 5) standardiser le nom de la colonne prix
    rename_map = {
        "adj close": "price",
        "adj_close": "price",
        "close": "price",
        "price_price": "price",   # cas d'un MultiIndex aplati
        "prix": "price",
    }
    for k, v in rename_map.items():
        if k in df.columns:
            df = df.rename(columns={k: v})

    if "price" not in df.columns:
        # Heuristique: garder la 1re colonne numérique != volume
        num_cols = [c for c in df.columns if c != "volume" and pd.api.types.is_numeric_dtype(df[c])]
        if len(num_cols) == 0:
            raise ValueError(f"Impossible d'identifier la colonne de prix parmi {list(df.columns)}")
        df = df.rename(columns={num_cols[0]: "price"})

    # 6) ne garder que price, forcer float
    df = df[["price"]].copy()
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    # 7) enlever doublons éventuels sur l'index
    if df.index.has_duplicates:
        df = df[~df.index.duplicated(keep="first")]

    return df

def quick_checks(df: pd.DataFrame) -> None:
    banner("INFO DE BASE")
    print(f"Rows               : {len(df):,}")
    print(f"Date range         : {df.index.min().date()} → {df.index.max().date()}")
    print(f"Index monotonic    : {df.index.is_monotonic_increasing}")
    print(f"Duplicate dates    : {df.index.duplicated().sum()}")

    banner("NA & VALIDITÉ")
    na_count = int(df["price"].isna().sum())
    nonpos = int((df["price"] <= 0).sum())
    print(f"NA in price        : {na_count}")
    print(f"Non-positive prices: {nonpos}")

    banner("CALENDRIER JOURS OUVRÉS")
    bdays = pd.bdate_range(df.index.min(), df.index.max(), freq="B")
    missing = bdays.difference(df.index)
    print(f"Business days expected: {len(bdays):,}")
    print(f"Business days present : {df.index.size:,}")
    print(f"Missing business days : {len(missing):,}")
    if len(missing) > 0:
        # On n'affiche que les 5 premiers pour info
        print("Exemples dates manquantes:", [d.date() for d in list(missing)[:5]])

    banner("DESCRIPTIF PRIX")
    print(df["price"].describe())

    banner("APERÇU")
    print("HEAD:\n", df.head(3))
    print("\nTAIL:\n", df.tail(3))

if __name__ == "__main__":
    banner("CHEMINS")
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("DATA_RAW    :", DATA_RAW)
    print("RAW_FILE    :", RAW_FILE)

    df = read_eurchf_csv(RAW_FILE)
    quick_checks(df)

    banner("RÉSUMÉ")
    print("Chargement OK. DataFrame disponible en mémoire sous le nom 'df' (colonnes: ['price']).")
    
  

# 1. Calcul des rendements log
df["log_return"] = np.log(df["price"] / df["price"].shift(1))
df.dropna(inplace=True)  # enlève la première ligne NaN

# 2. Analyse descriptive
print("Descriptif log-returns")
print(df["log_return"].describe())
print("Skewness :", df["log_return"].skew())
print("Kurtosis :", df["log_return"].kurt())

# 3. Détection d’outliers (z-score > 3)
threshold = 3
mean = df["log_return"].mean()
std = df["log_return"].std()
outliers_z = df[np.abs((df["log_return"] - mean) / std) > threshold]

# 4. Détection d’outliers (IQR)
Q1 = df["log_return"].quantile(0.25)
Q3 = df["log_return"].quantile(0.75)
IQR = Q3 - Q1
outliers_iqr = df[(df["log_return"] < Q1 - 1.5*IQR) | (df["log_return"] > Q3 + 1.5*IQR)]

print("Outliers (Z-score):", len(outliers_z))
print("Outliers (IQR):", len(outliers_iqr))

# ============================================
# Part 2.3 - Log-Returns: Distribution, Normality & Visualization
# ============================================


# --- Paths (extend)
DATA_PROC = os.path.join(PROJECT_ROOT, "data", "processed")
REPORTS   = os.path.join(PROJECT_ROOT, "reports")
os.makedirs(DATA_PROC, exist_ok=True)
os.makedirs(REPORTS, exist_ok=True)

banner("2.3 - LOG-RETURNS: STATS, NORMALITY, VISUALS")

# 1) (Re)compute log-returns to be safe and drop first NaN
if "log_return" not in df.columns:
    df["log_return"] = np.log(df["price"]) - np.log(df["price"].shift(1))
df = df.dropna(subset=["log_return"]).copy()

# 2) Descriptive statistics
lr = df["log_return"]
mean_, std_ = lr.mean(), lr.std(ddof=1)
skew_, kurt_ = lr.skew(), lr.kurt()

print("== Log-Returns Descriptive Stats ==")
print(lr.describe())
print(f"Skewness : {skew_:.6f}")
print(f"Kurtosis : {kurt_:.6f}")

# 3) Outliers recap (Z-score & IQR) - recompute here for the report
z_threshold = 3.0
zscore = (lr - mean_) / std_
outliers_z_idx = zscore.abs() > z_threshold
outliers_z_count = int(outliers_z_idx.sum())

Q1, Q3 = lr.quantile(0.25), lr.quantile(0.75)
IQR = Q3 - Q1
low_bound, high_bound = Q1 - 1.5*IQR, Q3 + 1.5*IQR
outliers_iqr_idx = (lr < low_bound) | (lr > high_bound)
outliers_iqr_count = int(outliers_iqr_idx.sum())

print(f"Outliers (Z-score > {z_threshold}): {outliers_z_count}")
print(f"Outliers (IQR rule)              : {outliers_iqr_count}")

# 4) Normality test (Shapiro-Wilk) and QQ-plot
#    Shapiro is OK up to a few thousand samples (here ~2.8k)
shapiro_stat, shapiro_p = stats.shapiro(lr.values)
print("== Shapiro-Wilk Test ==")
print(f"W statistic: {shapiro_stat:.6f} | p-value: {shapiro_p:.6g}")

# 5) Visualizations
# 5.a Histogram + normal density overlay
fig_hist, ax_hist = plt.subplots(figsize=(8, 5))
# histogram (density)
ax_hist.hist(lr, bins=50, density=True, alpha=0.6)
# normal pdf with same mean/std
x = np.linspace(mean_ - 5*std_, mean_ + 5*std_, 400)
pdf = stats.norm.pdf(x, loc=mean_, scale=std_)
ax_hist.plot(x, pdf, linewidth=2)
ax_hist.set_title("EUR/CHF Log-Returns - Histogram & Normal Density")
ax_hist.set_xlabel("log_return")
ax_hist.set_ylabel("Density")
hist_path = os.path.join(REPORTS, "eur_chf_logret_hist_density.png")
fig_hist.tight_layout()
fig_hist.savefig(hist_path, dpi=150)
plt.close(fig_hist)

# 5.b Boxplot
fig_box, ax_box = plt.subplots(figsize=(6, 5))
ax_box.boxplot(lr, vert=True, showfliers=True)
ax_box.set_title("EUR/CHF Log-Returns - Boxplot")
ax_box.set_ylabel("log_return")
box_path = os.path.join(REPORTS, "eur_chf_logret_boxplot.png")
fig_box.tight_layout()
fig_box.savefig(box_path, dpi=150)
plt.close(fig_box)

# 5.c QQ-plot vs Normal
fig_qq = plt.figure(figsize=(6, 6))
stats.probplot(lr, dist="norm", plot=plt)
plt.title("EUR/CHF Log-Returns - QQ Plot (Normal)")
qq_path = os.path.join(REPORTS, "eur_chf_logret_qqplot.png")
plt.tight_layout()
plt.savefig(qq_path, dpi=150)
plt.close(fig_qq)

# 6) Save processed returns & summary
returns_out_csv    = os.path.join(DATA_PROC, "eur_chf_returns.csv")
returns_out_parquet= os.path.join(DATA_PROC, "eur_chf_returns.parquet")
summary_txt        = os.path.join(REPORTS,   "eur_chf_logret_summary.txt")

# Save DataFrame with price + log_return
df.to_csv(returns_out_csv, index=True)
try:
    df.to_parquet(returns_out_parquet, index=True)
except Exception:
    # parquet optional
    pass

with open(summary_txt, "w") as f:
    f.write("EUR/CHF Log-Returns Summary (Part 2.3)\n")
    f.write("-" * 40 + "\n")
    f.write(f"Rows                 : {len(lr):,}\n")
    f.write(f"Date range           : {df.index.min().date()} -> {df.index.max().date()}\n")
    f.write(f"Mean                 : {mean_:.8f}\n")
    f.write(f"Std (ddof=1)         : {std_:.8f}\n")
    f.write(f"Skewness             : {skew_:.6f}\n")
    f.write(f"Kurtosis             : {kurt_:.6f}\n")
    f.write(f"Outliers (Z>{z_threshold}) : {outliers_z_count}\n")
    f.write(f"Outliers (IQR rule)       : {outliers_iqr_count}\n")
    f.write(f"Shapiro-Wilk W       : {shapiro_stat:.6f}\n")
    f.write(f"Shapiro-Wilk p-value : {shapiro_p:.6g}\n")
    f.write("\nSaved figures:\n")
    f.write(f" - Histogram+density : {hist_path}\n")
    f.write(f" - Boxplot           : {box_path}\n")
    f.write(f" - QQ-plot           : {qq_path}\n")
    f.write("\nSaved datasets:\n")
    f.write(f" - CSV     : {returns_out_csv}\n")
    f.write(f" - Parquet : {returns_out_parquet}\n")

banner("2.3 DONE")
print("Figures saved to:")
print(" -", hist_path)
print(" -", box_path)
print(" -", qq_path)
print("Processed returns saved to:")
print(" -", returns_out_csv)
print(" -", returns_out_parquet)



# ============================================
# Part 2.4 - Preliminary Analysis of Returns
# ============================================

banner("2.4 - PRELIMINARY ANALYSIS OF RETURNS")

# 1) NA check after shift
na_price = int(df["price"].isna().sum())
na_lr = int(df["log_return"].isna().sum())
print(f"NA in price     : {na_price}")
print(f"NA in log_return: {na_lr}")
if na_lr > 0:
    print("Dropping remaining NaN in log_return...")
    df = df.dropna(subset=["log_return"]).copy()
    na_lr = int(df["log_return"].isna().sum())
    print(f"NA in log_return after drop: {na_lr}")

# 2) Descriptive statistics
print("\n== Descriptive statistics (log_return) ==")
desc = df["log_return"].describe()
print(desc)
print(f"Skewness : {df['log_return'].skew():.6f}")
print(f"Kurtosis : {df['log_return'].kurt():.6f}")

# 3) Time series plot - returns
fig_ret, ax_ret = plt.subplots(figsize=(10, 4))
ax_ret.plot(df.index, df["log_return"], linewidth=1)
ax_ret.axhline(0.0, linestyle="--", linewidth=1)
ax_ret.set_title("EUR/CHF Log-Returns - Time Series")
ax_ret.set_xlabel("Date")
ax_ret.set_ylabel("log_return")
ret_ts_path = os.path.join(REPORTS, "eur_chf_logret_timeseries.png")
fig_ret.tight_layout()
fig_ret.savefig(ret_ts_path, dpi=150)
plt.close(fig_ret)

# 4) Price vs Returns - side-by-side style (rebased price for visual scale)
price_rebased = df["price"] / df["price"].iloc[0]

fig_cmp, (ax_p, ax_r) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
ax_p.plot(df.index, price_rebased, linewidth=1.2)
ax_p.set_title("EUR/CHF Price (Rebased to 1 at start)")
ax_p.set_ylabel("Rebased price")

ax_r.plot(df.index, df["log_return"], linewidth=1)
ax_r.axhline(0.0, linestyle="--", linewidth=1)
ax_r.set_title("EUR/CHF Log-Returns")
ax_r.set_xlabel("Date")
ax_r.set_ylabel("log_return")

cmp_ts_path = os.path.join(REPORTS, "eur_chf_price_vs_returns.png")
fig_cmp.tight_layout()
fig_cmp.savefig(cmp_ts_path, dpi=150)
plt.close(fig_cmp)

# 5) (Optional) Quick rolling stats on returns to eyeball volatility stability
#    (comment out if you prefer pure minimal plots)
win = 21  # ~1 trading month
roll_mean = df["log_return"].rolling(win).mean()
roll_std  = df["log_return"].rolling(win).std()

fig_roll, ax_roll = plt.subplots(figsize=(10, 4))
ax_roll.plot(df.index, roll_mean, linewidth=1.2, label=f"Rolling mean ({win})")
ax_roll.plot(df.index, roll_std,  linewidth=1.2, label=f"Rolling std ({win})")
ax_roll.axhline(0.0, linestyle="--", linewidth=1)
ax_roll.set_title(f"Rolling Stats on Log-Returns (window={win})")
ax_roll.set_xlabel("Date")
ax_roll.set_ylabel("Value")
ax_roll.legend()
roll_ts_path = os.path.join(REPORTS, "eur_chf_logret_rolling_stats.png")
fig_roll.tight_layout()
fig_roll.savefig(roll_ts_path, dpi=150)
plt.close(fig_roll)

# 6) Console summary
print("\nFigures saved to:")
print(" -", ret_ts_path)
print(" -", cmp_ts_path)
print(" -", roll_ts_path)

# ==========================================================
# 2.5 / Train-Test Split (chronological, avoid data leakage)
# ==========================================================

banner("# 2.5 / Train-Test Split (chronological, avoid data leakage)")

def load_returns_robust(csv_path: str) -> pd.DataFrame:
    """Read processed returns CSV robustly:
    - accepts 'Date' or 'date' column as datetime index
    - or uses first column as datetime index if unnamed
    """
    # 1) peek columns first
    peek = pd.read_csv(csv_path, nrows=5)
    cols = [c.strip() for c in peek.columns.tolist()]

    if "Date" in cols:
        df_ = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date")
    elif "date" in cols:
        df_ = pd.read_csv(csv_path, parse_dates=["date"], index_col="date")
    else:
        # assume first column is the datetime index (unnamed)
        df_ = pd.read_csv(csv_path, parse_dates=[0], index_col=0)
        # optionnel : renommer l’index
        df_.index.name = "date"

    # tri + contrôle basique
    df_ = df_.sort_index()
    if df_.index.tz is not None:
        df_.index = df_.index.tz_localize(None)
    return df_

# Re-load processed returns (robust)
file_path = os.path.join(DATA_PROC, "eur_chf_returns.csv")
df = load_returns_robust(file_path)

# Define chronological split (adjust dates if needed)
train = df.loc["2015-01-01":"2022-12-31"].copy()
test  = df.loc["2023-01-01":"2025-12-31"].copy()  # up to end of data anyway

# Check sizes and proportions
print("Train shape:", train.shape)
print("Test shape :", test.shape)
prop_train = len(train) / len(df)
prop_test  = len(test) / len(df)
print(f"Train proportion: {prop_train:.2%}")
print(f"Test proportion : {prop_test:.2%}")

# Save to processed folder
train_csv = os.path.join(DATA_PROC, "eur_chf_train.csv")
test_csv  = os.path.join(DATA_PROC, "eur_chf_test.csv")
train.to_csv(train_csv)
test.to_csv(test_csv)

# Optional: Parquet
try:
    train.to_parquet(os.path.join(DATA_PROC, "eur_chf_train.parquet"))
    test.to_parquet(os.path.join(DATA_PROC, "eur_chf_test.parquet"))
except Exception:
    pass

# Final check of periods (no overlap)
print("Train period:", train.index.min().date(), "→", train.index.max().date())
print("Test period :", test.index.min().date(),  "→", test.index.max().date())

# ==========================================================
# 2.6 / ML Feature Prep: lags, rolling stats, scaling (fit on train)
# ==========================================================


banner("2.6 - ML FEATURE PREP & SCALING")

# -- 0) Chargement des données returns (version complète) pour construire les features
#     On reconstruit d'abord les features sur TOUT l'historique,
#     puis on découpe selon les bornes train/test déjà utilisées.
returns_csv = os.path.join(DATA_PROC, "eur_chf_returns.csv")

def load_returns_robust(csv_path: str) -> pd.DataFrame:
    peek = pd.read_csv(csv_path, nrows=5)
    cols = [c.strip() for c in peek.columns.tolist()]
    if "Date" in cols:
        df_ = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date")
    elif "date" in cols:
        df_ = pd.read_csv(csv_path, parse_dates=["date"], index_col="date")
    else:
        df_ = pd.read_csv(csv_path, parse_dates=[0], index_col=0)
        df_.index.name = "date"
    df_ = df_.sort_index()
    if df_.index.tz is not None:
        df_.index = df_.index.tz_localize(None)
    return df_

df_all = load_returns_robust(returns_csv)

# -- 1) Définir les fenêtres et lags
lag_list = [1, 5, 10, 21]        # lags de rendements (1 jour, 1 semaine, ~2 semaines, ~1 mois)
roll_windows = [5, 21, 63]       # ~1 semaine, ~1 mois, ~1 trimestre (jours ouvrés)

# -- 2) Construction des features (SANS fuite: uniquement infos passées)
#    Cible: return du lendemain -> y = log_return.shift(-1)
feat = pd.DataFrame(index=df_all.index)
feat["r_t"] = df_all["log_return"]                               # rendement du jour t (disponible pour prédire t+1)
for L in lag_list:
    feat[f"lag_{L}"] = df_all["log_return"].shift(L)             # rendements passés

for W in roll_windows:
    # stats roulantes calculées jusqu'à t (donc utilisables pour prédire t+1)
    feat[f"roll_mean_{W}"] = df_all["log_return"].rolling(W).mean()
    feat[f"roll_std_{W}"]  = df_all["log_return"].rolling(W).std(ddof=1)

# Cible: rendement du lendemain
feat["target_next_return"] = df_all["log_return"].shift(-1)

# Drop des lignes incomplètes (début pour lags/rolling, fin pour target)
feat = feat.dropna().copy()

# -- 3) Découpage train/test avec les mêmes bornes qu'en 2.5
#     On relit les CSV train/test pour récupérer les bornes exactes
train_csv = os.path.join(DATA_PROC, "eur_chf_train.csv")
test_csv  = os.path.join(DATA_PROC, "eur_chf_test.csv")
train_idx = pd.read_csv(train_csv, parse_dates=[0], index_col=0).index
test_idx  = pd.read_csv(test_csv,  parse_dates=[0], index_col=0).index

train_start, train_end = train_idx.min(), train_idx.max()
test_start,  test_end  = test_idx.min(),  test_idx.max()

feat_train = feat.loc[train_start:train_end].copy()
feat_test  = feat.loc[test_start:test_end].copy()

# Il peut y avoir quelques lignes en moins au début du train (fenêtres roulantes):
# on affiche les tailles finales
print("Features train shape:", feat_train.shape)
print("Features test  shape:", feat_test.shape)

# -- 4) Séparation X / y
target_col = "target_next_return"
X_train = feat_train.drop(columns=[target_col])
y_train = feat_train[target_col]
X_test  = feat_test.drop(columns=[target_col])
y_test  = feat_test[target_col]

# -- 5) Standardisation (Z-score) : FIT sur TRAIN, APPLY sur TEST (anti-leakage)
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    index=X_train.index, columns=X_train.columns
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    index=X_test.index, columns=X_test.columns
)

# -- 6) Sauvegardes
# Version "brute" (non-scalée) all-in-one
features_all_csv = os.path.join(DATA_PROC, "eur_chf_features.csv")
feat.to_csv(features_all_csv)

# Versions train/test (non-scalées)
feat_train.to_csv(os.path.join(DATA_PROC, "eur_chf_train_features.csv"))
feat_test.to_csv(os.path.join(DATA_PROC, "eur_chf_test_features.csv"))

# Versions train/test SCALÉES (X uniquement, y séparé)
X_train_scaled.to_csv(os.path.join(DATA_PROC, "eur_chf_train_X_scaled.csv"))
X_test_scaled.to_csv(os.path.join(DATA_PROC, "eur_chf_test_X_scaled.csv"))
y_train.to_csv(os.path.join(DATA_PROC, "eur_chf_train_y.csv"))
y_test.to_csv(os.path.join(DATA_PROC, "eur_chf_test_y.csv"))

# -- 7) Petit résumé console
print("\n=== Feature Columns ===")
print(list(X_train.columns))
print("\nTrain period:", X_train.index.min().date(), "→", X_train.index.max().date())
print("Test  period:", X_test.index.min().date(),  "→", X_test.index.max().date())
print("\nSaved:")
print(" -", features_all_csv)
print(" -", os.path.join(DATA_PROC, "eur_chf_train_features.csv"))
print(" -", os.path.join(DATA_PROC, "eur_chf_test_features.csv"))
print(" -", os.path.join(DATA_PROC, "eur_chf_train_X_scaled.csv"))
print(" -", os.path.join(DATA_PROC, "eur_chf_test_X_scaled.csv"))
print(" -", os.path.join(DATA_PROC, "eur_chf_train_y.csv"))
print(" -", os.path.join(DATA_PROC, "eur_chf_test_y.csv"))

# ================================================================
# 2.7 - Documentation & Sauvegardes 
# ================================================================


REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

banner("2.7 - DOC & SAVES")

# --- A) S'assurer que df a bien 'r_t' ---
# Priorité: utiliser 'r_t' si présent; sinon copier depuis 'log_return';
# sinon le recalculer depuis 'price'.
if "r_t" not in df.columns:
    if "log_return" in df.columns:
        df["r_t"] = df["log_return"]
    else:
        df["r_t"] = np.log(df["price"]) - np.log(df["price"].shift(1))
        df = df.dropna(subset=["r_t"]).copy()

# Même traitement pour train/test (peuvent venir de 2.5 sans 'r_t')
def ensure_rt(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "r_t" not in out.columns:
        if "log_return" in out.columns:
            out["r_t"] = out["log_return"]
        elif "price" in out.columns:
            out["r_t"] = np.log(out["price"]) - np.log(out["price"].shift(1))
        out = out.dropna(subset=["r_t"])
    return out

train = ensure_rt(train)
test  = ensure_rt(test)

# --- B) Sauvegardes datasets principaux ---
returns_save_path = os.path.join(DATA_PROC, "eur_chf_returns.csv")
df[["price", "r_t"]].to_csv(returns_save_path, index=True)

train_save_path = os.path.join(DATA_PROC, "eur_chf_train.csv")
test_save_path  = os.path.join(DATA_PROC, "eur_chf_test.csv")
train.to_csv(train_save_path, index=True)
test.to_csv(test_save_path, index=True)

# Sauvegarde des features complètes:
# - si l'objet 'feat' existe (créé en 2.6), on l'utilise
# - sinon on recharge depuis le fichier produit en 2.6
features_save_path = os.path.join(DATA_PROC, "eur_chf_features.csv")
if "feat" in globals():
    feat.to_csv(features_save_path, index=True)
else:
    feat_reloaded = pd.read_csv(features_save_path) if os.path.exists(features_save_path) else None
    if feat_reloaded is None:
        print("⚠️ Features file not found; skip saving features (ensure 2.6 ran).")

print("Datasets saved:")
print(" -", returns_save_path)
print(" -", train_save_path)
print(" -", test_save_path)
print(" -", features_save_path, "(if available)")

# --- C) Résumé statistique ---
summary_path = os.path.join(REPORTS_DIR, "eur_chf_returns_summary.txt")
with open(summary_path, "w") as f:
    f.write("=== EUR/CHF Returns Summary ===\n\n")
    f.write(f"Period: {df.index.min().date()} → {df.index.max().date()}\n")
    f.write(f"Observations: {df['r_t'].dropna().shape[0]}\n\n")
    f.write(df["r_t"].describe().to_string())
    f.write("\n\nSkewness: {:.6f}".format(df["r_t"].skew()))
    f.write("\nKurtosis: {:.6f}".format(df["r_t"].kurt()))
print("Stat summary saved:", summary_path)

# --- D) Graphiques ---
# (a) Histogramme des rendements
plt.figure(figsize=(8,5))
sns.histplot(df["r_t"].dropna(), bins=100, kde=True)
plt.title("Histogram of EUR/CHF Log Returns")
plt.xlabel("Log return")
plt.ylabel("Frequency")
plt.axvline(0, linestyle="--", alpha=0.7)
hist_path = os.path.join(FIGURES_DIR, "eur_chf_returns_hist.png")
plt.tight_layout(); plt.savefig(hist_path, dpi=300); plt.close()

# (b) Série temporelle des rendements
plt.figure(figsize=(12,5))
plt.plot(df.index, df["r_t"], linewidth=0.8)
plt.axhline(0, linestyle="--", alpha=0.7)
plt.title("EUR/CHF Log Returns - Time Series")
plt.xlabel("Date"); plt.ylabel("Log return")
ts_path = os.path.join(FIGURES_DIR, "eur_chf_returns_timeseries.png")
plt.tight_layout(); plt.savefig(ts_path, dpi=300); plt.close()

# (c) Distribution Train vs Test (densités normalisées)
plt.figure(figsize=(8,5))
sns.histplot(train["r_t"].dropna(), bins=80, kde=True, stat="density", alpha=0.6, label="Train")
sns.histplot(test["r_t"].dropna(),  bins=80, kde=True, stat="density", alpha=0.6, label="Test")
plt.title("Distribution of Returns - Train vs Test")
plt.xlabel("Log return"); plt.ylabel("Density"); plt.legend()
dist_path = os.path.join(FIGURES_DIR, "eur_chf_train_vs_test_dist.png")
plt.tight_layout(); plt.savefig(dist_path, dpi=300); plt.close()


print("Figures saved:")
print(" -", hist_path)
print(" -", ts_path)
print(" -", dist_path)



