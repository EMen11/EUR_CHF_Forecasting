
import sys
import statsmodels
 


import os
import pandas as pd
import yfinance as yf

import numpy as np
import matplotlib.pyplot as plt



# --- 1.6 / Bloc A : imports + dossiers ---




# Utilise le dossier de travail courant de Spyder comme racine du projet
PROJECT_ROOT = os.getcwd()

# Dossiers data
DATA_RAW = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROC = os.path.join(PROJECT_ROOT, "data", "processed")
os.makedirs(DATA_RAW, exist_ok=True)
os.makedirs(DATA_PROC, exist_ok=True)

print("PROJECT_ROOT:", PROJECT_ROOT)
print("DATA_RAW:", DATA_RAW)
print("DATA_PROC:", DATA_PROC)

# --- 1.6 / Bloc B : téléchargement des données brutes EUR/CHF ---

# Paramètres
TICKER = "EURCHF=X"
START = "2015-01-01"
END = None  # jusqu'à aujourd'hui

# Téléchargement avec yfinance
df = yf.download(
    TICKER,
    start=START,
    end=END,
    interval="1d",
    auto_adjust=True,
    progress=False
)

print("Shape:", df.shape)
print(df.head(3))
print(df.tail(3))

# --- 1.6 / Bloc B-fix : normalisation des colonnes ---

# Certains retours de yfinance utilisent des colonnes MultiIndex (field, ticker).
# On gère les deux cas (avec ou sans MultiIndex).

if isinstance(df.columns, pd.MultiIndex):
    # Cas MultiIndex -> on sélectionne les colonnes du ticker, puis on aplatit.
    # On met le ticker en premier niveau pour sélectionner proprement.
    df = df.swaplevel(axis=1)           # niveau 0: ticker, niveau 1: field
    df = df[TICKER]                     # on garde uniquement notre ticker
    # À ce stade, colonnes: ['Open','High','Low','Close','Adj Close','Volume'] (simple Index)
    # Certaines installations renvoient 'Adj Close' au lieu de 'Close' pertinent :
    if 'Close' not in df.columns and 'Adj Close' in df.columns:
        df = df.rename(columns={'Adj Close': 'Close'})
else:
    # Cas simple -> parfois la colonne utile est 'Adj Close'
    if 'Close' not in df.columns and 'Adj Close' in df.columns:
        df = df.rename(columns={'Adj Close': 'Close'})

print("Colonnes après normalisation:", list(df.columns))



# --- 1.6 / Bloc C : nettoyage calendrier + sauvegarde ---

# 1) On garde le prix de clôture et on renomme
df = df[['Close']].rename(columns={'Close': 'price'})
df.index.name = "date"

# 2) Calendrier jours ouvrés + propagation vers l'avant
df = df.asfreq("B")
df['price'] = df['price'].ffill()

# 3) Nettoyages de base
df = df.sort_index()
df = df[~df.index.duplicated(keep='first')]
df = df.dropna(subset=['price'])

# 4) Sauvegarde
parquet_path = os.path.join(DATA_RAW, "eur_chf_daily.parquet")
csv_path     = os.path.join(DATA_RAW, "eur_chf_daily.csv")

df.to_parquet(parquet_path)
df.to_csv(csv_path)

# 5) Résumé rapide
summary = {
    "rows": int(len(df)),
    "start": str(df.index.min().date()),
    "end": str(df.index.max().date()),
    "price_min": float(df['price'].min()),
    "price_max": float(df['price'].max()),
    "csv": os.path.abspath(csv_path),
    "parquet": os.path.abspath(parquet_path),
}
print("SUMMARY:", summary)

print("\nHEAD:")
print(df.head(3))
print("\nTAIL:")
print(df.tail(3))

# --- 1.6 / Bloc D : contrôles qualité + mini plot ---



# 0) Dossier reports (au cas où)
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

# 1) Vérifs de base
print("\n=== QA CHECKS ===")
print("Index type:", type(df.index))
print("Is DatetimeIndex:", isinstance(df.index, pd.DatetimeIndex))
print("Monotonic increasing:", df.index.is_monotonic_increasing)
print("Has duplicates in index:", bool(df.index.duplicated().sum()))
print("NA in price:", int(df['price'].isna().sum()))

# 2) Vérif calendrier business days
expected = pd.date_range(df.index.min(), df.index.max(), freq="B")
missing_from_df = expected.difference(df.index)
print("Business days expected:", len(expected))
print("Business days present :", len(df))
print("Missing B-days after asfreq (should be 0):", len(missing_from_df))

# 3) Stats descriptives rapides
desc = df['price'].describe()
print("\n=== PRICE DESCRIBE ===")
print(desc)

# 4) Graphe simple + export PNG
plt.figure(figsize=(10, 4))
plt.plot(df.index, df['price'])
plt.title("EUR/CHF – Daily Close (B-calendar, ffill)")
plt.xlabel("Date")
plt.ylabel("Price")
plt.tight_layout()

png_path = os.path.join(REPORTS_DIR, "eur_chf_price.png")
plt.savefig(png_path, dpi=150)
print("\nFigure saved to:", os.path.abspath(png_path))
plt.show()

# --- 1.7 / Bloc E : Quick quality checks ---

# A) Vérification de la monotonie de l’index
is_monotonic = df.index.is_monotonic_increasing
print("Monotonic index:", is_monotonic)

# B) Vérification des jours ouvrés manquants
expected = pd.date_range(df.index.min(), df.index.max(), freq="B")
missing_days = expected.difference(df.index)
print("Missing business days:", len(missing_days))

# C) Aperçu des premières et dernières lignes
print("\nHEAD:")
print(df.head(3))
print("\nTAIL:")
print(df.tail(3))

# --- 1.8 / Bloc V0 : helper générique pour télécharger & sauvegarder un actif ---



# Par sécurité si START n'existe pas encore (au cas où tu lances V0 seul)
try:
    START
except NameError:
    START = "2015-01-01"

# Assure-toi que les dossiers existent (au cas où)
if 'PROJECT_ROOT' not in globals():
    PROJECT_ROOT = os.getcwd()
DATA_RAW = os.path.join(PROJECT_ROOT, "data", "raw")
os.makedirs(DATA_RAW, exist_ok=True)

def download_asset(ticker: str, start: str, name: str, interval: str = "1d"):
    """
    Télécharge un actif via yfinance, gère les colonnes MultiIndex,
    garde le Close (ou Adj Close), réindexe en jours ouvrés (B) + ffill,
    sauvegarde en CSV & Parquet dans data/raw, et renvoie le DataFrame.
    """
    print(f"\n--- Downloading {ticker} → {name} ---")
    df = yf.download(
        ticker,
        start=start,
        end=None,
        interval=interval,
        auto_adjust=True,
        progress=False
    )

    if df.empty:
        raise ValueError(f"Aucune donnée renvoyée pour {ticker}. Vérifie le ticker/intervalle.")

    # Normalisation colonnes (cas MultiIndex vs simple)
    if isinstance(df.columns, pd.MultiIndex):
        df = df.swaplevel(axis=1)   # (ticker, field) -> (field, ticker) → puis sélection du ticker
        df = df[ticker]
        if 'Close' not in df.columns and 'Adj Close' in df.columns:
            df = df.rename(columns={'Adj Close': 'Close'})
    else:
        if 'Close' not in df.columns and 'Adj Close' in df.columns:
            df = df.rename(columns={'Adj Close': 'Close'})

    # On garde le prix de clôture
    df = df[['Close']].rename(columns={'Close': 'price'})
    df.index.name = "date"

    # Calendrier jours ouvrés + ffill
    df = df.asfreq("B")
    df['price'] = df['price'].ffill()

    # Nettoyages
    df = df.sort_index()
    df = df[~df.index.duplicated(keep='first')]
    df = df.dropna(subset=['price'])

    # Sauvegarde
    parquet_path = os.path.join(DATA_RAW, f"{name}.parquet")
    csv_path     = os.path.join(DATA_RAW, f"{name}.csv")
    df.to_parquet(parquet_path)
    df.to_csv(csv_path)

    # Résumé
    summary = {
        "name": name,
        "rows": int(len(df)),
        "start": str(df.index.min().date()),
        "end": str(df.index.max().date()),
        "price_min": float(df['price'].min()),
        "price_max": float(df['price'].max()),
        "csv": os.path.abspath(csv_path),
        "parquet": os.path.abspath(parquet_path),
    }
    print("SUMMARY:", summary)
    print("\nHEAD:\n", df.head(3))
    print("\nTAIL:\n", df.tail(3))
    return df

# --- 1.8 / Bloc V1 : téléchargement S&P 500 (^GSPC) ---

sp500 = download_asset("^GSPC", START, "sp500")

# --- 1.8 / Bloc V2 : téléchargement Apple (AAPL) ---

aapl = download_asset("AAPL", START, "apple")

# --- 1.8 / Bloc V3 : téléchargement CPI mensuel (CPIAUCSL, FRED) ---

cpi_proxy = download_asset("TIP", START, "cpi_proxy", interval="1d")

# --- 1.9 / Bloc F : résumé comparatif des datasets ---



# Dossiers
DATA_RAW = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROC = os.path.join(PROJECT_ROOT, "data", "processed")
os.makedirs(DATA_PROC, exist_ok=True)

# Spécification des fichiers attendus : {nom lisible: (ticker, nom_fichier)}
assets = {
    "EUR/CHF (FX)":        ("EURCHF=X", "eur_chf_daily.csv"),
    "S&P 500 (Index)":     ("^GSPC",     "sp500.csv"),
    "Apple (AAPL)":        ("AAPL",      "apple.csv"),
    "Inflation proxy (TIP)": ("TIP",     "cpi_proxy.csv"),
}

def summarize_csv(name, ticker, filename):
    path = os.path.join(DATA_RAW, filename)
    if not os.path.exists(path):
        return {
            "Name": name, "Ticker": ticker, "File": filename, "Status": "MISSING",
            "Rows": 0, "Start": None, "End": None, "Min Price": None, "Max Price": None, "Frequency": None,
        }
    # Lecture
    df_ = pd.read_csv(path, parse_dates=["date"], index_col="date")
    df_ = df_.sort_index()
    # Fréquence inférée (peut renvoyer None selon les données, on fallback sur 'B' si quotidien business)
    freq = pd.infer_freq(df_.index)
    # Résumé
    out = {
        "Name": name,
        "Ticker": ticker,
        "File": filename,
        "Status": "OK",
        "Rows": int(len(df_)),
        "Start": str(df_.index.min().date()) if len(df_) else None,
        "End": str(df_.index.max().date()) if len(df_) else None,
        "Min Price": float(df_["price"].min()) if "price" in df_.columns and len(df_) else None,
        "Max Price": float(df_["price"].max()) if "price" in df_.columns and len(df_) else None,
        "Frequency": freq if freq is not None else "B"  # nos séries sont en business days
    }
    return out

# Construire le tableau
rows = []
for name, (ticker, filename) in assets.items():
    rows.append(summarize_csv(name, ticker, filename))

summary_df = pd.DataFrame(rows, columns=[
    "Name","Ticker","File","Status","Rows","Start","End","Min Price","Max Price","Frequency"
]).sort_values("Name")

print("\n=== DATASETS SUMMARY ===")
print(summary_df)

# Sauvegarde
summary_csv = os.path.join(DATA_PROC, "datasets_summary.csv")
summary_parquet = os.path.join(DATA_PROC, "datasets_summary.parquet")
summary_df.to_csv(summary_csv, index=False)
summary_df.to_parquet(summary_parquet, index=False)
print("\nSaved summary to:")
print(" -", os.path.abspath(summary_csv))
print(" -", os.path.abspath(summary_parquet))





# Lister tous les fichiers CSV dans  data/raw et data/processed
folders = [DATA_RAW, DATA_PROC]

for folder in folders:
    print(f"\n--- Fichiers dans {folder} ---")
    if os.path.exists(folder):
        for file in os.listdir(folder):
            if file.endswith(".csv"):
                print(os.path.abspath(os.path.join(folder, file)))
    else:
        print("⚠️ Dossier introuvable :", folder)

