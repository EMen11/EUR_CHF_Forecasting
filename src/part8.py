# =========================================================
# Part 8 — Evaluation (fast & clean)
# 8.0 Setup  | 8.1 Summary table | 8.2 Figures (3 plots)
# =========================================================
import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Paths ----------
PROJECT_ROOT = "/Users/eliemenassa/Desktop/Projet Forecasting"
DATA_PROC    = os.path.join(PROJECT_ROOT, "data", "processed")
RES_FINAL    = os.path.join(PROJECT_ROOT, "results", "final")
RES_ML_ADV   = os.path.join(PROJECT_ROOT, "results", "ml_advanced")
RES_BASE     = os.path.join(PROJECT_ROOT, "results", "baseline")
RES_STATS    = os.path.join(PROJECT_ROOT, "results", "stats")
RES_ML_BASE  = os.path.join(PROJECT_ROOT, "results", "ml_baseline")

FIG_DIR = os.path.join(RES_FINAL, "figs")
os.makedirs(RES_FINAL, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# ---------- Helpers ----------
def _safe_read_pred(path):
    """Read a predictions CSV and return a DF indexed by date with a y_pred column.
       Accepts a variety of shapes: [date,y_pred], [date,y_true,y_pred], or single-column y_pred with date index."""
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    # normalize date index
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).set_index("date").sort_index()
    else:
        # maybe first column is the index
        if df.columns[0].lower().startswith("unnamed"):
            df = df.set_index(df.columns[0])
        try:
            df.index = pd.to_datetime(df.index, errors="coerce")
            df = df.sort_index()
        except Exception:
            pass

    # find y_pred column
    colmap = {c.lower(): c for c in df.columns}
    ypred = None
    for key in ("y_pred","yhat","pred","prediction","preds"):
        if key in colmap:
            ypred = colmap[key]; break
    if ypred is None:
        # if only one numeric column, assume it's y_pred
        numcols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if len(numcols) == 1:
            ypred = numcols[0]
        else:
            # if the file contains y_true & y_pred in unexpected names, try fallback
            for c in df.columns:
                if "pred" in c.lower():
                    ypred = c; break
    if ypred is None:
        warnings.warn(f"[8.x] Could not find y_pred in {path} → skipping")
        return None

    out = df[[ypred]].rename(columns={ypred:"y_pred"})
    return out

def _read_y_test():
    """Load test target with dates. Tries eur_chf_test_y.csv, then eur_chf_test.csv."""
    # Preferred: dedicated y file
    y_path = os.path.join(DATA_PROC, "eur_chf_test_y.csv")
    if os.path.exists(y_path):
        ydf = pd.read_csv(y_path)
        if "date" in ydf.columns:
            ydf["date"] = pd.to_datetime(ydf["date"], errors="coerce")
            ydf = ydf.dropna(subset=["date"]).set_index("date").sort_index()
            # pick the first numeric column as target
            numcols = [c for c in ydf.columns if pd.api.types.is_numeric_dtype(ydf[c])]
            if not numcols:
                raise ValueError("No numeric target column found in eur_chf_test_y.csv")
            tgt = numcols[0]
            return ydf[[tgt]].rename(columns={tgt:"y_true"})
        else:
            # fallback to the general test file for the dates
            x_path = os.path.join(DATA_PROC, "eur_chf_test.csv")
            xdf = pd.read_csv(x_path, parse_dates=["date"]).set_index("date").sort_index()
            # guess target column name in ydf (first numeric)
            numcols = [c for c in ydf.columns if pd.api.types.is_numeric_dtype(ydf[c])]
            if not numcols:
                raise ValueError("No numeric target column found in eur_chf_test_y.csv")
            tgt = numcols[0]
            ydf.index = xdf.index[:len(ydf)]  # align length if needed
            return ydf[[tgt]].rename(columns={tgt:"y_true"}).sort_index()

    # Fallback: take it from eur_chf_test.csv (assumes 'log_return' or 'target' exists)
    x_path = os.path.join(DATA_PROC, "eur_chf_test.csv")
    xdf = pd.read_csv(x_path, parse_dates=["date"]).set_index("date").sort_index()
    cand = None
    for c in ("log_return","target","y","y_true","r_t"):
        if c in xdf.columns:
            cand = c; break
    if cand is None:
        raise ValueError("No suitable target column found in eur_chf_test.csv")
    return xdf[[cand]].rename(columns={cand:"y_true"})

def _rmse_mae(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rmse = float(np.sqrt(np.mean((y_true - y_pred)**2)))
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    # MAPE non-pertinent mais on le calcule pour cohérence
    eps = 1e-12
    mape = float(np.mean(np.abs((y_true+eps)-(y_pred+eps)) / (np.abs(y_true)+eps)))
    return rmse, mae, mape

# ---------- 8.1 — Summary table ----------
# Base: reuse the Part 7.7 summary (déjà consolidé)
p77 = os.path.join(RES_ML_ADV, "part7_summary_metrics.csv")
if not os.path.exists(p77):
    raise FileNotFoundError(f"Missing {p77}. Run Part 7.7 first.")
df7 = pd.read_csv(p77)

# Optionally append baselines from Part 4 if available (Naive/MA)
p4sum = os.path.join(PROJECT_ROOT, "results", "baseline", "part4_summary_metrics.csv")
extra_rows = []
if os.path.exists(p4sum):
    df4 = pd.read_csv(p4sum)
    # keep key baselines to illustrate families
    keep = ["Baseline_Naive","Baseline_MA_k20"]
    for k in keep:
        r = df4[df4["model"]==k]
        if len(r):
            rmse, mae, mape = float(r["rmse"].iloc[0]), float(r["mae"].iloc[0]), float(r["mape_pct"].iloc[0])
            extra_rows.append({"label": k.replace("_"," "), "rmse": rmse, "mae": mae, "mape_pct": mape})

df8 = pd.concat([
    df7[["label","rmse","mae","mape_pct"]],
    pd.DataFrame(extra_rows)
], ignore_index=True)

# Sort & save final summary
df8 = df8.sort_values("rmse", ascending=True).reset_index(drop=True)
out_csv = os.path.join(RES_FINAL, "part8_summary_metrics.csv")
df8.to_csv(out_csv, index=False)
print(f"[8.1] Summary saved → {out_csv}")

# ---------- 8.2 — Figures ----------
# (a) RMSE bar chart (all)
plt.figure(figsize=(10, 6))
ypos = np.arange(len(df8))
plt.barh(ypos, df8["rmse"])
plt.yticks(ypos, df8["label"])
plt.gca().invert_yaxis()
best_rmse = df8["rmse"].min()
for i, (lab, v) in enumerate(zip(df8["label"], df8["rmse"])):
    gap = 100.0*(v - best_rmse)/best_rmse
    plt.text(v, i, f"  {v:.6f}  (+{gap:.2f}%)", va="center")
plt.title("Part 8 — RMSE comparison (lower is better)")
plt.xlabel("RMSE")
plt.tight_layout()
fig1 = os.path.join(FIG_DIR, "part8_rmse_comparison.png")
plt.savefig(fig1, dpi=150); plt.close()
print(f"[8.2] Figure saved → {fig1}")

# (b) Overlay truth vs top-3 models
# Map labels -> prediction file paths
pred_map = {
    "Baseline_SES (Part 4)": os.path.join(RES_BASE, "ses_predictions.csv"),
    "HW_SES (Part 5)":       os.path.join(RES_STATS, "hw_ses_predictions.csv"),
    "ARIMA (Part 5)":        os.path.join(RES_STATS, "arima_predictions.csv"),
    "STACK (Part 7)":        os.path.join(RES_ML_ADV, "preds", "part7_stack_test_predictions.csv"),
    "RF (Part 7)":           os.path.join(RES_ML_ADV, "preds", "part7_rf_test_predictions.csv"),
    "GBDT (Part 7)":         os.path.join(RES_ML_ADV, "preds", "part7_gbdt_test_predictions.csv"),
    "XGB (Part 7)":          os.path.join(RES_ML_ADV, "preds", "part7_xgb_test_predictions.csv"),
    "LGBM (Part 7)":         os.path.join(RES_ML_ADV, "preds", "part7_lgbm_test_predictions.csv"),
    "Linear (Part 6)":       os.path.join(PROJECT_ROOT, "results", "ml_baseline", "part6_linear_predictions.csv"),
    "BLEND (Part 7)":        os.path.join(RES_ML_ADV, "preds", "part7_blend_test_predictions.csv"),
    # Optional older baselines:
    "Baseline Naive":        os.path.join(PROJECT_ROOT, "results", "baseline", "naive_predictions.csv"),
    "Baseline MA k20":       os.path.join(PROJECT_ROOT, "results", "baseline", "ma_k20_predictions.csv"),
}

# pick top-3 from df8
top3 = df8.head(3)["label"].tolist()

# load y_true
yte = _read_y_test()  # index=date, col=y_true

# build overlay
plt.figure(figsize=(12, 4))
plt.plot(yte.index, yte["y_true"], label="y_true", linewidth=1)

resid_for_plot = {}
for lab in top3:
    pth = pred_map.get(lab)
    if pth is None or not os.path.exists(pth):
        warnings.warn(f"[8.2] Missing predictions for {lab} ({pth}) → skipping overlay")
        continue
    pred = _safe_read_pred(pth)
    if pred is None: 
        continue
    dfj = yte.join(pred, how="inner")
    if dfj.empty:
        warnings.warn(f"[8.2] Empty join for {lab} → skipping")
        continue
    plt.plot(dfj.index, dfj["y_pred"], label=lab, linewidth=1)
    resid_for_plot[lab] = (dfj["y_true"] - dfj["y_pred"]).values

plt.title("Part 8 — Truth vs Top-3 forecasts")
plt.legend(ncol=2)
plt.tight_layout()
fig2 = os.path.join(FIG_DIR, "part8_top3_vs_truth.png")
plt.savefig(fig2, dpi=150); plt.close()
print(f"[8.2] Figure saved → {fig2}")

# (c) Residual distributions (top-3)
if resid_for_plot:
    plt.figure(figsize=(12, 3.6))
    n = len(resid_for_plot)
    for i, (lab, res) in enumerate(resid_for_plot.items(), start=1):
        plt.subplot(1, n, i)
        plt.hist(res, bins=50)
        plt.title(lab)
        plt.xlabel("Residual"); plt.ylabel("Count")
    plt.suptitle("Part 8 — Residual distributions (Top-3)")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig3 = os.path.join(FIG_DIR, "part8_residual_distributions.png")
    plt.savefig(fig3, dpi=150); plt.close()
    print(f"[8.2] Figure saved → {fig3}")
else:
    print("[8.2] Skipped residual distributions (no overlapping preds)")

print("[Part 8] Done.")
