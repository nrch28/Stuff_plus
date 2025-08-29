from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import zscore

from pybaseball import statcast
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss
from lightgbm import LGBMRegressor, plot_importance

from joblib import dump


# ──────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────

# Used All 2024 Data
YEARS = [2024]
CACHE_DIR = Path("cache"); CACHE_DIR.mkdir(exist_ok=True)

VALID_PITCHES = [
    "CH","CU","FC","FF","FO","FS","FT","GY","KC","KN","SL","ST","SC","SI"
]

RAW = [
    "release_speed",
    "release_spin_rate",
    "release_extension",
    "pfx_x",
    "pfx_z",
    "release_pos_x",
    "release_pos_z",
    "vx0",
    "vy0",
    "vz0",
    "ax",
    "ay"
]
ENGR = ["spin_axis_sin","spin_axis_cos","velo_diff","ivb_diff","hb_diff"]
FEATS = RAW + ENGR

COMP = [
    "hit_into_play","foul","ball","swinging_strike","called_strike",
    "foul_tip","blocked_ball","swinging_strike_blocked","hit_by_pitch",
    "foul_bunt","missed_bunt","bunt_foul_tip"
]
CSW = ["called_strike","swinging_strike","swinging_strike_blocked"]


# ──────────────────────────────────────────────────────────────────────────
# Data download with cache
# ──────────────────────────────────────────────────────────────────────────
def _load_year(y: int) -> pd.DataFrame:
    f = CACHE_DIR / f"statcast_{y}.csv"
    if f.exists():
        return pd.read_csv(f)
    df = statcast(start_dt=f"{y}-01-01", end_dt=f"{y}-12-31")
    df.to_csv(f, index=False); return df


def load_data() -> pd.DataFrame:
    df = pd.concat([_load_year(y) for y in YEARS], ignore_index=True)
    df = df[df["pitch_type"].isin(VALID_PITCHES)].copy()
    df["is_competitive"] = df["description"].isin(COMP).astype(int)
    df["is_CSW"] = df["description"].isin(CSW).astype(int)
    return df


# ──────────────────────────────────────────────────────────────────────────
# Engineered features
# ──────────────────────────────────────────────────────────────────────────
def engineer(df: pd.DataFrame) -> pd.DataFrame:
    df["spin_axis"] = pd.to_numeric(df["spin_axis"], errors="coerce").astype(float)
    df["spin_axis"] = (
        df.groupby("pitch_type")["spin_axis"]
          .transform(lambda s: s.fillna(s.mean()))
          .fillna(180.0)
    )

    # break spin axis in to sin and cos
    # important because of circular nature of spin axis
    # for example, 359° and 1° should be the same distance as 1° and 3°
    df["spin_axis_sin"] = np.sin(np.radians(df["spin_axis"]))
    df["spin_axis_cos"] = np.cos(np.radians(df["spin_axis"]))

 
    # find primary (most frequent) fastball type 
    fb_mode = (
        df[df["pitch_type"].isin(['FF', 'SI', 'FC', 'FT'])]
        .groupby("pitcher")["pitch_type"]
        .agg(lambda s: s.value_counts().idxmax())
        .rename("primary_fb")
    )
    df = df.join(fb_mode, on="pitcher")

    # compute velo and break means of primary fastball
    fb_stats = (
        df[df["pitch_type"] == df["primary_fb"]]
        .groupby("pitcher")
        .agg(fb_velo=("release_speed","mean"),
             fb_ivb=("pfx_z","mean"),
             fb_hb=("pfx_x","mean"))
    )
    df = df.join(fb_stats, on="pitcher")

    # calculate differences from primary fastball means 
    df["velo_diff"] = df["release_speed"] - df["fb_velo"]
    df["ivb_diff"]  = df["pfx_z"]         - df["fb_ivb"]
    df["hb_diff"]   = df["pfx_x"]         - df["fb_hb"]
    return df

# ──────────────────────────────────────────────────────────────────────────
def main():
    df = engineer(load_data())
    mask = (df["is_competitive"] == 1) & df[RAW].notnull().all(axis=1)
    data = df[mask].copy()

    X, y = data[FEATS], data["is_CSW"].astype(float)

    # Train/test split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, random_state=44, shuffle=True
    )

     # Scale before model
    scaler = RobustScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_te_scaled = scaler.transform(X_te)

    # Save scaler parameters to a format R can use
    import json
    scaler_params = {
        'center_': scaler.center_.tolist(),
        'scale_': scaler.scale_.tolist(),
        'quantile_range': scaler.quantile_range,
        'with_centering': scaler.with_centering,
        'with_scaling': scaler.with_scaling
    }
    with open('robust_scaler_params.json', 'w') as f:
        json.dump(scaler_params, f)
    
    # 4. Put back in DataFrames to preserve feature names
    X_tr_scaled = pd.DataFrame(X_tr_scaled, columns=X_tr.columns)
    X_te_scaled = pd.DataFrame(X_te_scaled, columns=X_te.columns)

    dump(scaler, 'robust_scaler.joblib')

    # 5. Define LightGBM model 
    model = LGBMRegressor(
        n_estimators=1000, 
        learning_rate=0.01,
        num_leaves=31, 
        max_depth=-1, 
        min_child_samples=20,
        subsample=0.8, 
        colsample_bytree=0.8,
        reg_alpha=0.1, 
        reg_lambda=0.2,
        random_state=42, 
        force_row_wise=True,
        n_jobs=-1, 
        objective="regression"
    )

    # 6. Fit model
    model.fit(X_tr_scaled, y_tr)

     # 7. Predict on test set
    y_pred = np.clip(model.predict(X_te_scaled), 0, 1)
    print(f"AUC-ROC : {roc_auc_score(y_te, y_pred):.4f}")
    print(f"Log-loss: {log_loss(y_te, y_pred):.4f}")

    # 8. Predict on all data
    X_scaled = scaler.transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    data["pred_CSW_prob"] = np.clip(model.predict(X_scaled), 0, 1)

    # 9. Stuff+ calculation
    data["Stuff+"] = 100 + 10 * zscore(data["pred_CSW_prob"])

    tbl = (
        data.groupby(["pitcher","player_name","pitch_type"])
            .agg(pitches=("is_competitive","sum"),
                 CSW_events=("is_CSW","sum"),
                 actual_CSW_rate=("is_CSW","mean"),
                 pred_CSW_rate=("pred_CSW_prob","mean"),
                 stuff_plus=("Stuff+","mean"))
            .query("pitches >= 50")
            .sort_values("stuff_plus", ascending=False)
    )
    tbl.to_csv("stuff_plus_ratings_full.csv")

    # 10. Save model
    model.booster_.save_model("lightboost_stuff_model.txt")

    # 11. Print top-10 lines
    print(tbl.sort_values("actual_CSW_rate", ascending=False)[
        ["pitches","CSW_events","actual_CSW_rate","pred_CSW_rate","stuff_plus"]
    ].head(10))

    # 12. Plots
    plt.figure(figsize=(10,6))
    plt.hist(tbl["stuff_plus"], bins=30, edgecolor="k")
    plt.axvline(100, color="red", ls="--")
    plt.xlabel("Stuff+ (100 = league avg)")
    plt.ylabel("Pitcher–pitch types")
    plt.title("Stuff+ distribution (2024)")
    plt.show()

    ax = plot_importance(model, importance_type="gain", max_num_features=15)
    ax.set_title("LightGBM feature importance (gain)")
    plt.show()


if __name__ == "__main__":
    main() 