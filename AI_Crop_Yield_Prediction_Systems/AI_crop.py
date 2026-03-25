#!/usr/bin/env python
# ============================================================
# Yield Prediction Using NDVI + SENTINEL-1 + DUAL MODEL
# v2.2 – Dual-Model Architecture
#
#   Model 1 — Plot-Level Yield Model  (training/validation)
#     Features: NDVI_fused_auc, NDVI_fused_mean,
#               VV_mean, VH_mean,
#               rainfall, soil_organic_carbon, clay_content, temp
#
#   Model 2 — Within-Field Variability Model  (spatial yield maps)
#     Features: NDVI_fused_mean, NDVI_fused_auc, VV_mean, VH_mean
#     (spatially varying bands only – no ENV constants)
#
#   - Optimized XGBoost hyperparameters for better generalization
#   - SCL cloud masking, Kalman S2+S1 fusion,
#     nested SFS within Model 1 candidate set
# ============================================================

import os
import sys
import warnings
import numpy as np
import pandas as pd
import rasterio
import ee
import geemap
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from contextlib import contextmanager

from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

from xgboost import XGBRegressor

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ============================================================
# CONTEXT MANAGER – suppress verbose geemap/GEE logging
# ============================================================

@contextmanager
def suppress_output():
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        yield
    finally:
        sys.stdout.close()
        sys.stdout = original_stdout


# ============================================================
# GOOGLE EARTH ENGINE INITIALIZATION
# ============================================================

PROJECT_ID = "quiet-subset-447718-q0"

try:
    ee.Initialize(project=PROJECT_ID)
except Exception:
    ee.Authenticate()
    ee.Initialize(project=PROJECT_ID)

print("✅ Google Earth Engine initialized")


# ============================================================
# PATHS
# ============================================================

BASE_DIRECTORY   = os.path.dirname(os.path.abspath(__file__))
AOI_SHP          = os.path.join(BASE_DIRECTORY, "Farm",       "witz_farm.shp")
PLOT_SHP         = os.path.join(BASE_DIRECTORY, "GGE_vector", "GGE_Harvest_150_gcs.shp")
DATA_FILE_PATH   = os.path.join(BASE_DIRECTORY, "plot_satellite_indices_cloud_robust.csv")
OUTPUT_DIRECTORY = os.path.join(BASE_DIRECTORY, "output")
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)


# ============================================================
# LOAD AOI & PLOTS
# ============================================================

AOI = geemap.shp_to_ee(AOI_SHP).geometry()

# STRICT AOI MASK (forces identical raster footprint)
AOI_MASK = ee.Image.constant(1).clip(AOI)

print("✅ AOI loaded with strict geometry mask")


# ============================================================
# TIME RANGE & GROWTH STAGES
# ============================================================

START_DATE = "2025-03-01"
END_DATE   = "2025-08-10"

GROWTH_STAGES = {
    "early": ("2025-03-01", "2025-04-30"),
    "mid":   ("2025-05-01", "2025-06-30"),
    "late":  ("2025-07-01", "2025-08-10"),
}
STAGE_ORDER = ["early", "mid", "late"]


# ============================================================
# CLOUD MASKING (SCL)
# ============================================================

CLEAR_SCL_CLASSES = [4, 5, 6]

def apply_scl_cloud_mask(img):
    scl = img.select("SCL")
    clear_mask = (
        scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6))
    )
    return img.updateMask(clear_mask)

def add_ndvi_cloud_masked(img):
    masked = apply_scl_cloud_mask(img)
    ndvi = masked.normalizedDifference(["B8", "B4"]).rename("NDVI")
    return masked.addBands(ndvi)

s2_full = (
    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterBounds(AOI)
    .filterDate(START_DATE, END_DATE)
    .map(add_ndvi_cloud_masked)
)

# ============================================================
# SENTINEL-1
# ============================================================

def add_rvi(img):
    vv = img.select("VV")
    vh = img.select("VH")
    rvi = vh.multiply(4).divide(vv.add(vh)).rename("RVI")
    return img.addBands(rvi)

s1 = (
    ee.ImageCollection("COPERNICUS/S1_GRD")
    .filterBounds(AOI)
    .filterDate(START_DATE, END_DATE)
    .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
    .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
    .map(add_rvi)
)


# ============================================================
# KALMAN FUSION (AOI CLIPPED)
# ============================================================

S1_SPECKLE_VAR = 0.05 ** 2

def kalman_fuse_ndvi(s2_ndvi_median, s1_collection, start, end):
    s2_window = (
        s2_full.filterDate(start, end)
               .select("NDVI")
    )
    s2_var = s2_window.reduce(ee.Reducer.variance()).rename("s2_var").clip(AOI)
    s2_var_safe = s2_var.where(s2_var.lte(0), 9999).unmask(9999)
    s1_rvi = (
        s1_collection.filterDate(start, end)
        .select("RVI")
        .median()
        .clip(AOI)
    )
    s1_ndvi = s1_rvi.unitScale(0, 1).rename("NDVI")
    s1_var = ee.Image.constant(S1_SPECKLE_VAR)
    denom = s1_var.add(s2_var_safe)
    w_s2 = s1_var.divide(denom)
    w_s1 = s2_var_safe.divide(denom)
    blended = (
        w_s2.multiply(s2_ndvi_median)
            .add(w_s1.multiply(s1_ndvi))
    ).rename("NDVI")
    # ✅ STRICT – no artificial filling
    fused = blended
    return fused.clip(AOI).updateMask(AOI_MASK)

# ============================================================
# STAGE NDVI GENERATION
# ============================================================

stage_s2_only_ndvi = {}
stage_fused_ndvi   = {}

for stage, (start, end) in GROWTH_STAGES.items():

    s2_ndvi = (
        s2_full.filterDate(start, end)
               .select("NDVI")
               .median()
               .clip(AOI)
               .updateMask(AOI_MASK)
    )

    stage_s2_only_ndvi[stage] = s2_ndvi
    stage_fused_ndvi[stage] = kalman_fuse_ndvi(s2_ndvi, s1, start, end)

print("✅ NDVI generation complete with strict AOI geometry")

# ============================================================
# EXPORT NDVI MAPS
# ============================================================

for stage in STAGE_ORDER:
    s2_img    = stage_s2_only_ndvi[stage].clip(AOI).updateMask(AOI_MASK).unmask(-9999)
    fused_img = stage_fused_ndvi[stage].clip(AOI).updateMask(AOI_MASK).unmask(-9999)

    with suppress_output():
        geemap.ee_export_image(
            s2_img,
            os.path.join(OUTPUT_DIRECTORY, f"NDVI_S2_ONLY_{stage.upper()}_stage.tif"),
            scale=10,
            region=AOI,
            file_per_band=False,
        )

        geemap.ee_export_image(
            fused_img,
            os.path.join(OUTPUT_DIRECTORY, f"NDVI_FUSED_S2_S1_{stage.upper()}_stage.tif"),
            scale=10,
            region=AOI,
            file_per_band=False,
        )

print("✅ NDVI exports complete with identical AOI footprint")


# ============================================================
# FARM POLYGON CLIP – FUSED vs S2-ONLY COMPARISON MAPS
# ============================================================

FARM_GEOM = geemap.shp_to_ee(AOI_SHP).geometry()

print("\n🌾 Exporting farm-boundary-clipped comparison maps (Fused vs S2-only)...")

for stage in STAGE_ORDER:

    fused_farm = (
        stage_fused_ndvi[stage]
        .clip(FARM_GEOM)
        .updateMask(ee.Image.constant(1).clip(FARM_GEOM))
        .unmask(-9999)
    )

    s2_farm = (
        stage_s2_only_ndvi[stage]
        .clip(FARM_GEOM)
        .updateMask(ee.Image.constant(1).clip(FARM_GEOM))
        .unmask(-9999)
    )

    with suppress_output():
        geemap.ee_export_image(
            fused_farm,
            os.path.join(OUTPUT_DIRECTORY,
                         f"FARM_NDVI_FUSED_S2_S1_{stage.upper()}_stage.tif"),
            scale=10,
            region=FARM_GEOM,
            file_per_band=False,
        )
        geemap.ee_export_image(
            s2_farm,
            os.path.join(OUTPUT_DIRECTORY,
                         f"FARM_NDVI_S2_ONLY_{stage.upper()}_stage.tif"),
            scale=10,
            region=FARM_GEOM,
            file_per_band=False,
        )

    print(f"   ✓ {stage.upper()}: FARM_NDVI_FUSED_S2_S1 + FARM_NDVI_S2_ONLY exported")

print("✅ Farm-clipped comparison maps exported")


# ============================================================
# VISUALISE FARM COMPARISON MAPS (all stages in one figure)
# ============================================================

print("\n🖼️  Generating farm NDVI comparison figure (Fused vs S2-only)...")

all_files_exist = True
for stage in STAGE_ORDER:
    fused_tif = os.path.join(OUTPUT_DIRECTORY,
                             f"FARM_NDVI_FUSED_S2_S1_{stage.upper()}_stage.tif")
    s2_tif    = os.path.join(OUTPUT_DIRECTORY,
                             f"FARM_NDVI_S2_ONLY_{stage.upper()}_stage.tif")
    if not (os.path.exists(fused_tif) and os.path.exists(s2_tif)):
        all_files_exist = False
        break

if all_files_exist:
    fig, axes = plt.subplots(3, 2, figsize=(14, 18))

    all_ndvi_vals = []
    stage_arrays = {}

    for stage in STAGE_ORDER:
        fused_tif = os.path.join(OUTPUT_DIRECTORY,
                                 f"FARM_NDVI_FUSED_S2_S1_{stage.upper()}_stage.tif")
        s2_tif    = os.path.join(OUTPUT_DIRECTORY,
                                 f"FARM_NDVI_S2_ONLY_{stage.upper()}_stage.tif")

        with rasterio.open(fused_tif) as src:
            fused_arr = src.read(1).astype(np.float32)
            fused_arr[fused_arr == src.nodata] = np.nan

        with rasterio.open(s2_tif) as src:
            s2_arr = src.read(1).astype(np.float32)
            s2_arr[s2_arr == src.nodata] = np.nan

        stage_arrays[stage] = {'fused': fused_arr, 's2': s2_arr}

        all_ndvi_vals.extend(fused_arr[np.isfinite(fused_arr)].ravel())
        all_ndvi_vals.extend(s2_arr[np.isfinite(s2_arr)].ravel())

    if len(all_ndvi_vals) > 0:
        vmin = float(np.percentile(all_ndvi_vals, 2))
        vmax = float(np.percentile(all_ndvi_vals, 98))
    else:
        vmin, vmax = 0, 1

    cmap = plt.cm.RdYlGn

    for idx, stage in enumerate(STAGE_ORDER):
        s2_arr    = stage_arrays[stage]['s2']
        fused_arr = stage_arrays[stage]['fused']

        im0 = axes[idx, 0].imshow(s2_arr, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
        axes[idx, 0].set_title(f"S2-Only NDVI – {stage.capitalize()} Stage",
                               fontsize=12, fontweight='bold')
        axes[idx, 0].axis('off')

        im1 = axes[idx, 1].imshow(fused_arr, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
        axes[idx, 1].set_title(f"Fused S2+S1 NDVI – {stage.capitalize()} Stage",
                               fontsize=12, fontweight='bold')
        axes[idx, 1].axis('off')

        plt.colorbar(im0, ax=axes[idx, 0], fraction=0.046, pad=0.04, label="NDVI")
        plt.colorbar(im1, ax=axes[idx, 1], fraction=0.046, pad=0.04, label="NDVI")

    fig.suptitle("Farm NDVI Comparison: S2-Only vs Kalman-Fused S2+S1\n"
                 "All Growth Stages (Common Color Scale)",
                 fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()

    combined_fig_path = os.path.join(OUTPUT_DIRECTORY,
                                     "FARM_NDVI_Comparison_ALL_STAGES.png")
    plt.savefig(combined_fig_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"✅ Combined NDVI comparison figure saved → {combined_fig_path}")
    plt.close()
else:
    print("⚠️  Some NDVI export files not found, skipping combined comparison figure.")

print("✅ Farm comparison figures complete")


# ============================================================
# LOAD TABULAR DATA
# ============================================================

print("\n📊 Loading tabular training data...")
data = pd.read_csv(DATA_FILE_PATH)

TARGET   = "dryYieldg"
LOCATION = "location"
STAGE    = "growth_stage"

# ============================================================
# MODEL 1 — Plot-Level Yield Model
#   Used for: training, cross-validation, predicted-vs-observed plots,
#             uncertainty estimation, and CSV confidence table.
#   8 features: 2 VI + 2 S1 + 4 ENV
# ============================================================

M1_VI_COLUMNS  = ["NDVI_fused_auc"]
M1_S1_COLUMNS  = ["VV_mean", "VH_mean"]
M1_ENV_COLUMNS = ["rainfall", "soil_organic_carbon", "clay_content", "temp"]

M1_FEATURE_COLUMNS = M1_VI_COLUMNS + M1_S1_COLUMNS + M1_ENV_COLUMNS

# Aliases used throughout the training loop (keeps variable names short)
VI_COLUMNS  = M1_VI_COLUMNS
S1_COLUMNS  = M1_S1_COLUMNS
ENV_COLUMNS = M1_ENV_COLUMNS
FEATURE_COLUMNS = M1_FEATURE_COLUMNS

# ============================================================
# MODEL 2 — Within-Field Variability Model
#   Used for: pixel-by-pixel spatial yield maps only.
#   4 spatially-varying features (no ENV constants).
#   Retrained on the same best-stage data using these 4 features.
# ============================================================

M2_FEATURE_COLUMNS = ["NDVI_fused_mean", "VV_mean", "VH_mean"]

print(f"\n{'='*80}")
print("MODEL 1 — Plot-Level Yield Model  (training / validation)")
print(f"{'='*80}")
print(f"  VI  ({len(M1_VI_COLUMNS)}): {M1_VI_COLUMNS}")
print(f"  S1  ({len(M1_S1_COLUMNS)}): {M1_S1_COLUMNS}")
print(f"  ENV ({len(M1_ENV_COLUMNS)}): {M1_ENV_COLUMNS}")
print(f"  TOTAL: {len(M1_FEATURE_COLUMNS)} features")
print(f"\nMODEL 2 — Within-Field Variability Model  (spatial yield maps)")
print(f"{'='*80}")
print(f"  Spatial ({len(M2_FEATURE_COLUMNS)}): {M2_FEATURE_COLUMNS}")
print(f"{'='*80}\n")


# ============================================================
# MODEL FACTORY (Optimized hyperparameters)
# ============================================================

def build_model():
    """
    Build XGBoost regressor with optimized hyperparameters.
    """
    return XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=0.5,
        reg_lambda=1.5,
        min_child_weight=4,
        gamma=0.0,
        random_state=42,
    )


# ============================================================
# NESTED FEATURE SELECTION
# ============================================================
# Inner SFS runs only over the locked 8-feature candidate set.
# The pre-screening step is capped at min(10, len(candidates))
# so it works correctly when the candidate pool is small.

def nested_cv_with_sfs(
    X_df, y_series, groups_series, candidate_features,
    n_select=3, n_outer_splits=3, n_inner_splits=5, verbose=True,
):
    """
    Nested GroupKFold CV with inner Sequential Feature Selection.
    Returns OOF predictions and per-feature selection counts.
    """
    n_outer = min(n_outer_splits, groups_series.nunique())
    gkf     = GroupKFold(n_splits=n_outer)

    residuals, y_true_all, y_pred_all = [], [], []
    train_rmse_list, train_r2_list    = [], []
    groups_all      = []
    selected_counts = {f: 0 for f in candidate_features}

    for fold_idx, (tr_idx, te_idx) in enumerate(
        gkf.split(X_df, y_series, groups_series), start=1
    ):
        X_tr = X_df.iloc[tr_idx][candidate_features]
        X_te = X_df.iloc[te_idx][candidate_features]
        y_tr = y_series.iloc[tr_idx]
        y_te = y_series.iloc[te_idx]

        # ── Inner SFS (training data only) ───────────────────────
        scaler   = StandardScaler()
        X_tr_sc  = scaler.fit_transform(X_tr)

        # Pre-screen: top-N by abs correlation on THIS fold's training set
        # Capped at all available candidates (important when pool is small)
        tr_df     = pd.DataFrame(X_tr.values, columns=candidate_features)
        corr_tr   = tr_df.corrwith(y_tr.reset_index(drop=True)).abs()
        n_prescreen = min(len(candidate_features), 10)
        top_cands = corr_tr.sort_values(ascending=False).head(n_prescreen).index.tolist()
        feat_idx  = [candidate_features.index(f) for f in top_cands]
        X_tr_top  = X_tr_sc[:, feat_idx]

        sfs = SequentialFeatureSelector(
            build_model(),
            n_features_to_select=n_select,
            direction="forward",
            scoring="neg_mean_absolute_error",
            cv=KFold(n_splits=n_inner_splits, shuffle=True, random_state=42),
            n_jobs=-1,
        )
        sfs.fit(X_tr_top, y_tr)

        fold_selected = [top_cands[i] for i, s in enumerate(sfs.get_support()) if s]
        for f in fold_selected:
            selected_counts[f] += 1

        # ── Outer fold evaluation ─────────────────────────────────
        model = build_model()
        model.fit(X_tr[fold_selected], y_tr)

        preds = model.predict(X_te[fold_selected])
        residuals.extend(y_te.values - preds)
        y_true_all.extend(y_te.values)
        y_pred_all.extend(preds)
        groups_all.extend(groups_series.iloc[te_idx].values)

        tr_preds = model.predict(X_tr[fold_selected])
        train_rmse_list.append(np.sqrt(mean_squared_error(y_tr, tr_preds)))
        train_r2_list.append(r2_score(y_tr, tr_preds))

        if verbose:
            print(
                f"     Fold {fold_idx}: {fold_selected}"
                f"  OOF-MAE={mean_absolute_error(y_te, preds):.1f}g"
            )

    return (
        np.array(residuals), np.array(y_true_all), np.array(y_pred_all),
        groups_all, selected_counts,
        train_rmse_list, train_r2_list, n_outer,
    )


# ============================================================
# STAGE-WISE ML + UNCERTAINTY ESTIMATION
# ============================================================

stage_results        = {}
stage_uncertainty    = {}
stage_cv_predictions = {}

for stage in STAGE_ORDER:

    stage_data = data[data[STAGE] == stage]
    if stage_data.empty:
        continue

    print(f"\n{'='*80}")
    print(f"Processing stage: {stage.upper()}")
    print(f"{'='*80}")

    # Keep only locked features that are present and have variance
    X = stage_data[FEATURE_COLUMNS].copy()
    X = X.loc[:, X.std() > 0]

    y      = stage_data[TARGET]
    groups = stage_data[LOCATION]

    vi_avail  = [c for c in VI_COLUMNS  if c in X.columns]
    s1_avail  = [c for c in S1_COLUMNS  if c in X.columns]
    env_avail = [c for c in ENV_COLUMNS if c in X.columns]
    all_avail = vi_avail + s1_avail + env_avail

    # Informational correlations
    if vi_avail:
        vi_corr = X[vi_avail].corrwith(y).abs().sort_values(ascending=False)
        print(f"\n📈 VI correlations:")
        print("   " + "  ".join(f"{k}({v:.3f})" for k, v in vi_corr.items()))

    if s1_avail:
        s1_corr = X[s1_avail].corrwith(y).abs().sort_values(ascending=False)
        print(f"\n📡 S1 correlations:")
        print("   " + "  ".join(f"{k}({v:.3f})" for k, v in s1_corr.items()))

    if env_avail:
        env_corr = X[env_avail].corrwith(y).abs().sort_values(ascending=False)
        print(f"\n🌍 ENV correlations:")
        print("   " + "  ".join(f"{k}({v:.3f})" for k, v in env_corr.items()))

    # ── Nested CV + SFS ──────────────────────────────────────────
    print(f"\n🔁 Nested CV + SFS ({stage.upper()})...")

    (residuals, y_true_all, y_pred_all, groups_all,
     selected_counts, train_rmse_list, train_r2_list,
     n_outer_actual) = nested_cv_with_sfs(
        X_df=X, y_series=y, groups_series=groups,
        candidate_features=all_avail,
        n_select=3, n_outer_splits=3, n_inner_splits=5, verbose=True,
    )

    # Stable feature set = majority-vote across outer folds
    majority = n_outer_actual // 2 + 1
    stable_features = [
        f for f, cnt in sorted(selected_counts.items(), key=lambda kv: -kv[1])
        if cnt >= majority
    ]
    if len(stable_features) < 3:
        stable_features = sorted(selected_counts, key=lambda f: -selected_counts[f])[:3]

    print(f"\n✅ Stable features (selected in ≥{majority}/{n_outer_actual} folds):")
    for f in stable_features:
        ft  = "VI" if f in vi_avail else ("S1" if f in s1_avail else "ENV")
        print(f"   [{ft}] {f}  ({selected_counts[f]}/{n_outer_actual} folds)")

    # ── Metrics in grams ─────────────────────────────────────────
    mae_g  = mean_absolute_error(y_true_all, y_pred_all)
    rmse_g = np.sqrt(mean_squared_error(y_true_all, y_pred_all))
    r2     = r2_score(y_true_all, y_pred_all)

    avg_train_rmse = float(np.mean(train_rmse_list))
    avg_train_r2   = float(np.mean(train_r2_list))

    plot_area_ha        = stage_data["plot_area_m2"].mean() / 10_000
    uncertainty_tons_ha = (residuals.std() / 1_000_000) / plot_area_ha

    ci_68_lower = -uncertainty_tons_ha
    ci_68_upper = +uncertainty_tons_ha
    ci_95_lower = -1.96 * uncertainty_tons_ha
    ci_95_upper = +1.96 * uncertainty_tons_ha

    y_true_arr = np.array(y_true_all)
    y_pred_arr = np.array(y_pred_all)
    pa_arr     = stage_data["plot_area_m2"].values[:len(y_true_arr)]

    y_true_tha = (y_true_arr / 1_000_000) / (pa_arr / 10_000)
    y_pred_tha = (y_pred_arr / 1_000_000) / (pa_arr / 10_000)

    std_true_tha     = float(np.std(y_true_tha))
    std_pred_tha     = float(np.std(y_pred_tha))
    std_residual_tha = float(np.std(y_true_tha - y_pred_tha))

    print(f"\n📊 Performance")
    print(f"   Train RMSE  : {avg_train_rmse:.2f} g   Train R² : {avg_train_r2:.3f}")
    print(f"   Val MAE     : {mae_g:.2f} g")
    print(f"   Val RMSE    : {rmse_g:.2f} g   Val R²   : {r2:.3f}")
    print(f"   Uncertainty : ±{uncertainty_tons_ha:.2f} t/ha")
    print(f"   68% CI      : {ci_68_lower:.2f} → {ci_68_upper:.2f} t/ha")
    print(f"   95% CI      : {ci_95_lower:.2f} → {ci_95_upper:.2f} t/ha")
    print(f"   Std Dev (Observed)  : {std_true_tha:.3f} t/ha")
    print(f"   Std Dev (Predicted) : {std_pred_tha:.3f} t/ha")
    print(f"   Std Dev (Residuals) : {std_residual_tha:.3f} t/ha")

    best_vi_label = vi_corr.index[0] if vi_avail else "N/A"

    stage_results[stage] = {
        "mae_g": mae_g, "rmse_g": rmse_g, "r2": r2,
        "train_rmse": avg_train_rmse, "train_r2": avg_train_r2,
        "uncertainty_tons_ha": uncertainty_tons_ha,
        "ci_68_lower": ci_68_lower, "ci_68_upper": ci_68_upper,
        "ci_95_lower": ci_95_lower, "ci_95_upper": ci_95_upper,
        "features": stable_features, "best_vi": best_vi_label,
        "selection_counts": selected_counts,
        "std_observed_tha": std_true_tha,
        "std_predicted_tha": std_pred_tha,
        "std_residuals_tha": std_residual_tha,
    }
    stage_uncertainty[stage] = uncertainty_tons_ha

    pa = stage_data["plot_area_m2"].values
    y_true_tons = (y_true_all / 1_000_000) / (pa[:len(y_true_all)] / 10_000)
    y_pred_tons = (y_pred_all / 1_000_000) / (pa[:len(y_pred_all)] / 10_000)
    stage_cv_predictions[stage] = {
        "y_true": y_true_tons, "y_pred": y_pred_tons, "groups": groups_all,
    }


# ============================================================
# BEST STAGE
# ============================================================

BEST_STAGE = min(stage_results, key=lambda s: stage_results[s]["mae_g"])
res_best   = stage_results[BEST_STAGE]

print(f"\n{'='*80}")
print(f"🏆 Best stage: {BEST_STAGE.upper()}")
print(f"{'='*80}")
print(f"   MAE={res_best['mae_g']:.2f}g  RMSE={res_best['rmse_g']:.2f}g  R²={res_best['r2']:.3f}")
print(f"   Stable features  : {', '.join(res_best['features'])}")
print(f"   Uncertainty      : ±{res_best['uncertainty_tons_ha']:.2f} t/ha")
print(f"   68% CI : {res_best['ci_68_lower']:.2f} → {res_best['ci_68_upper']:.2f} t/ha")
print(f"   95% CI : {res_best['ci_95_lower']:.2f} → {res_best['ci_95_upper']:.2f} t/ha")
print(f"   Std Dev (Observed)  : {res_best['std_observed_tha']:.3f} t/ha")
print(f"   Std Dev (Predicted) : {res_best['std_predicted_tha']:.3f} t/ha")
print(f"   Std Dev (Residuals) : {res_best['std_residuals_tha']:.3f} t/ha")


# ============================================================
# FINAL MODEL – trained on full best-stage data
# ============================================================

print(f"\n🔧 Training final XGBoost model on {BEST_STAGE.upper()} stage (all samples)...")

best_data   = data[data[STAGE] == BEST_STAGE].copy()
final_feats = res_best["features"]
X_final     = best_data[final_feats]
y_final     = best_data[TARGET]

final_model = build_model()
final_model.fit(X_final, y_final)

# Training metrics
tr_preds_g   = final_model.predict(X_final)
tr_preds_tha = (tr_preds_g   / 1_000_000) / (best_data["plot_area_m2"].values / 10_000)
y_tr_tha     = (y_final.values / 1_000_000) / (best_data["plot_area_m2"].values / 10_000)
train_rmse_tha = np.sqrt(mean_squared_error(y_tr_tha, tr_preds_tha))
train_r2_tha   = r2_score(y_tr_tha, tr_preds_tha)
mean_yield, std_yield = tr_preds_tha.mean(), tr_preds_tha.std()

# Validation (GroupKFold, stable features, no re-selection)
y_true_val, y_pred_val = [], []
gkf = GroupKFold(n_splits=min(3, best_data[LOCATION].nunique()))
for tr, te in gkf.split(X_final, y_final, best_data[LOCATION]):
    m = build_model()
    m.fit(X_final.iloc[tr], y_final.iloc[tr])
    p   = m.predict(X_final.iloc[te])
    p_t = (p / 1_000_000) / (best_data.iloc[te]["plot_area_m2"].values / 10_000)
    y_t = (y_final.iloc[te].values / 1_000_000) / (
          best_data.iloc[te]["plot_area_m2"].values / 10_000)
    y_true_val.extend(y_t); y_pred_val.extend(p_t)

y_true_val  = np.array(y_true_val)
y_pred_val  = np.array(y_pred_val)
val_rmse_tha = np.sqrt(mean_squared_error(y_true_val, y_pred_val))
val_r2_tha   = r2_score(y_true_val, y_pred_val)

std_true_val  = float(np.std(y_true_val))
std_pred_val  = float(np.std(y_pred_val))
std_resid_val = float(np.std(y_true_val - y_pred_val))

val_residuals_tha     = y_true_val - y_pred_val
final_uncertainty_tha = float(val_residuals_tha.std())
final_ci_68_lower     = -final_uncertainty_tha
final_ci_68_upper     = +final_uncertainty_tha
final_ci_95_lower     = -1.96 * final_uncertainty_tha
final_ci_95_upper     = +1.96 * final_uncertainty_tha

print(f"\n📊 Final model ({BEST_STAGE.upper()}):")
print(f"   Mean yield     : {mean_yield:.2f} t/ha   Std: {std_yield:.2f}")
print(f"   Training RMSE  : {train_rmse_tha:.3f} t/ha   R²: {train_r2_tha:.3f}")
print(f"   Validation RMSE: {val_rmse_tha:.3f} t/ha   R²: {val_r2_tha:.3f}")
print(f"   Uncertainty    : ±{final_uncertainty_tha:.3f} t/ha")
print(f"   68% CI         : {final_ci_68_lower:.3f} → {final_ci_68_upper:.3f} t/ha")
print(f"   95% CI         : {final_ci_95_lower:.3f} → {final_ci_95_upper:.3f} t/ha")
print(f"   Std Dev (Val Observed)  : {std_true_val:.3f} t/ha")
print(f"   Std Dev (Val Predicted) : {std_pred_val:.3f} t/ha")
print(f"   Std Dev (Val Residuals) : {std_resid_val:.3f} t/ha")


# ============================================================
# MODEL 2 — Within-Field Variability Model
# Trained on the same best-stage data using only the 4
# spatially-varying features (NDVI_fused_mean, NDVI_fused_auc,
# VV_mean, VH_mean).  ENV columns are intentionally excluded
# because they are spatially constant at the field scale and
# cannot drive pixel-to-pixel variability in the yield maps.
# ============================================================

print(f"\n{'='*80}")
print(f"🗺️  Training Model 2 (Within-Field Variability) on {BEST_STAGE.upper()} stage...")
print(f"   Features: {M2_FEATURE_COLUMNS}")
print(f"{'='*80}")

# Restrict to features that actually exist and have variance
m2_feats_avail = [f for f in M2_FEATURE_COLUMNS
                  if f in best_data.columns and best_data[f].std() > 0]
if len(m2_feats_avail) < len(M2_FEATURE_COLUMNS):
    missing = set(M2_FEATURE_COLUMNS) - set(m2_feats_avail)
    print(f"   ⚠️  Missing/zero-variance Model 2 features dropped: {missing}")

X_m2    = best_data[m2_feats_avail]
map_model = build_model()
map_model.fit(X_m2, y_final)

# Quick training-set check (no separate val needed – maps are exploratory)
m2_tr_preds_g   = map_model.predict(X_m2)
m2_tr_preds_tha = (m2_tr_preds_g / 1_000_000) / (best_data["plot_area_m2"].values / 10_000)
m2_train_rmse   = np.sqrt(mean_squared_error(y_tr_tha, m2_tr_preds_tha))
m2_train_r2     = r2_score(y_tr_tha, m2_tr_preds_tha)

print(f"   Model 2 training RMSE : {m2_train_rmse:.3f} t/ha   R²: {m2_train_r2:.3f}")
print(f"   (Maps use Model 2; predictions/validation use Model 1)")


# ============================================================
# SINGLE-PANEL: PREDICTED VS OBSERVED (BEST STAGE)
# ============================================================

fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(y_true_val, y_pred_val, alpha=0.6, edgecolors='k', s=80)
lo = min(y_true_val.min(), y_pred_val.min())
hi = max(y_true_val.max(), y_pred_val.max())
ax.plot([lo, hi], [lo, hi], 'r--', lw=2, label='1:1 Line')
ax.set_xlabel("Observed Yield (tons/ha)",  fontsize=12, fontweight='bold')
ax.set_ylabel("Predicted Yield (tons/ha)", fontsize=12, fontweight='bold')
ax.set_title(f"Predicted vs Observed – {BEST_STAGE.upper()} Stage",
             fontsize=14, fontweight='bold')
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
ax.text(0.95, 0.05,
        f"Val R² = {val_r2_tha:.3f}\nVal RMSE = {val_rmse_tha:.3f} t/ha\n"
        f"Train R² = {train_r2_tha:.3f}\nTrain RMSE = {train_rmse_tha:.3f} t/ha\n"
        f"Uncertainty = ±{final_uncertainty_tha:.3f} t/ha\n"
        f"68% CI: {final_ci_68_lower:.3f} → {final_ci_68_upper:.3f} t/ha\n"
        f"95% CI: {final_ci_95_lower:.3f} → {final_ci_95_upper:.3f} t/ha\n"
        f"σ Obs = {std_true_val:.3f} t/ha\nσ Pred = {std_pred_val:.3f} t/ha",
        transform=ax.transAxes, fontsize=10, va='bottom', ha='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIRECTORY, f"Predicted_vs_Observed_{BEST_STAGE.upper()}.png"),
    dpi=300)
print("\n✅ Single-panel Predicted vs Observed saved")
plt.close()


# ============================================================
# MULTI-PANEL: PREDICTED VS OBSERVED (3 STAGES)
# ============================================================

print("\n📊 Generating multi-panel Predicted vs Observed...")

all_grps = []
for s in STAGE_ORDER:
    if s in stage_cv_predictions:
        all_grps.extend(stage_cv_predictions[s]["groups"])
all_locs = sorted(set(all_grps))
_n = len(all_locs)
_pal = ([plt.cm.tab10(i) for i in range(_n)]     if _n <= 10
        else [plt.cm.tab20(i) for i in range(_n)] if _n <= 20
        else [plt.cm.tab20(i % 20) for i in range(_n)])
loc_color_map = {loc: _pal[i] for i, loc in enumerate(all_locs)}

panels = [
    ("A", "Early Stage", stage_cv_predictions.get("early"), "early"),
    ("B", "Mid Stage",   stage_cv_predictions.get("mid"),   "mid"),
    ("C", "Late Stage",  stage_cv_predictions.get("late"),  "late"),
]

fig = plt.figure(figsize=(14, 13))
gs  = GridSpec(2, 2, figure=fig, top=0.91, bottom=0.07,
               left=0.08, right=0.97, hspace=0.50, wspace=0.30)
axes = [fig.add_subplot(gs[r, c]) for r, c in [(0,0),(0,1),(1,0)]]
legend_handles, legend_labels = [], []

for ax, (letter, title, pdata, sk) in zip(axes, panels):
    if pdata is None or len(pdata["y_true"]) == 0:
        ax.set_visible(False); continue

    yt, yp = np.asarray(pdata["y_true"]), np.asarray(pdata["y_pred"])
    grps   = list(pdata["groups"])
    res    = stage_results[sk]

    for loc in all_locs:
        mask = np.array([g == loc for g in grps])
        if mask.any():
            sc = ax.scatter(yp[mask], yt[mask], color=loc_color_map[loc],
                            s=45, alpha=0.7, edgecolors='none', zorder=3)
            if str(loc) not in legend_labels:
                legend_handles.append(sc); legend_labels.append(str(loc))

    pad = 0.05
    lo_r, hi_r = min(yt.min(), yp.min()), max(yt.max(), yp.max())
    span = hi_r - lo_r
    lim_lo, lim_hi = max(0, lo_r - pad*span), hi_r + pad*span
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], 'k-', lw=2, alpha=0.9, zorder=2)
    ax.set_xlim(lim_lo, lim_hi); ax.set_ylim(lim_lo, lim_hi)
    ax.set_xlabel("Predicted yield (Ton/Ha)", fontsize=13)
    ax.set_ylabel("Reported yield (Ton/Ha)",  fontsize=13)
    ax.tick_params(labelsize=11, width=1, length=4)
    for sp in ax.spines.values():
        sp.set_linewidth(1.2); sp.set_color('black')
    ax.set_facecolor('white'); ax.grid(False)

    ax.text(0.05, 0.95, f"({letter}) {title}",
            transform=ax.transAxes, fontsize=13, va='top', ha='left', fontweight='bold')
    ax.text(0.5, 1.10,
            f"MAE={res['mae_g']:.2f}g    RMSE={res['rmse_g']:.2f}g    "
            f"σ Obs={res['std_observed_tha']:.3f}    σ Pred={res['std_predicted_tha']:.3f} t/ha",
            transform=ax.transAxes, fontsize=11, ha='center', va='bottom', clip_on=False)
    ax.text(0.5, 1.03,
            f"R²={res['r2']:.3f}    Uncertainty=±{res['uncertainty_tons_ha']:.2f} t/ha    "
            f"σ Residuals={res['std_residuals_tha']:.3f} t/ha",
            transform=ax.transAxes, fontsize=11, ha='center', va='bottom',
            fontweight='bold', clip_on=False)

ax_leg = fig.add_subplot(gs[1, 1])
ax_leg.set_axis_off()
ax_leg.legend(legend_handles, legend_labels, loc='center',
              ncol=(2 if len(legend_labels) > 14 else 1),
              frameon=True, framealpha=1.0, edgecolor='black',
              fontsize=10, handletextpad=0.5, labelspacing=0.55,
              columnspacing=1.0, borderpad=0.9, markerscale=1.5,
              title='Plot ID', title_fontsize=11)

mp_path = os.path.join(OUTPUT_DIRECTORY, "Predicted_vs_Observed_3STAGES_multipanel.png")
plt.savefig(mp_path, dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none', pad_inches=0.25)
print(f"✅ Multi-panel plot saved → {mp_path}")
plt.close()


# ============================================================
# XGBOOST YIELD MAPS (pixel-by-pixel inference — Model 2)
# ============================================================
# Model 2 uses only spatially-varying bands so every pixel
# gets a distinct prediction driven by real spatial signals:
#   NDVI_fused_mean / NDVI_fused_auc  → Kalman-fused NDVI median
#                                        (renamed to match column name)
#   VH_mean / VV_mean                 → S1 GRD median VH/VV bands
#
# ENV features are deliberately excluded here – they are
# spatially constant at field scale and would suppress
# within-field variability in the output raster.
#
# Pixels where ANY input band is NaN/nodata are written as NaN.


def build_gee_feature_image(feature_list, s2_fused_img, s1_coll,
                             start, end, aoi, training_df):
    """
    Build a stacked ee.Image with one band per feature in feature_list.
    Supports VI (fused NDVI), S1 (VV/VH), and ENV (constant) bands.
    """
    s1_stage = s1_coll.filterDate(start, end)
    s1_vv    = s1_stage.select("VV").median().clip(aoi).rename("VV_mean")
    s1_vh    = s1_stage.select("VH").median().clip(aoi).rename("VH_mean")

    band_imgs = []
    for feat in feature_list:
        if feat == "VV_mean":
            band_imgs.append(s1_vv)
        elif feat == "VH_mean":
            band_imgs.append(s1_vh)
        elif feat in M1_ENV_COLUMNS:
            feat_mean = float(training_df[feat].mean())
            band_imgs.append(
                ee.Image.constant(feat_mean).float().rename(feat).clip(aoi)
            )
        else:
            # VI feature (NDVI_fused_mean or NDVI_fused_auc):
            # use fused NDVI renamed to this column name
            band_imgs.append(s2_fused_img.rename(feat))

    stacked = band_imgs[0]
    for img in band_imgs[1:]:
        stacked = stacked.addBands(img)

    return stacked


def apply_xgb_to_raster(tif_path, model, avg_plot_area_m2, output_path):
    """
    Load a multi-band GeoTIFF, run XGBoost predict() on every valid pixel,
    and write a single-band yield (tons/ha) GeoTIFF.
    """
    with rasterio.open(tif_path) as src:
        data_arr = src.read()
        profile  = src.profile.copy()
        nodata   = src.nodata

    bands, rows, cols = data_arr.shape
    flat = data_arr.reshape(bands, rows * cols).T

    if nodata is not None:
        nan_mask = np.any((flat == nodata) | ~np.isfinite(flat), axis=1)
    else:
        nan_mask = np.any(~np.isfinite(flat), axis=1)

    valid_flat   = flat[~nan_mask]
    plot_area_ha = avg_plot_area_m2 / 10_000

    yield_flat = np.full(rows * cols, np.nan, dtype=np.float32)
    if valid_flat.shape[0] > 0:
        preds_g               = model.predict(valid_flat)
        preds_tha             = (preds_g / 1_000_000) / plot_area_ha
        yield_flat[~nan_mask] = preds_tha.astype(np.float32)

    yield_arr = yield_flat.reshape(rows, cols)

    profile.update(count=1, dtype="float32", nodata=np.nan)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(yield_arr[np.newaxis, :, :])

    return yield_arr


print("\n🗺️  Generating XGBoost yield maps (pixel-by-pixel inference — Model 2)...")
print(f"   Model 2 features: {m2_feats_avail}\n")

stage_yield_maps = {}

for stage in STAGE_ORDER:
    stage_data_s = data[data[STAGE] == stage]
    if stage_data_s.empty:
        print(f"   ⚠️  No data for {stage.upper()}, skipping map.")
        continue

    start_s, end_s = GROWTH_STAGES[stage]
    avg_area_m2    = stage_data_s["plot_area_m2"].mean()

    # Build GEE feature stack with Model 2 spatial features only
    feat_img = build_gee_feature_image(
        feature_list   = m2_feats_avail,
        s2_fused_img   = stage_fused_ndvi[stage],
        s1_coll        = s1,
        start          = start_s,
        end            = end_s,
        aoi            = AOI,
        training_df    = stage_data_s,
    )

    feat_tif = os.path.join(OUTPUT_DIRECTORY, f"FeatureStack_M2_{stage.upper()}.tif")
    print(f"   [{stage.upper()}] Exporting GEE feature stack (Model 2)...")
    with suppress_output():
        geemap.ee_export_image(
            feat_img, feat_tif,
            scale=10, region=AOI, file_per_band=False,
        )
    print(f"   [{stage.upper()}] Feature stack saved → {feat_tif}")

    yield_tif = os.path.join(
        OUTPUT_DIRECTORY, f"Yield_XGB_M2_{stage.upper()}_stage_TONS_PER_HA.tif"
    )
    print(f"   [{stage.upper()}] Applying Model 2 to raster...")
    yield_arr = apply_xgb_to_raster(feat_tif, map_model, avg_area_m2, yield_tif)
    stage_yield_maps[stage] = yield_arr

    valid_px = yield_arr[np.isfinite(yield_arr)]
    if len(valid_px):
        print(f"   [{stage.upper()}] Yield stats  "
              f"min={valid_px.min():.2f}  mean={valid_px.mean():.2f}"
              f"  max={valid_px.max():.2f}  std={valid_px.std():.2f}  t/ha")
    print(f"   [{stage.upper()}] Yield map saved → {yield_tif}")

print("✅ All Model 2 yield maps generated")


# ============================================================
# VISUALISE XGBOOST YIELD MAPS
# ============================================================

print("\n🖼️  Visualising XGBoost yield maps (3-panel figure)...")

stages_with_maps = [s for s in STAGE_ORDER if s in stage_yield_maps]
n_panels = len(stages_with_maps)

if n_panels > 0:
    all_vals = np.concatenate([
        stage_yield_maps[s][np.isfinite(stage_yield_maps[s])].ravel()
        for s in stages_with_maps
    ])
    vmin = float(np.percentile(all_vals, 2))
    vmax = float(np.percentile(all_vals, 98))
    cmap = plt.cm.RdYlGn

    fig_ym, axes_ym = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    if n_panels == 1:
        axes_ym = [axes_ym]

    for ax_ym, stage in zip(axes_ym, stages_with_maps):
        valid_px = stage_yield_maps[stage][np.isfinite(stage_yield_maps[stage])]
        mean_map = float(valid_px.mean()) if len(valid_px) else float('nan')
        std_map  = float(valid_px.std()) if len(valid_px) else float('nan')

        stage_res        = stage_results[stage]
        stage_r2         = stage_res['r2']
        stage_rmse_g     = stage_res['rmse_g']
        stage_plot_area_ha = data[data[STAGE] == stage]["plot_area_m2"].mean() / 10_000
        stage_rmse_tha   = (stage_rmse_g / 1_000_000) / stage_plot_area_ha

        im = ax_ym.imshow(
            stage_yield_maps[stage], cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto'
        )
        ax_ym.set_title(
            f"{stage.capitalize()} Stage (XGBoost)\n"
            f"μ={mean_map:.2f} t/ha, σ={std_map:.2f} t/ha\n"
            f"Val R²={stage_r2:.3f}, RMSE={stage_rmse_tha:.3f} t/ha",
            fontsize=11, fontweight='bold'
        )
        ax_ym.axis('off')
        plt.colorbar(im, ax=ax_ym, fraction=0.046, pad=0.04, label="Yield (t/ha)")

    fig_ym.suptitle("XGBoost Yield Maps – All Growth Stages",
                    fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    ym_fig_path = os.path.join(OUTPUT_DIRECTORY, "YieldMaps_XGBoost_3stages.png")
    plt.savefig(ym_fig_path, dpi=300, bbox_inches='tight')
    print(f"✅ Yield map figure saved → {ym_fig_path}")
    plt.close()


# ============================================================
# INDIVIDUAL YIELD MAPS PER STAGE
# ============================================================

print("\n🖼️  Generating individual yield maps for each growth stage...")

for stage in stages_with_maps:
    yield_arr = stage_yield_maps[stage]
    valid_px  = yield_arr[np.isfinite(yield_arr)]

    if len(valid_px) == 0:
        print(f"   ⚠️  No valid pixels for {stage.upper()}, skipping individual map.")
        continue

    stage_res          = stage_results[stage]
    stage_r2           = stage_res['r2']
    stage_rmse_g       = stage_res['rmse_g']
    stage_mae_g        = stage_res['mae_g']
    stage_plot_area_ha = data[data[STAGE] == stage]["plot_area_m2"].mean() / 10_000
    stage_rmse_tha     = (stage_rmse_g / 1_000_000) / stage_plot_area_ha
    stage_mae_tha      = (stage_mae_g  / 1_000_000) / stage_plot_area_ha

    vmin_stage = float(np.percentile(valid_px, 2))
    vmax_stage = float(np.percentile(valid_px, 98))

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(yield_arr, cmap=plt.cm.RdYlGn,
                   vmin=vmin_stage, vmax=vmax_stage, aspect='auto')
    ax.set_title(f"Predicted Yield Map – {stage.capitalize()} Stage\n"
                 f"(XGBoost Model, {len(stage_res['features'])} features)",
                 fontsize=14, fontweight='bold', pad=20)
    ax.axis('off')

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Predicted Yield (t/ha)", fontsize=12, fontweight='bold')

    stats_text = (
        f"Model Performance:\n"
        f"Val R² = {stage_r2:.3f}\n"
        f"Val RMSE = {stage_rmse_tha:.3f} t/ha\n"
        f"Val MAE = {stage_mae_tha:.3f} t/ha"
    )
    ax.text(0.02, 0.98, stats_text,
            transform=ax.transAxes, fontsize=11, va='top', ha='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9,
                      edgecolor='black', linewidth=1.5),
            family='monospace')

    plt.tight_layout()
    individual_map_path = os.path.join(
        OUTPUT_DIRECTORY, f"YieldMap_XGB_{stage.upper()}_stage_individual.png"
    )
    plt.savefig(individual_map_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"   ✓ YieldMap_XGB_{stage.upper()}_stage_individual.png")
    plt.close()

print("✅ Individual yield maps saved")


# ============================================================
# STAGE COMPARISON: SIDE-BY-SIDE YIELD MAPS
# ============================================================

print("\n🖼️  Generating side-by-side yield comparison across stages...")

if len(stages_with_maps) >= 2:
    all_valid = np.concatenate([
        stage_yield_maps[s][np.isfinite(stage_yield_maps[s])].ravel()
        for s in stages_with_maps
    ])
    vmin_comp = float(np.percentile(all_valid, 2))
    vmax_comp = float(np.percentile(all_valid, 98))

    fig, axes = plt.subplots(1, len(stages_with_maps),
                             figsize=(7 * len(stages_with_maps), 6))
    if len(stages_with_maps) == 1:
        axes = [axes]

    for ax, stage in zip(axes, stages_with_maps):
        yield_arr = stage_yield_maps[stage]
        valid_px  = yield_arr[np.isfinite(yield_arr)]

        im = ax.imshow(yield_arr, cmap=plt.cm.RdYlGn,
                       vmin=vmin_comp, vmax=vmax_comp, aspect='auto')

        mean_y = valid_px.mean() if len(valid_px) > 0 else 0
        std_y  = valid_px.std()  if len(valid_px) > 0 else 0

        stage_res          = stage_results[stage]
        stage_r2           = stage_res['r2']
        stage_rmse_g       = stage_res['rmse_g']
        stage_plot_area_ha = data[data[STAGE] == stage]["plot_area_m2"].mean() / 10_000
        stage_rmse_tha     = (stage_rmse_g / 1_000_000) / stage_plot_area_ha

        ax.set_title(f"{stage.capitalize()} Stage\n"
                     f"μ={mean_y:.2f} t/ha, σ={std_y:.2f} t/ha\n"
                     f"Val R²={stage_r2:.3f}, RMSE={stage_rmse_tha:.3f} t/ha",
                     fontsize=11, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Yield (t/ha)")

    fig.suptitle("Growth Stage Yield Comparison\n"
                 "(Common Color Scale for Direct Comparison)",
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    comparison_path = os.path.join(OUTPUT_DIRECTORY,
                                   "YieldMap_Comparison_ALL_STAGES.png")
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"✅ Stage comparison map saved → {comparison_path}")
    plt.close()


# ============================================================
# YIELD DIFFERENCE MAPS
# ============================================================

if len(stages_with_maps) >= 2:
    print("\n🖼️  Generating yield difference maps between stages...")

    for i in range(len(stages_with_maps) - 1):
        stage1 = stages_with_maps[i]
        stage2 = stages_with_maps[i + 1]

        arr1 = stage_yield_maps[stage1]
        arr2 = stage_yield_maps[stage2]
        diff_arr   = arr2 - arr1
        valid_diff = diff_arr[np.isfinite(diff_arr)]

        if len(valid_diff) == 0:
            continue

        abs_max  = max(abs(np.percentile(valid_diff, 2)),
                       abs(np.percentile(valid_diff, 98)))
        vmin_diff = -abs_max
        vmax_diff =  abs_max

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(diff_arr, cmap=plt.cm.RdBu_r,
                       vmin=vmin_diff, vmax=vmax_diff, aspect='auto')
        ax.set_title(f"Yield Change: {stage2.capitalize()} → {stage1.capitalize()}\n"
                     f"({stage2.capitalize()} stage minus {stage1.capitalize()} stage)\n"
                     f"(Positive = Yield Increase from {stage1.capitalize()} to {stage2.capitalize()})",
                     fontsize=13, fontweight='bold', pad=20)
        ax.axis('off')

        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Yield Difference (t/ha)", fontsize=12, fontweight='bold')

        mean_diff    = float(valid_diff.mean())
        std_diff     = float(valid_diff.std())
        pct_increase = float((valid_diff > 0).sum() / len(valid_diff) * 100)
        pct_decrease = float((valid_diff < 0).sum() / len(valid_diff) * 100)

        stats_text = (
            f"Mean change: {mean_diff:+.2f} t/ha\n"
            f"Std: {std_diff:.2f} t/ha\n"
            f"Pixels increasing: {pct_increase:.1f}%\n"
            f"Pixels decreasing: {pct_decrease:.1f}%"
        )
        ax.text(0.02, 0.98, stats_text,
                transform=ax.transAxes, fontsize=10, va='top', ha='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8,
                          edgecolor='black', linewidth=1.5))

        plt.tight_layout()
        diff_path = os.path.join(
            OUTPUT_DIRECTORY,
            f"YieldMap_Difference_{stage2.upper()}_minus_{stage1.upper()}.png"
        )
        plt.savefig(diff_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"   ✓ YieldMap_Difference_{stage2.upper()}_minus_{stage1.upper()}.png")
        plt.close()

    print("✅ Yield difference maps saved")


# ============================================================
# COMBINED NDVI MAPS (all stages averaged)
# ============================================================

print("\n🗺️  Exporting combined NDVI maps (season average)...")

combined_s2_only = (
    stage_s2_only_ndvi["early"]
    .add(stage_s2_only_ndvi["mid"])
    .add(stage_s2_only_ndvi["late"])
    .divide(3)
)
with suppress_output():
    geemap.ee_export_image(
        combined_s2_only,
        os.path.join(OUTPUT_DIRECTORY, "NDVI_S2_ONLY_COMBINED_all_stages.tif"),
        scale=10, region=AOI, file_per_band=False,
    )

combined_fused = (
    stage_fused_ndvi["early"]
    .add(stage_fused_ndvi["mid"])
    .add(stage_fused_ndvi["late"])
    .divide(3)
)
with suppress_output():
    geemap.ee_export_image(
        combined_fused,
        os.path.join(OUTPUT_DIRECTORY, "NDVI_FUSED_S2_S1_COMBINED_all_stages.tif"),
        scale=10, region=AOI, file_per_band=False,
    )

print("✅ Combined NDVI maps exported")


# ============================================================
# CONFIDENCE LEVEL SUMMARY TABLE (CSV)
# ============================================================

print("\n📊 Exporting uncertainty confidence levels table...")

conf_rows = []
for stage in STAGE_ORDER:
    if stage not in stage_results:
        continue
    r = stage_results[stage]
    conf_rows.append({
        "Growth Stage":              stage.upper(),
        "Selected Features":         ", ".join(r["features"]),
        "Uncertainty (t/ha)":        r["uncertainty_tons_ha"],
        "68% CI Lower (t/ha)":       r["ci_68_lower"],
        "68% CI Upper (t/ha)":       r["ci_68_upper"],
        "95% CI Lower (t/ha)":       r["ci_95_lower"],
        "95% CI Upper (t/ha)":       r["ci_95_upper"],
        "MAE (g)":                   r["mae_g"],
        "RMSE (g)":                  r["rmse_g"],
        "R²":                        r["r2"],
        "Std Dev Observed (t/ha)":   r["std_observed_tha"],
        "Std Dev Predicted (t/ha)":  r["std_predicted_tha"],
        "Std Dev Residuals (t/ha)":  r["std_residuals_tha"],
    })

conf_df  = pd.DataFrame(conf_rows)
conf_csv = os.path.join(OUTPUT_DIRECTORY, "uncertainty_confidence_levels.csv")
conf_df.to_csv(conf_csv, index=False, float_format="%.3f")

print(f"\n{'='*100}")
print("UNCERTAINTY CONFIDENCE LEVELS SUMMARY")
print(f"{'='*100}")
print(conf_df.to_string(index=False))
print(f"{'='*100}")
print(f"✅ Confidence table saved → {conf_csv}")


# ============================================================
# PIPELINE SUMMARY
# ============================================================

print(f"\n{'='*80}")
print("✅ PIPELINE COMPLETED SUCCESSFULLY  (v2.2 – Dual-Model Architecture)")
print(f"{'='*80}")
print(f"\n🎯 Dual-Model design:")
print(f"   Model 1  (Plot-Level Yield)       → training, validation, scatter plots, CSV")
print(f"   Model 2  (Within-Field Variability) → pixel-by-pixel spatial yield maps")
print(f"\n🔒 Model 1 features ({len(M1_FEATURE_COLUMNS)} total):")
print(f"   VI  : {M1_VI_COLUMNS}")
print(f"   S1  : {M1_S1_COLUMNS}")
print(f"   ENV : {M1_ENV_COLUMNS}")
print(f"\n🔒 Model 2 features ({len(m2_feats_avail)} spatial only):")
print(f"   {m2_feats_avail}")
print(f"\n📊 Best stage      : {BEST_STAGE.upper()}")
print(f"   M1 stable feats  : {', '.join(final_feats)}")
print(f"   M1 Val RMSE      : {val_rmse_tha:.3f} t/ha   Val R²: {val_r2_tha:.3f}")
print(f"   M1 Uncertainty   : ±{final_uncertainty_tha:.3f} t/ha")
print(f"   M1 68% CI        : {final_ci_68_lower:.3f} → {final_ci_68_upper:.3f} t/ha")
print(f"   M1 95% CI        : {final_ci_95_lower:.3f} → {final_ci_95_upper:.3f} t/ha")
print(f"   M1 σ Observed    : {std_true_val:.3f} t/ha")
print(f"   M1 σ Predicted   : {std_pred_val:.3f} t/ha")
print(f"   M1 σ Residuals   : {std_resid_val:.3f} t/ha")
print(f"   M2 Train RMSE    : {m2_train_rmse:.3f} t/ha   R²: {m2_train_r2:.3f}")
print(f"\n📂 Output directory: {OUTPUT_DIRECTORY}/")
print(f"\n📂 Key outputs:")
print(f"   NDVI_S2_ONLY_[STAGE]_stage.tif                 cloud-masked, genuine gaps")
print(f"   NDVI_FUSED_S2_S1_[STAGE]_stage.tif             Kalman-fused S2+S1")
print(f"   FARM_NDVI_Comparison_ALL_STAGES.png            Combined NDVI comparison (3x2 grid)")
print(f"   FeatureStack_M2_[STAGE].tif                    Model 2 GEE feature bands")
print(f"   Yield_XGB_M2_[STAGE]_stage_TONS_PER_HA.tif    Model 2 pixel yield map (GeoTIFF)")
print(f"   YieldMap_XGB_[STAGE]_stage_individual.png      Individual stage yield visualization")
print(f"   YieldMaps_XGBoost_3stages.png                  3-panel comparison")
print(f"   YieldMap_Comparison_ALL_STAGES.png             Side-by-side stage comparison")
print(f"   YieldMap_Difference_[STAGE]_minus_[STAGE].png  Yield change between stages")
print(f"   Predicted_vs_Observed_*.png                    Model 1 validation plots")
print(f"   uncertainty_confidence_levels.csv              Model 1 confidence table")
print(f"{'='*80}")
