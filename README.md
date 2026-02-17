# 🌾 AI Crop Yield Prediction System

## 🌍 Overview

A dual-model machine learning pipeline for predicting crop yields using fused Sentinel-2 (optical) and Sentinel-1 (SAR) satellite imagery, combined with environmental variables. The system integrates Google Earth Engine (GEE) for satellite data retrieval, Kalman filtering for image fusion, and XGBoost for yield prediction — producing both plot-level yield estimates and pixel-by-pixel spatial yield maps.

---

## 🏗️ Architecture

The pipeline uses a **dual-model design** to separate concerns between plot-level accuracy and spatial mapping.

### 🔬 Model 1 — Plot-Level Yield Model

Used for training, cross-validation, uncertainty estimation, and the confidence CSV output.

| Feature Group | Features |
|---|---|
| 🌿 Vegetation Index (VI) | `NDVI_fused_auc` |
| 📡 SAR / Sentinel-1 (S1) | `VV_mean`, `VH_mean` |
| 🌍 Environmental (ENV) | `rainfall`, `soil_organic_carbon`, `clay_content`, `temp` |
| **Total** | **7 features** |

### 🗺️ Model 2 — Within-Field Variability Model

Used exclusively for pixel-by-pixel spatial yield map generation. ENV features are intentionally excluded because they are spatially constant at the field scale and would suppress within-field variability.

| Feature Group | Features |
|---|---|
| 🛰️ Spatial (VI + S1) | `NDVI_fused_mean`, `VV_mean`, `VH_mean` |
| **Total** | **3 features** |

---

## ⚙️ Pipeline Steps

```
1.  ✅ GEE Initialization
2.  📍 AOI & Plot Geometry Loading
3.  ☁️  SCL Cloud Masking (Sentinel-2)
4.  📡 Sentinel-1 SAR Processing (VV, VH, RVI)
5.  🔀 Kalman Fusion (S2 + S1 NDVI per growth stage)
6.  📤 NDVI Export (S2-only + Fused GeoTIFFs)
7.  🌾 Farm-Boundary Comparison Maps
8.  📊 Tabular Data Loading
9.  🔁 Model 1 — Nested CV + Sequential Feature Selection (SFS)
10. 📏 Uncertainty & Confidence Interval Estimation
11. 🧠 Final Model 1 Training (best stage)
12. 🗺️  Model 2 Training (spatial features only)
13. 📤 GEE Feature Stack Export (per stage)
14. 🖼️  Pixel-by-pixel Yield Map Inference (Model 2)
15. 📈 Visualisation & Figure Generation
16. 💾 Confidence Level CSV Export
```

---

## 🌱 Growth Stages

| Stage | Date Range |
|---|---|
| 🌱 Early | 2025-03-01 → 2025-04-30 |
| 🌿 Mid | 2025-05-01 → 2025-06-30 |
| 🌾 Late | 2025-07-01 → 2025-08-10 |

---

## 📊 Results Summary (from latest run)

### 🏆 Best Stage: **LATE**

| Metric | Value |
|---|---|
| 🔒 Stable Features | `VV_mean`, `rainfall`, `soil_organic_carbon` |
| 📉 Val RMSE | 1.186 t/ha |
| 📈 Val R² | 0.172 |
| ⚠️ Uncertainty | ±1.138 t/ha |
| 📊 68% CI | −1.138 → +1.138 t/ha |
| 📊 95% CI | −2.231 → +2.231 t/ha |
| σ Observed | 1.303 t/ha |
| σ Predicted | 0.723 t/ha |
| σ Residuals | 1.138 t/ha |

### 🗺️ Model 2 (Spatial Maps — LATE Stage)

| Metric | Value |
|---|---|
| 📉 Training RMSE | 0.605 t/ha |
| 📈 Training R² | 0.784 |

### 📋 Per-Stage Uncertainty Summary

| Stage | MAE (g) | RMSE (g) | R² | Uncertainty (t/ha) | 68% CI | 95% CI |
|---|---|---|---|---|---|---|
| 🌱 EARLY | 132.88 | 162.92 | −0.327 | ±1.033 | −1.033 → +1.033 | −2.025 → +2.025 |
| 🌿 MID | 138.25 | 171.58 | −0.472 | ±1.081 | −1.081 → +1.081 | −2.118 → +2.118 |
| 🌾 LATE | 129.14 | 157.64 | −0.242 | ±0.994 | −0.994 → +0.994 | −1.948 → +1.948 |

### 🖼️ Yield Map Statistics (Model 2)

| Stage | Min (t/ha) | Mean (t/ha) | Max (t/ha) | Std (t/ha) |
|---|---|---|---|---|
| 🌱 EARLY | 0.78 | 1.28 | 1.54 | 0.32 |
| 🌿 MID | 0.78 | 1.23 | 1.54 | 0.35 |
| 🌾 LATE | 0.78 | 1.32 | 1.74 | 0.30 |

---

## 📦 Requirements

```bash
pip install numpy pandas rasterio earthengine-api geemap matplotlib scikit-learn xgboost
```

### 🐍 Python Dependencies

| Package | Purpose |
|---|---|
| `earthengine-api` | 🌍 Google Earth Engine access |
| `geemap` | 🗺️ GEE utilities & image export |
| `rasterio` | 🗂️ GeoTIFF read/write |
| `xgboost` | 🤖 Yield prediction model |
| `scikit-learn` | 🔬 Feature selection, cross-validation, metrics |
| `numpy` / `pandas` | 🔢 Data processing |
| `matplotlib` | 📈 Visualisation |

---

## 🚀 Setup

### 1. 🔐 Authenticate with Google Earth Engine

```python
import ee
ee.Authenticate()
ee.Initialize(project="your-project-id")
```

Update `PROJECT_ID` in the script to match your GEE project.

### 2. 📂 Required Input Files

| File | Description |
|---|---|
| `Farm/witz_farm.shp` | 🌾 Farm boundary shapefile (AOI) |
| `GGE_vector/GGE_Harvest_150_gcs.shp` | 📍 Plot-level harvest boundaries |
| `plot_satellite_indices_cloud_robust.csv` | 📊 Tabular training data |

### 3. 📋 CSV Training Data Schema

The tabular CSV (`plot_satellite_indices_cloud_robust.csv`) must contain the following columns:

| Column | Description |
|---|---|
| `dryYieldg` | 🎯 Target variable — dry yield in grams |
| `location` | 📍 Plot/location identifier (used for GroupKFold) |
| `growth_stage` | 🌱 Stage label: `early`, `mid`, or `late` |
| `plot_area_m2` | 📐 Plot area in square metres |
| `NDVI_fused_auc` | 🌿 Area under the NDVI curve (Kalman-fused) |
| `NDVI_fused_mean` | 🌿 Mean NDVI (Kalman-fused) |
| `VV_mean` | 📡 Sentinel-1 VV polarisation mean |
| `VH_mean` | 📡 Sentinel-1 VH polarisation mean |
| `rainfall` | 🌧️ Accumulated rainfall (mm) |
| `soil_organic_carbon` | 🪱 Soil organic carbon (%) |
| `clay_content` | 🏔️ Clay content (%) |
| `temp` | 🌡️ Mean temperature (°C) |

---

## ▶️ Running the Pipeline

```bash
python AI_crop.py
```

or if executable:

```bash
./AI_crop.py
```

---

## 📂 Outputs

All outputs are saved to the `output/` directory.

### 🗺️ GeoTIFFs

| File | Description |
|---|---|
| `NDVI_S2_ONLY_[STAGE]_stage.tif` | ☁️ Cloud-masked Sentinel-2 NDVI (genuine gaps preserved) |
| `NDVI_FUSED_S2_S1_[STAGE]_stage.tif` | 🔀 Kalman-fused S2+S1 NDVI |
| `FARM_NDVI_FUSED_S2_S1_[STAGE]_stage.tif` | 🌾 Farm-clipped fused NDVI |
| `FARM_NDVI_S2_ONLY_[STAGE]_stage.tif` | 🌾 Farm-clipped S2-only NDVI |
| `NDVI_S2_ONLY_COMBINED_all_stages.tif` | 📅 Season-average S2-only NDVI |
| `NDVI_FUSED_S2_S1_COMBINED_all_stages.tif` | 📅 Season-average fused NDVI |
| `FeatureStack_M2_[STAGE].tif` | 🧱 Model 2 input feature bands (GEE export) |
| `Yield_XGB_M2_[STAGE]_stage_TONS_PER_HA.tif` | 🌾 Pixel-level yield map in t/ha |

### 🖼️ Figures (PNG)

| File | Description |
|---|---|
| `FARM_NDVI_Comparison_ALL_STAGES.png` | 📊 3×2 grid: S2-only vs fused NDVI for all stages |
| `Predicted_vs_Observed_[BEST_STAGE].png` | 🎯 Single-panel validation scatter plot |
| `Predicted_vs_Observed_3STAGES_multipanel.png` | 📈 Multi-panel validation across all stages |
| `YieldMaps_XGBoost_3stages.png` | 🗺️ 3-panel yield map comparison |
| `YieldMap_XGB_[STAGE]_stage_individual.png` | 🖼️ Individual yield map per stage |
| `YieldMap_Comparison_ALL_STAGES.png` | 🔍 Side-by-side stage comparison (common colour scale) |
| `YieldMap_Difference_[STAGE]_minus_[STAGE].png` | 📉 Yield change maps between consecutive stages |

### 📄 CSV

| File | Description |
|---|---|
| `uncertainty_confidence_levels.csv` | 📊 Per-stage uncertainty, CI, MAE, RMSE, R², std dev summary |

---

## 🔬 Methodology Notes

### ☁️ Cloud Masking

Sentinel-2 images are filtered using the Scene Classification Layer (SCL). Only clear-sky pixels (SCL classes 4 = vegetation, 5 = bare soil, 6 = water) are retained. Cloudy pixels remain as genuine gaps — no artificial filling is applied.

### 🔀 Kalman Fusion

For each growth stage, a pixel-wise weighted average blends S2 NDVI (high spatial detail) and S1 RVI (cloud-penetrating SAR) using their respective variance as weights:

```
w_S2       = σ²_S1 / (σ²_S1 + σ²_S2)
w_S1       = σ²_S2 / (σ²_S1 + σ²_S2)
NDVI_fused = w_S2 × NDVI_S2 + w_S1 × NDVI_S1
```

S1 speckle variance is fixed at `0.05² = 0.0025`. A high default variance of `9999` is assigned to pixels with missing S2 data, so the fused output gracefully degrades to the S1 signal only where S2 coverage is absent.

### 🔁 Feature Selection

Nested cross-validation with inner Sequential Feature Selection (SFS) is used to identify a stable feature set:

- **🔄 Outer loop**: GroupKFold (3 splits, grouped by location) — prevents data leakage across plots
- **🔄 Inner loop**: KFold SFS (5 splits) on each fold's training data
- **✅ Stability rule**: Features selected in the majority of outer folds (≥ ⌊n/2⌋ + 1) are retained as the final stable set

### 🤖 XGBoost Hyperparameters

```python
XGBRegressor(
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
```

### 📏 Uncertainty Estimation

Uncertainty is calculated as the standard deviation of out-of-fold residuals (in t/ha), providing empirical 68% and 95% confidence intervals around each prediction:

```
Uncertainty  = std(residuals)
68% CI       = ±1.0  × Uncertainty
95% CI       = ±1.96 × Uncertainty
```

---

## 📁 Project Structure

```
AI_Crop_Yield_Prediction_Systems/
├── 🐍 AI_crop.py                               # Main pipeline script
├── 📊 plot_satellite_indices_cloud_robust.csv  # Tabular training data
├── 🌾 Farm/
│   └── witz_farm.shp                           # Farm AOI shapefile
├── 📍 GGE_vector/
│   └── GGE_Harvest_150_gcs.shp                 # Plot harvest boundaries
└── 📂 output/                                  # All generated outputs
    ├── 🗺️  *.tif                               # GeoTIFF rasters
    ├── 🖼️  *.png                               # Figures
    └── 📄 uncertainty_confidence_levels.csv    # Confidence summary
```

---

## 🏷️ Version

**v2.2 — Dual-Model Architecture**
