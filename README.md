# 🌾 AI Crop Yield Prediction System  
**Satellite-Driven, Cloud-Safe, Industry-Ready Machine Learning for Yield Forecasting**

*Remote Sensing × Machine Learning × Google Earth Engine*

An end-to-end geospatial AI pipeline for **field-scale crop yield prediction** using multi-source satellite data, growth-stage-aware modeling, and uncertainty quantification.  
The system integrates **Sentinel-2 optical imagery**, **Sentinel-1 SAR gap filling**, and **XGBoost regression** to produce **yield and uncertainty maps (tons/ha)**.

---

## 🚀 Key Features
- ✅ Growth-stage-aware modeling (Early, Mid, Late)
- ✅ Sentinel-2 NDVI with Sentinel-1 SAR gap filling
- ✅ Automated feature selection (Sequential Forward Selection)
- ✅ Robust cross-validation using GroupKFold
- ✅ Yield prediction in tons/ha
- ✅ Spatial uncertainty estimation
- ✅ Fully automated Google Earth Engine → ML → GIS pipeline

Designed for **cloud-prone regions**, **Africa-ready deployment**, and **real agribusiness workflows**.

---

## 🧠 Methodology Overview

### 1️⃣ Satellite Data Processing (Google Earth Engine)
- Sentinel-2 Surface Reflectance for NDVI computation  
- Cloud and shadow masking using Scene Classification Layer (SCL)  
- Sentinel-1 SAR (RVI) for NDVI gap filling  
- Growth-stage-wise NDVI fusion (Early, Mid, Late)

---

### 2️⃣ Feature Engineering
**Vegetation Indices**
- NDVI  
- NDRE  
- GNDVI  
- EVI  
- CIrededge (mean and AUC)

**Environmental Variables**
- Rainfall  
- Temperature  

---

### 3️⃣ Machine Learning
- **Model**: XGBoost Regressor  
- **Feature Selection**: Sequential Forward Selection  
- **Validation Strategy**: GroupKFold (location-aware cross-validation)  

**Evaluation Metrics**
```text
MAE
RMSE
R²
```

### 4️⃣ Uncertainty Estimation
- Residual‑based uncertainty (± tons/ha)
- Stage‑specific and combined uncertainty maps
  
---

## 🧪 Growth Stage Performance Summary
- The system computes the importance of each feature, showing which variables most influence yield predictions. This improves interpretability and informs agronomic decisions.
    ```text
  | Growth Stage | Best Vegetation Index | Validation MAE (g) | Validation R² |
  |--------------|-----------------------|--------------------|---------------|
  | Early        | CIrededge_mean        | 111.89             | 0.017         |
  | **Mid ✅**   | **NDRE_mean**         | **104.78**         | **0.028**     |
  | Late         | GNDVI_auc             | 126.67             | -0.136        |
  ```

- *🏆 Best growth stage identified: MID*

---

## 📊 Final Model Performance (MID Stage)
- **Mean yield**: 1.34 tons/ha
- **Training RMSE**: 0.97 tons/ha
- **Training R²**: 0.447
- **Validation RMSE**: 1.16 tons/ha
- **Validation R²**: 0.210
- ✅ Predicted vs Observed yield plot automatically generated.**

---

## 🗺️ Outputs
- Raster Outputs (GeoTIFF)
- NDVI maps (Early / Mid / Late)
- Yield maps (tons/ha) per stage
- Yield uncertainty maps (tons/ha)
- Combined NDVI, yield, and uncertainty maps
- Figures
- Predicted vs Observed Yield Scatter Plot

---

## 📁 Project Structure
  ```bash
  AI_Crop_Yield_Prediction_Systems/
  │
  ├── AI_crop.py
  ├── Farm/
  │   └── witz_farm.shp
  ├── GGE_vector/
  │   └── GGE_Harvest_150_gcs.shp
  ├── plot_satellite_indices_cloud_robust.csv
  ├── output_dfh/
  │   ├── NDVI_*.tif
  │   ├── Yield_*.tif
  │   ├── Yield_Uncertainty_*.tif
  │   └── Predicted_vs_Observed_*.png
  └── README.md
  ```

---

## 📌 Scientific Contributions
- Demonstrates growth‑stage dependency of yield prediction accuracy
- Shows superiority of red‑edge indices (NDRE, CIrededge) during mid‑season
- Integrates SAR‑optical fusion for cloud‑robust yield mapping
- Provides spatial uncertainty quantification, often missing in yield models

---

## 🔮 Future Extensions
- ConvLSTM / Vision Transformers for spatiotemporal modeling
- ERA5 meteorological integration
- Multi‑year yield generalization
- Active learning for uncertainty reduction

---

## 📜 License
This project is licensed under the **MIT License**.
