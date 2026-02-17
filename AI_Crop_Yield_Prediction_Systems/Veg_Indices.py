#!/usr/bin/env python
# ============================================================
# VEGETATION INDEX EXTRACTION – CLOUD-ROBUST (OPTICAL + SAR)
# + SOIL, TOPOGRAPHY, MOISTURE + KALMAN-FUSED NDVI
# ============================================================

import ee
import geopandas as gpd
import pandas as pd

# ============================================================
# INITIALIZE EARTH ENGINE
# ============================================================
ee.Initialize(project="quiet-subset-447718-q0")

# ============================================================
# INPUTS
# ============================================================
PLOT_SHP   = "/Users/afriyie/Downloads/AI_Crop_Yield_Prediction_Systems-main/AI_Crop_Yield_Prediction_Systems/GGE_vector/GGE_Harvest_150.shp"
CROP_COL   = "Field"
AREA_COL   = "Shape_Area"
OUTPUT_CSV = "plot_satellite_indices_cloud_robust.csv"
BATCH_SIZE = 10

# ============================================================
# SEASONAL STAGES – WINTER WHEAT (CENTRAL GERMANY)
# ============================================================
STAGES = {
    "early": ("2025-03-01", "2025-04-30"),
    "mid":   ("2025-05-01", "2025-06-30"),
    "late":  ("2025-07-01", "2025-08-10"),
}

# ============================================================
# KALMAN FUSION PARAMETERS
# (must match the crop-yield pipeline)
# ============================================================
S1_SPECKLE_VAR = 0.05 ** 2   # assumed S1 noise variance

# ============================================================
# LOAD & FIX SHAPEFILE
# ============================================================
gdf = gpd.read_file(PLOT_SHP)
gdf["geometry"] = gdf["geometry"].buffer(0)

gdf_utm = gdf.to_crs(epsg=32632)
gdf_utm["geometry"] = gdf_utm.geometry.centroid
gdf = gdf_utm.to_crs(epsg=4326)

print(f"✅ Loaded {len(gdf)} plots")

# ============================================================
# AOI – bounding box of all plots (used for Kalman variance)
# ============================================================
AOI = ee.FeatureCollection(
    [ee.Feature(ee.Geometry.Point(r.geometry.x, r.geometry.y))
     for _, r in gdf.iterrows()]
).geometry().bounds()

# ============================================================
# STATIC DATASETS (LOADED ONCE)
# ============================================================
soilgrids_soc  = ee.Image("projects/soilgrids-isric/soc_mean").select("soc_0-5cm_mean")
soilgrids_clay = ee.Image("projects/soilgrids-isric/clay_mean").select("clay_0-5cm_mean")

dem       = ee.Image("USGS/SRTMGL1_003")
elevation = dem.select("elevation")
slope     = ee.Terrain.slope(dem)

# ============================================================
# VEGETATION INDICES (OPTICAL)
# ============================================================
def add_indices(img):
    nir      = img.select("B8")
    red      = img.select("B4")
    green    = img.select("B3")
    rededge  = img.select("B5")

    ndvi     = nir.subtract(red).divide(nir.add(red)).rename("NDVI")
    evi      = nir.subtract(red).divide(
        nir.add(red.multiply(6)).add(1)
    ).multiply(2.5).rename("EVI")
    ndre     = nir.subtract(rededge).divide(nir.add(rededge)).rename("NDRE")
    gndvi    = nir.subtract(green).divide(nir.add(green)).rename("GNDVI")
    ciredge  = nir.divide(rededge).subtract(1).rename("CIrededge")

    return img.addBands([ndvi, evi, ndre, gndvi, ciredge])

# ============================================================
# SCL CLOUD MASK  (same logic as crop-yield pipeline)
# ============================================================
CLEAR_SCL_CLASSES = [4, 5, 6]   # vegetation, bare soil, water

def apply_scl_cloud_mask(img):
    scl = img.select("SCL")
    clear_mask = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6))
    return img.updateMask(clear_mask)

def add_ndvi_cloud_masked(img):
    """Cloud-mask then compute NDVI – used for Kalman S2 branch."""
    masked = apply_scl_cloud_mask(img)
    ndvi   = masked.normalizedDifference(["B8", "B4"]).rename("NDVI")
    return masked.addBands(ndvi)

# ============================================================
# SENTINEL-1 HELPERS
# ============================================================
def add_s1_features(img):
    vh_linear    = img.expression("pow(10, vh / 10)", {"vh": img.select("VH")}).rename("VH_linear")
    vv_linear    = img.expression("pow(10, vv / 10)", {"vv": img.select("VV")}).rename("VV_linear")
    soil_moisture = img.select("VH").divide(img.select("VV")).rename("soil_moisture")
    return img.addBands([vh_linear, vv_linear, soil_moisture])

def add_rvi(img):
    """Radar Vegetation Index (same as crop-yield pipeline)."""
    vv  = img.select("VV")
    vh  = img.select("VH")
    rvi = vh.multiply(4).divide(vv.add(vh)).rename("RVI")
    return img.addBands(rvi)

# ============================================================
# KALMAN FUSION  (pixel-level; mirrors crop-yield pipeline)
# ============================================================
def kalman_fuse_ndvi(s2_collection, s1_collection, start, end, geom):
    """
    Blend cloud-masked Sentinel-2 NDVI with Sentinel-1 RVI-derived
    NDVI using a Kalman-style weighting identical to the crop-yield
    pipeline.

    The S2 weight is proportional to the S1 noise variance, and the
    S1 weight is proportional to the S2 inter-image variance – so that
    whichever source has lower uncertainty contributes more.

    Parameters
    ----------
    s2_collection : ee.ImageCollection  S2_SR_HARMONIZED, pre-filtered to stage dates
    s1_collection : ee.ImageCollection  S1_GRD,           pre-filtered to stage dates
    start, end    : str                 ISO date strings for this stage
    geom          : ee.Geometry         point / polygon for clipping

    Returns
    -------
    fused_ndvi : ee.Image  – single-band "NDVI_fused" image
    """
    # ── S2 branch: cloud-masked NDVI ─────────────────────────
    s2_window = (
        s2_collection
        .filterDate(start, end)
        .map(add_ndvi_cloud_masked)
        .select("NDVI")
    )
    s2_ndvi_median = s2_window.median().clip(geom)

    # S2 variance across the stage window (uncertainty estimate)
    s2_var = s2_window.reduce(ee.Reducer.variance()).rename("s2_var").clip(geom)
    # Guard against zero / masked variance (9999 → weight → 0)
    s2_var_safe = s2_var.where(s2_var.lte(0), 9999).unmask(9999)

    # ── S1 branch: RVI → proxy NDVI ──────────────────────────
    s1_rvi = (
        s1_collection
        .filterDate(start, end)
        .map(add_rvi)
        .select("RVI")
        .median()
        .clip(geom)
    )
    # Scale RVI [0, 1] as an NDVI proxy
    s1_ndvi = s1_rvi.unitScale(0, 1).rename("NDVI")
    s1_var  = ee.Image.constant(S1_SPECKLE_VAR)   # fixed noise model

    # ── Kalman blend weights ──────────────────────────────────
    denom = s1_var.add(s2_var_safe)
    w_s2  = s1_var.divide(denom)       # high when S2 uncertainty is LOW
    w_s1  = s2_var_safe.divide(denom)  # high when S1 uncertainty is LOW

    fused = (
        w_s2.multiply(s2_ndvi_median)
            .add(w_s1.multiply(s1_ndvi))
    ).rename("NDVI_fused")

    return fused.clip(geom)


def safe_get(d, key):
    return ee.Algorithms.If(d.contains(key), d.get(key), None)

OPTICAL_INDICES = ["NDVI", "EVI", "NDRE", "GNDVI", "CIrededge"]

# ============================================================
# EXTRACTION FUNCTION
# ============================================================
def extract_plot(idx, row):
    geom     = ee.Geometry.Point(row.geometry.x, row.geometry.y)
    features = []

    # ----------------------------------------------------------
    # STATIC FEATURES (extracted once per plot)
    # ----------------------------------------------------------
    static_features = {
        "soil_organic_carbon": safe_get(
            soilgrids_soc.reduceRegion(ee.Reducer.mean(), geom, 250),
            "soc_0-5cm_mean"
        ),
        "clay_content": safe_get(
            soilgrids_clay.reduceRegion(ee.Reducer.mean(), geom, 250),
            "clay_0-5cm_mean"
        ),
        "elevation": safe_get(
            elevation.reduceRegion(ee.Reducer.mean(), geom, 30),
            "elevation"
        ),
        "slope": safe_get(
            slope.reduceRegion(ee.Reducer.mean(), geom, 30),
            "slope"
        ),
    }

    # ----------------------------------------------------------
    # PRE-BUILD STAGE-LEVEL S1 COLLECTION ONCE
    # (filtered to the full season; further filtered per stage)
    # ----------------------------------------------------------
    s1_full = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(geom)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
    )

    for stage, (start, end) in STAGES.items():

        record = {
            "location":     f"plot_{idx}",
            "crop":         row[CROP_COL],
            "growth_stage": stage,
            "plot_area_m2": row[AREA_COL],
        }
        record.update(static_features)

        # ------------------------------------------------------
        # Sentinel-2 collection for this stage
        # ------------------------------------------------------
        s2_stage = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(geom)
            .filterDate(start, end)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
            .map(add_indices)
        )

        record["S2_count"] = s2_stage.size()

        # -- Standard optical VI means & AUC -------------------
        for idx_name in OPTICAL_INDICES:
            record[f"{idx_name}_mean"] = safe_get(
                s2_stage.select(idx_name).mean()
                .reduceRegion(ee.Reducer.mean(), geom, 10),
                idx_name
            )
            record[f"{idx_name}_auc"] = safe_get(
                s2_stage.select(idx_name).sum()
                .reduceRegion(ee.Reducer.mean(), geom, 10),
                idx_name
            )

        # ------------------------------------------------------
        # Kalman-fused NDVI  ← NEW COLUMN
        # ------------------------------------------------------
        # The full S2 collection (with SCL masking) is used for
        # the Kalman variance estimate; the S1 collection supplies
        # the RVI-based proxy.  This mirrors the crop-yield
        # pipeline exactly so training & inference use identical
        # feature values.
        s2_full_stage = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(geom)
            .filterDate(start, end)
        )

        fused_img = kalman_fuse_ndvi(s2_full_stage, s1_full, start, end, geom)

        record["NDVI_fused_mean"] = safe_get(
            fused_img.reduceRegion(ee.Reducer.mean(), geom, 10),
            "NDVI_fused"
        )
        record["NDVI_fused_auc"] = safe_get(
            # AUC proxy: mean × number of S2 observations in window
            fused_img.multiply(s2_full_stage.size())
            .reduceRegion(ee.Reducer.mean(), geom, 10),
            "NDVI_fused"
        )

        # ------------------------------------------------------
        # Sentinel-1 – VH, VV, soil moisture
        # ------------------------------------------------------
        s1_stage = (
            s1_full
            .filterDate(start, end)
            .map(add_s1_features)
        )

        record["VH_mean"] = safe_get(
            s1_stage.select("VH_linear").mean()
            .reduceRegion(ee.Reducer.mean(), geom, 10),
            "VH_linear"
        )
        record["VV_mean"] = safe_get(
            s1_stage.select("VV_linear").mean()
            .reduceRegion(ee.Reducer.mean(), geom, 10),
            "VV_linear"
        )
        record["soil_moisture_mean"] = safe_get(
            s1_stage.select("soil_moisture").mean()
            .reduceRegion(ee.Reducer.mean(), geom, 10),
            "soil_moisture"
        )
        record["soil_moisture_std"] = safe_get(
            s1_stage.select("soil_moisture")
            .reduce(ee.Reducer.stdDev())
            .reduceRegion(ee.Reducer.mean(), geom, 10),
            "soil_moisture"
        )

        # ------------------------------------------------------
        # ERA5 WEATHER
        # ------------------------------------------------------
        era5 = (
            ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
            .filterBounds(geom)
            .filterDate(start, end)
        )

        record["rainfall"] = safe_get(
            era5.select("total_precipitation_sum").sum()
            .reduceRegion(ee.Reducer.mean(), geom, 1000),
            "total_precipitation_sum"
        )
        record["temp"] = safe_get(
            era5.select("temperature_2m").mean()
            .reduceRegion(ee.Reducer.mean(), geom, 1000),
            "temperature_2m"
        )

        features.append(ee.Feature(None, record))

    return features

# ============================================================
# BATCH PROCESSING
# ============================================================
dfs = []

for start in range(0, len(gdf), BATCH_SIZE):
    end_idx = min(start + BATCH_SIZE, len(gdf))
    print(f"📦 Processing plots {start}–{end_idx - 1}")

    feats = []
    for i in range(start, end_idx):
        feats.extend(extract_plot(i, gdf.iloc[i]))

    fc = ee.FeatureCollection(feats)

    df = ee.data.computeFeatures({
        "expression": fc,
        "fileFormat": "PANDAS_DATAFRAME",
    })

    dfs.append(df)

# ============================================================
# FINAL OUTPUT
# ============================================================
out_df = pd.concat(dfs, ignore_index=True)

# Enforce stage ordering
out_df["growth_stage"] = pd.Categorical(
    out_df["growth_stage"],
    ["early", "mid", "late"],
    ordered=True,
)

out_df = out_df.sort_values(["growth_stage", "location"]).reset_index(drop=True)

out_df.to_csv(OUTPUT_CSV, index=False)

print("✅ Cloud-robust dataset exported with soil, topography, moisture, and Kalman-fused NDVI")
print(f"📁 Output saved to: {OUTPUT_CSV}")

# Show new columns clearly
new_cols = [c for c in out_df.columns if "fused" in c.lower()]
print(f"\n🆕 New fused NDVI columns added: {new_cols}")
print(out_df[["location", "growth_stage"] + new_cols].head(9).to_string(index=False))
