#!/usr/bin/env python
# ============================================================
# Diagnostic Script: Check S1/S2 Data Availability & Fusion
# ============================================================

import os
import ee
import geemap
import numpy as np
from contextlib import contextmanager
import sys

# ============================================================
# CONTEXT MANAGER TO SUPPRESS VERBOSE OUTPUT
# ============================================================

@contextmanager
def suppress_output():
    """Suppress stdout to hide verbose geemap logging."""
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

BASE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
AOI_SHP = os.path.join(BASE_DIRECTORY, "Farm", "witz_farm.shp")
OUTPUT_DIRECTORY = os.path.join(BASE_DIRECTORY, "output_dfh")

AOI = geemap.shp_to_ee(AOI_SHP).geometry()

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

# ============================================================
# CHECK DATA AVAILABILITY
# ============================================================

def add_ndvi_no_mask(img):
    ndvi = img.normalizedDifference(["B8", "B4"]).rename("NDVI")
    return img.addBands(ndvi)

def add_rvi(img):
    vv = img.select("VV")
    vh = img.select("VH")
    rvi = vh.multiply(4).divide(vv.add(vh)).rename("RVI")
    return img.addBands(rvi)

s2_full = (
    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterBounds(AOI)
    .filterDate(START_DATE, END_DATE)
    .map(add_ndvi_no_mask)
)

s1 = (
    ee.ImageCollection("COPERNICUS/S1_GRD")
    .filterBounds(AOI)
    .filterDate(START_DATE, END_DATE)
    .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
    .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
    .map(add_rvi)
)

print("\n" + "="*70)
print("DIAGNOSTIC: DATA AVAILABILITY CHECK")
print("="*70)

for stage, (start, end) in GROWTH_STAGES.items():
    print(f"\n📅 {stage.upper()} Stage ({start} to {end}):")
    print("-" * 70)
    
    # Count Sentinel-2 images
    s2_count = s2_full.filterDate(start, end).size().getInfo()
    print(f"   Sentinel-2 images available: {s2_count}")
    
    # Count Sentinel-1 images
    s1_count = s1.filterDate(start, end).size().getInfo()
    print(f"   Sentinel-1 images available: {s1_count}")
    
    # Get S2 NDVI
    s2_ndvi = s2_full.filterDate(start, end).select("NDVI").median().clip(AOI)
    
    # Get S1 RVI
    s1_rvi = s1.filterDate(start, end).select("RVI").median().clip(AOI)
    
    # Check data coverage
    s2_stats = s2_ndvi.reduceRegion(
        reducer=ee.Reducer.count().combine(
            ee.Reducer.mean(), sharedInputs=True
        ).combine(
            ee.Reducer.minMax(), sharedInputs=True
        ),
        geometry=AOI,
        scale=100,
        maxPixels=1e9
    ).getInfo()
    
    s1_stats = s1_rvi.reduceRegion(
        reducer=ee.Reducer.count().combine(
            ee.Reducer.mean(), sharedInputs=True
        ).combine(
            ee.Reducer.minMax(), sharedInputs=True
        ),
        geometry=AOI,
        scale=100,
        maxPixels=1e9
    ).getInfo()
    
    print(f"\n   📊 Sentinel-2 NDVI Statistics:")
    print(f"      Valid pixels: {s2_stats.get('NDVI_count', 0)}")
    print(f"      Mean NDVI: {s2_stats.get('NDVI_mean', 'N/A')}")
    print(f"      Min NDVI: {s2_stats.get('NDVI_min', 'N/A')}")
    print(f"      Max NDVI: {s2_stats.get('NDVI_max', 'N/A')}")
    
    print(f"\n   📊 Sentinel-1 RVI Statistics:")
    print(f"      Valid pixels: {s1_stats.get('RVI_count', 0)}")
    print(f"      Mean RVI: {s1_stats.get('RVI_mean', 'N/A')}")
    print(f"      Min RVI: {s1_stats.get('RVI_min', 'N/A')}")
    print(f"      Max RVI: {s1_stats.get('RVI_max', 'N/A')}")
    
    # Create mask comparison
    s2_mask = s2_ndvi.mask()
    s2_masked_area = s2_mask.Not().reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=AOI,
        scale=100,
        maxPixels=1e9
    ).getInfo()
    
    total_pixels = s2_mask.reduceRegion(
        reducer=ee.Reducer.count(),
        geometry=AOI,
        scale=100,
        maxPixels=1e9
    ).getInfo()
    
    masked_count = s2_masked_area.get('NDVI', 0)
    total_count = total_pixels.get('NDVI', 1)
    gap_percentage = (masked_count / total_count * 100) if total_count > 0 else 0
    
    print(f"\n   🔍 Gap Analysis:")
    print(f"      S2 masked pixels (gaps): {masked_count}")
    print(f"      Total pixels: {total_count}")
    print(f"      Gap percentage: {gap_percentage:.2f}%")
    
    if gap_percentage < 1:
        print(f"      ⚠️  WARNING: Very few gaps (<1%) - fusion may not be visible!")
    elif s1_count == 0:
        print(f"      ⚠️  WARNING: No S1 data available - fusion impossible!")

print("\n" + "="*70)
print("DIAGNOSTIC: EXPORTING COMPARISON MAPS")
print("="*70)

# Export comparison maps for visual inspection
for stage, (start, end) in GROWTH_STAGES.items():
    print(f"\n🗺️  Processing {stage.upper()} stage...")
    
    # S2 NDVI
    s2_ndvi = s2_full.filterDate(start, end).select("NDVI").median().clip(AOI)
    
    # S1 RVI scaled to NDVI range
    s1_rvi = s1.filterDate(start, end).select("RVI").median().clip(AOI)
    s1_ndvi_proxy = s1_rvi.unitScale(0, 1).rename("NDVI")
    
    # S2 mask (shows where gaps are)
    s2_mask = s2_ndvi.mask().selfMask().clip(AOI)
    
    # S2-only (unmasked)
    s2_only = s2_ndvi.unmask(0)
    
    # Fused
    fused = s2_ndvi.unmask(s1_ndvi_proxy).unmask(0.3)
    
    # S1-only (for comparison)
    s1_only = s1_ndvi_proxy.unmask(0)
    
    # Export all products
    with suppress_output():
        # S2 NDVI (original with gaps)
        geemap.ee_export_image(
            s2_ndvi,
            os.path.join(OUTPUT_DIRECTORY, f"DIAGNOSTIC_S2_with_gaps_{stage}.tif"),
            scale=10, region=AOI, file_per_band=False
        )
        
        # S2 mask (shows gap locations)
        geemap.ee_export_image(
            s2_mask,
            os.path.join(OUTPUT_DIRECTORY, f"DIAGNOSTIC_S2_mask_{stage}.tif"),
            scale=10, region=AOI, file_per_band=False
        )
        
        # S1 NDVI proxy
        geemap.ee_export_image(
            s1_only,
            os.path.join(OUTPUT_DIRECTORY, f"DIAGNOSTIC_S1_proxy_{stage}.tif"),
            scale=10, region=AOI, file_per_band=False
        )
        
        # S2-only (unmasked)
        geemap.ee_export_image(
            s2_only,
            os.path.join(OUTPUT_DIRECTORY, f"DIAGNOSTIC_S2_only_{stage}.tif"),
            scale=10, region=AOI, file_per_band=False
        )
        
        # Fused
        geemap.ee_export_image(
            fused,
            os.path.join(OUTPUT_DIRECTORY, f"DIAGNOSTIC_Fused_{stage}.tif"),
            scale=10, region=AOI, file_per_band=False
        )
        
        # Difference map (Fused - S2)
        diff = fused.subtract(s2_only).rename("difference")
        geemap.ee_export_image(
            diff,
            os.path.join(OUTPUT_DIRECTORY, f"DIAGNOSTIC_Difference_{stage}.tif"),
            scale=10, region=AOI, file_per_band=False
        )
    
    print(f"   ✓ Exported diagnostic maps for {stage.upper()}")

print("\n" + "="*70)
print("✅ DIAGNOSTIC COMPLETE")
print("="*70)
print("\nCheck the following files in output_dfh/:")
print("  • DIAGNOSTIC_S2_with_gaps_*.tif (shows S2 NDVI with data gaps)")
print("  • DIAGNOSTIC_S2_mask_*.tif (white = valid data, black/nodata = gaps)")
print("  • DIAGNOSTIC_S1_proxy_*.tif (S1-derived NDVI)")
print("  • DIAGNOSTIC_S2_only_*.tif (S2 with gaps filled with 0)")
print("  • DIAGNOSTIC_Fused_*.tif (S2 + S1 fusion)")
print("  • DIAGNOSTIC_Difference_*.tif (difference between fused and S2-only)")
print("\nInterpretation:")
print("  • If gap percentage < 1%: S2 has nearly complete coverage, fusion minimal")
print("  • If S1 count = 0: No S1 data available, fusion impossible")
print("  • If difference map is all zeros: Fusion is not working")
print("="*70)
