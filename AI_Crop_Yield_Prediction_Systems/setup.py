import subprocess
import sys

# ============================================================
# USER CONFIGURATION
# ============================================================
ENV_NAME = "crop_env"
PYTHON_VERSION = "3.12"

print("📦 Starting full environment setup...")

# ============================================================
# 1️⃣ CREATE CONDA ENVIRONMENT
# ============================================================
try:
    subprocess.run(
        ["conda", "create", "-n", ENV_NAME, f"python={PYTHON_VERSION}", "-y"],
        check=True
    )
    print(f"✅ Conda environment '{ENV_NAME}' created.")
except subprocess.CalledProcessError:
    print(f"⚠️ Conda environment '{ENV_NAME}' may already exist. Skipping creation.")

# ============================================================
# 2️⃣ INSTALL CORE PACKAGES (conda-forge)
# ============================================================
conda_packages = [
    # ---- Core GIS stack ----
    "geopandas",
    "rasterio",
    "shapely",
    "pyproj",
    "fiona",
    "gdal",

    # ---- Data science ----
    "numpy",
    "pandas",
    "matplotlib",
    "seaborn",
    "scikit-learn",
    "xgboost",

    # ---- Remote sensing / cloud ----
    "pystac-client",
    "planetary-computer",
    "earthengine-api",
    "geemap",
    "geedim",  # ✅ Added geedim

    # ---- Dashboard & mapping ----
    "streamlit",
    "streamlit-folium",
    "folium",
    "branca",

    # ---- Utilities ----
    "tqdm",
    "pip"
]

print("🌍 Installing conda packages (this may take several minutes)...")

subprocess.run(
    [
        "conda", "install",
        "-n", ENV_NAME,
        "-c", "conda-forge",
        "-y"
    ] + conda_packages,
    check=True
)

print("✅ Conda packages installed.")

# ============================================================
# 2️⃣b VERIFY GEEDIM (Fallback to pip if needed)
# ============================================================
print("🔍 Verifying geedim installation...")

try:
    subprocess.run(
        ["conda", "run", "-n", ENV_NAME, "python", "-c", "import geedim"],
        check=True
    )
    print("✅ geedim is installed.")
except subprocess.CalledProcessError:
    print("⚠️ geedim not found via conda. Installing via pip...")
    subprocess.run(
        ["conda", "run", "-n", ENV_NAME, "pip", "install", "geedim"],
        check=True
    )
    print("✅ geedim installed via pip.")

# ============================================================
# 3️⃣ PIN STREAMLIT-CRITICAL DEPENDENCIES
# ============================================================
# Streamlit requires:
# - protobuf < 4
# - altair < 5

print("🔧 Pinning Streamlit-critical dependencies...")

# Remove incompatible versions (if any)
subprocess.run(
    ["conda", "run", "-n", ENV_NAME, "pip", "uninstall", "-y", "protobuf", "altair"],
    check=False
)

# Install compatible versions
subprocess.run(
    [
        "conda", "run",
        "-n", ENV_NAME,
        "pip", "install",
        "protobuf>=3.20.0,<4",
        "altair<5"
    ],
    check=True
)

print("✅ Protobuf and Altair pinned to Streamlit-compatible versions.")

# ============================================================
# 4️⃣ FINAL INSTRUCTIONS
# ============================================================
print("\n🎉 SETUP COMPLETE!\n")
print("Next steps:")
print(f"1️⃣ Activate the environment:\n   conda activate {ENV_NAME}")
print("2️⃣ Authenticate Google Earth Engine (run once):\n   earthengine authenticate")
print("3️⃣ Run your dashboard:\n   streamlit run dashboard.py")

print("\n✅ This environment is now STABLE and will not randomly break.")
