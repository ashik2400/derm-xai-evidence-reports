"""All project paths and fixed settings in one place.

Every script imports from here, so if a folder moves we change one line.
"""
from pathlib import Path

# Repo root = the folder that contains src/
ROOT = Path(__file__).resolve().parents[1]

# --- Raw data (read-only, versioned by DVC) ---
RAW_DIR = ROOT / "data" / "raw" / "release_v0"
META_DIR = RAW_DIR / "meta"
IMAGES_DIR = RAW_DIR / "images"
META_CSV = META_DIR / "meta.csv"
SPLIT_FILES = {
    "train": META_DIR / "train_indexes.csv",
    "valid": META_DIR / "valid_indexes.csv",
    "test": META_DIR / "test_indexes.csv",
}

# --- Generated files (we write here, never into raw) ---
PROCESSED_DIR = ROOT / "data" / "processed"
CASES_CSV = PROCESSED_DIR / "cases.csv"

# --- Reports and figures ---
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# --- MLflow: one local SQLite file (gitignored) ---
MLFLOW_TRACKING_URI = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"

# --- Fixed settings ---
SEED = 42
