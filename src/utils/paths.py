"""Single source of truth for every file location in the project.

All paths are built from this file's own location, so scripts work no matter
which folder the terminal is in.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

RAW_DATA_FILE = RAW_DIR / "Online_Retail.xlsx"
CLEAN_DATA_FILE = PROCESSED_DIR / "transactions_clean.parquet"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODEL_REPORTS_DIR = REPORTS_DIR / "model_reports"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
PREPROCESSORS_DIR = ARTIFACTS_DIR / "preprocessors"
