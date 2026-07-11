# Contains file paths to avoid hadrcoding file paths throughout the project

from pathlib import Path

# Project Root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Dataset
RAW_DATASET = RAW_DATA_DIR / "processed.cleveland.data"

# Models
MODEL_DIR = PROJECT_ROOT / "models"

# Logs
LOG_DIR = PROJECT_ROOT / "logs"

# Reports
REPORT_DIR = PROJECT_ROOT / "reports"

# Create directories if they don't exist
for directory in [
    PROCESSED_DATA_DIR,
    MODEL_DIR,
    LOG_DIR,
    REPORT_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)
