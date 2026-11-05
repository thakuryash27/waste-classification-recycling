"""Central configuration for the Waste Classification project."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data locations
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "garbage_classification"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Output locations
MODELS_DIR = PROJECT_ROOT / "models"
PLOTS_DIR = PROJECT_ROOT / "outputs" / "plots"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"

# Image and training settings
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

# Dataset split (must add up to 1.0)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
