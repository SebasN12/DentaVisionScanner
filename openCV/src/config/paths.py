from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = PROJECT_ROOT / "data" / "test_dataset"

OUTPUT_PATH = PROJECT_ROOT / "output"

FEATURES_OUTPUT = OUTPUT_PATH / "features"
MATCHES_OUTPUT = OUTPUT_PATH / "matches"
INLIERS_OUTPUT = OUTPUT_PATH / "inliers"