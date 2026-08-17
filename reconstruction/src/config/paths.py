from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# desert dataset
# DATASET_PATH = PROJECT_ROOT / "data" / "test_dataset"

# middlebury dataset
# # midd1
# DATASET_PATH = PROJECT_ROOT / "data" / "test_dataset" / "middlebury" / "Midd1"

# # midd2
# DATASET_PATH = PROJECT_ROOT / "data" / "test_dataset" / "middlebury" / "Midd2"

# # Monopoly
# DATASET_PATH = PROJECT_ROOT / "data" / "test_dataset" / "middlebury" / "Monopoly"

# # Plastic
DATASET_PATH = PROJECT_ROOT / "data" / "test_dataset" / "middlebury" / "Plastic"

OUTPUT_PATH = PROJECT_ROOT / "output"

FEATURES_OUTPUT = OUTPUT_PATH / "features"
MATCHES_OUTPUT = OUTPUT_PATH / "matches"
INLIERS_OUTPUT = OUTPUT_PATH / "inliers"

RECONSTRUCTION_OUTPUT = OUTPUT_PATH / "reconstruction"