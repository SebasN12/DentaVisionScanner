import os
from pathlib import Path

from dotenv import load_dotenv


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")


# ---------------------------------------------------------------------------
# External paths
# ---------------------------------------------------------------------------

#OpenMVG

INPUT_IMAGES_DIRECTORY = Path(
    os.environ["INPUT_IMAGES_DIRECTORY"]
)

OPENMVG_ROOT = Path(
    os.environ["OPENMVG_ROOT"]
)

OPENMVG_BIN_DIRECTORY = (
    OPENMVG_ROOT
    / "build"
    / "Windows-AMD64-"
    / "Release"
)

SENSOR_WIDTH_DATABASE = (
    OPENMVG_ROOT
    / "src"
    / "openMVG"
    / "exif"
    / "sensor_width_database"
    / "sensor_width_camera_database.txt"
)

# OpenMVS

OPENMVS_ROOT = Path(
    os.environ["OPENMVS_ROOT"]
)


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

OUTPUT_DIRECTORY = PROJECT_ROOT / "output"

OPENMVG_OUTPUT_DIRECTORY = OUTPUT_DIRECTORY / "openmvg"

OPENMVS_OUTPUT_DIRECTORY = OUTPUT_DIRECTORY / "openmvs"


# ---------------------------------------------------------------------------
# OpenMVG feature extraction
# ---------------------------------------------------------------------------

DESCRIBER_METHOD = "SIFT"

DESCRIBER_PRESET = "NORMAL"

NUM_THREADS = 16


# ---------------------------------------------------------------------------
# OpenMVG matching
# ---------------------------------------------------------------------------

MATCH_RATIO = 0.8


# ---------------------------------------------------------------------------
# OpenMVG SfM
# ---------------------------------------------------------------------------

SFM_ENGINE = "INCREMENTAL"

CAMERA_MODEL = 3

REFINE_INTRINSICS = "ADJUST_ALL"


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

SHOW_WINDOWS = True