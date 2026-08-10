# DentaVisionScanner – 3D Reconstruction

This project contains two different 3D reconstruction pipelines:

- Pipeline 1 (src/) – Custom pairwise pipeline implemented with OpenCV.
- Pipeline 2 (src_v2/) – Reconstruction pipeline based on OpenMVG, with OpenMVG handling feature extraction, matching, geometric filtering, Structure-from-Motion, and sparse reconstruction.

## Requirements

- Python 3.11–3.12 recommended

For Pipeline 2, you additionally need a compiled installation of OpenMVG.

## Install dependencies

```bash
pip install -r requirements.txt
```

## Configuration

Create a ```.env``` file in the reconstruction folder.

The following paths are required:

```bash
INPUT_IMAGES_DIRECTORY=C:\path\to\your\images
OPENMVG_ROOT=C:\path\to\your\openMVG
```

```INPUT_IMAGES_DIRECTORY```

Path to the directory containing the input images used for reconstruction.

Example:

```
INPUT_IMAGES_DIRECTORY=C:\Users\username\DentaVisionScanner\reconstruction\data\test_dataset
```

```OPENMVG_ROOT```

Path to the root directory of the OpenMVG source/build tree.

Example:
```
OPENMVG_ROOT=C:\dev\openMVG
```

The expected OpenMVG structure is:
```
OPENMVG_ROOT/
├── src/
│   └── openMVG/
│       └── exif/
│           └── sensor_width_database/
│               └── sensor_width_camera_database.txt
│
└── build/
    └── Windows-AMD64-/
        └── Release/
            ├── openMVG_main_SfMInit_ImageListing.exe
            ├── openMVG_main_ComputeFeatures.exe
            ├── openMVG_main_ComputeMatches.exe
            ├── openMVG_main_GeometricFilter.exe
            ├── openMVG_main_SfM.exe
            └── ...
```
For information about compiling OpenMVG, follow the official build instructions:

OpenMVG – [Building the software](https://github.com/openMVG/openMVG/blob/develop/BUILD.md)

## Running the reconstruction

### Pipeline 1 – Custom OpenCV pipeline
Once the Python dependencies are installed, the pipeline can be run directly:
```
python main.py
```
No OpenMVG installation or additional configuration is required for this pipeline beyond the normal project configuration.

### Pipeline 2 – OpenMVG pipeline

This pipeline delegates most of the SfM processing to OpenMVG.

Before running this pipeline, make sure that:

1. OpenMVG has been compiled.
2. ```OPENMVG_ROOT``` in .env points to the OpenMVG root directory.
3. ```INPUT_IMAGES_DIRECTORY``` points to the image dataset.
4. The OpenMVG sensor-width database exists at the expected location.

The OpenMVG pipeline uses OpenMVG's sensor-width database during image listing to obtain camera information when available. The camera model is configured in:
```
src_v2/config/settings.py
```
The current configuration uses:
```
CAMERA_MODEL = 3
```
which corresponds to OpenMVG's Pinhole radial 3 camera model.

## Output

Reconstruction results are written to:

```
output/
```

The OpenMVG pipeline uses a dedicated subdirectory:

```
output/
└── openmvg/
```

This directory contains intermediate OpenMVG files as well as reconstruction results generated during the pipeline.

In particular, the OpenMVG reconstruction produces sparse point-cloud files and a final point cloud that can be visualized by the project.

The output/ directory may also contain results from other experiments and pipelines, so **do not delete the entire ```output/``` directory when cleaning OpenMVG intermediate files.**

## Structure
The current high-level structure is:
```
reconstruction/
│
├── data/
├── output/
├── main.py
├── requirements.txt
├── .env
│
├── src/
│   └── ...
│
└── src_v2/
    └── ...
```
The internal structure of ```src/``` and ```src_v2/``` are currently documented below for reference:

```
src/
    │
    ├── __init__.py
    │
    ├── config/
    │   ├── __init__.py
    │   ├── settings.py
    │
    ├── core/
    │   ├── __init__.py
    │   ├── frame.py
    │   ├── match_result.py
    │   └── reconstruction.py
    │
    ├── pipeline/
    │   ├── __init__.py
    │   ├── camera.py
    │   ├── features.py
    │   ├── matching.py
    │   ├── pose.py
    │   ├── triangulation.py
    │   └── pointcloud.py
    │
    ├── visualization/
    │   ├── __init__.py
    │   └── visualizer.py
    │
    └── utils/
        └── __init__.py
```
---
```
src_v2/
│
├── __init__.py
│
├── config/
│   ├── __init__.py
│   └── settings.py
│
├── reconstruction/
│   ├── __init__.py
│   ├── openmvg.py
│   ├── openmvs.py
│   └── pipeline.py
│
└── utils/
    ├── __init__.py
    └── process.py
```