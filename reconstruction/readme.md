# DentaVisionScanner – 3D Reconstruction

This project contains two different 3D reconstruction pipelines:

- Pipeline 1 (`src/`) – Custom pairwise pipeline implemented with OpenCV.
* Pipeline 2 (`src_v2/`) – Reconstruction pipeline based on OpenMVG and OpenMVS, with OpenMVG handling the SfM stage and OpenMVS generating the final dense point cloud.

## Requirements

- Python 3.11–3.12 recommended

For Pipeline 2, you additionally need a compiled installation of OpenMVG and the precompiled OpenMVS binaries.

The current Pipeline 2 configuration is designed around the directory structures of the precompiled Windows x64 distributions for OpenMVS used during development. Manually compiled installations may require changes to the configured paths and are not currently guaranteed to be supported.

OpenMVS is used for optional dense reconstruction. The sparse reconstruction stage can be run independently using OpenMVG.

## Install dependencies

```bash
pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the reconstruction folder.

The following paths are required:

```bash
INPUT_IMAGES_DIRECTORY=C:\path\to\your\images
OPENMVG_ROOT=C:\path\to\your\openMVG
OPENMVS_ROOT=C:\path\to\your\openMVS
```

`INPUT_IMAGES_DIRECTORY`

Path to the directory containing the input images used for reconstruction.

Example:

```
INPUT_IMAGES_DIRECTORY=C:\Users\username\DentaVisionScanner\reconstruction\data\test_dataset
```

---

`OPENMVG_ROOT`

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

Only the executables required by the current pipeline need to be present. In particular, the pipeline currently uses `openMVG_main_SfMInit_ImageListing.exe`, `openMVG_main_ComputeFeatures.exe`, `openMVG_main_ComputeMatches.exe`, `openMVG_main_GeometricFilter.exe`, `openMVG_main_SfM.exe`, and `openMVG_main_ComputeSfM_DataColor.exe`.

---

`OPENMVS_ROOT`

Path to the directory containing the precompiled OpenMVS Windows x64 binaries.

Example:
```
OPENMVS_ROOT=C:\dev\openMVS
```

The expected OpenMVS structure is:

```
OPENMVS_ROOT/
├── DensifyPointCloud.exe
├── InterfaceCOLMAP.exe
├── InterfaceMetashape.exe
├── InterfaceMVSNet.exe
├── InterfacePolycam.exe
├── ReconstructMesh.exe
├── RefineMesh.exe
├── Tests.exe
├── TextureMesh.exe
├── TransformScene.exe
└── Viewer.exe
```


The project currently uses the precompiled Windows x64 OpenMVS distribution. The OpenMVS executables are expected to be located directly inside `OPENMVS_ROOT`.

Download the required release from the official OpenMVS releases page:

OpenMVS – [Latest releases](https://github.com/cdcseacave/openMVS/releases/latest)

For the current development environment, the Windows x64 package is extracted to a directory such as:

```
C:\dev\openMVS\
```

and its path is configured through:

```
OPENMVS_ROOT=C:\dev\openMVS
```
Only the executables required by the current pipeline need to be present. In particular, the pipeline currently uses `DensifyPointCloud.exe` and the OpenMVG-to-OpenMVS conversion executable.

## Running the reconstruction

### Pipeline 1 – Custom OpenCV pipeline
Once the Python dependencies are installed, the pipeline can be run directly:
```
python main.py
```
No OpenMVG installation or additional configuration is required for this pipeline beyond the normal project configuration.

### Pipeline 2 – OpenMVG pipeline

This pipeline uses OpenMVG for sparse Structure-from-Motion (SfM) reconstruction and optionally uses OpenMVS for dense reconstruction.

The pipeline is divided into three independent stages:

1. Sparse reconstruction with OpenMVG.
2. OpenMVS scene preparation.
3. Dense reconstruction with OpenMVS.

The stages can be executed independently as long as the required output of the previous stage exists.

Before running the pipeline, make sure that:

1. OpenMVG has been compiled.
2. `OPENMVG_ROOT` in `.env` points to the OpenMVG root directory.
3. `OPENMVS_ROOT` in `.env` points to the OpenMVS binary directory.
4. `INPUT_IMAGES_DIRECTORY` points to the image dataset.
5. The OpenMVG sensor-width database exists at the expected location.

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
openmvg/
│   └── sfm/
```

This directory contains intermediate OpenMVG files as well as reconstruction results generated during the pipeline.

The OpenMVG stage produces the sparse reconstruction inside the `openmvg/` directory. The OpenMVS stages use a separate `openmvs/` directory for scene preparation and dense reconstruction.

```
openmvs/
    ├── scene.mvs
    ├── undistorted/
    └── dense/
        └── pointcloud.ply
```

The `openmvs/` directory contains the OpenMVS scene, the undistorted images used for dense reconstruction, and the generated dense point cloud.

The stages can therefore be cleaned independently without removing the results of the other reconstruction stage.

The `output/` directory may also contain results from other experiments and pipelines, so **do not delete the entire `output/` directory when cleaning intermediate files.**

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
The internal structure of `src/` and `src_v2/` are currently documented below for reference:

```
src/
    │
    ├── __init__.py
    │
    ├── config/
    │   ├── __init__.py
    │   ├── camera.py
    │   ├── paths.py
    │   └── settings.py
    │
    ├── core/
    │   ├── __init__.py
    │   ├── frame.py
    │   ├── match_result.py
    │   ├── point_cloud.py
    │   └── triangulation_result.py
    │
    ├── io/
    │   ├── __init__.py
    │   └── point_cloud_writer.py
    │
    ├── optimization/
    │   ├── __init__.py
    │   ├── ba_problem.py
    │   └── bundle_adjustment.py
    │
    ├── pipeline/
    │   ├── __init__.py
    │   ├── camera.py
    │   ├── dense_reconstruction.py
    │   ├── features.py
    │   ├── matching.py
    │   ├── pose.py
    │   ├── reconstructor.py
    │   └── triangulation.py
    │
    └── visualization/
        ├── __init__.py
        └── visualizer.py
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