# DentaVisionScanner – 3D Reconstruction

This project contains two different 3D reconstruction pipelines:

- Pipeline 1 (`src/`) – Pairwise pipeline.
* Pipeline 2 (`src_v2/`) – Reconstruction pipeline based on OpenMVG and OpenMVS, with OpenMVG handling the SfM stage and OpenMVS generating the final dense point cloud.

## Requirements

- Python 3.11–3.12 recommended

For Pipeline 2, you additionally need a compiled installation of OpenMVG and the precompiled OpenMVS binaries.

The current Pipeline 2 configuration is designed around the directory structures of the precompiled Windows x64 distributions for OpenMVS used during development. Manually compiled installations may require changes to the configured paths and are not currently guaranteed to be supported.

OpenMVS is used for optional dense reconstruction. The sparse reconstruction stage can be run independently using OpenMVG.

The IGEV stereo pipeline requires a CUDA-capable NVIDIA GPU. The current implementation uses PyTorch with CUDA and does not support CPU inference.

OpenStereo itself does not need to be modified except for disabling, by commenting it out, the optional FoundationStereo import when FoundationStereo is not being used. This is required because the original import introduces additional dependencies that are not required by the IGEV pipeline.

## Install dependencies

```bash
pip install -r requirements.txt
```
The IGEV pipeline additionally requires PyTorch and TorchVision with CUDA support.

The following configuration was tested during development:
```
pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cu132
```

The cu132 build used above is not specific to a particular GPU model. It corresponds to the CUDA-enabled PyTorch build used during development.

Depending on the operating system, NVIDIA GPU, drivers, and supported CUDA version, a different PyTorch installation command may be required. Use the official PyTorch installation selector to choose the appropriate configuration:

[PyTorch – Start Locally](https://pytorch.org/get-started/locally/)

The exact PyTorch versions above are the versions tested with the current OpenStereo and IGEV integration.

## Configuration

Both `src/config` and `src_v2/config` contain different configurations that depend on the dataset to be used. Configurations for palm desert and middlebury are available, but commented. Remember to check the camera configuration, dataset path and in pair_tests.py there is an import that must be changed depending on the dataset being used.

Create a `.env` file in the reconstruction folder.

The following paths are required:

```bash
INPUT_IMAGES_DIRECTORY=C:\path\to\your\images
OPENMVG_ROOT=C:\path\to\your\openMVG
OPENMVS_ROOT=C:\path\to\your\openMVS
OPENSTEREO_PATH=C:\path\to\OpenStereo
IGEV_CHECKPOINT=C:\path\to\igev\sceneflow.pth
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

---

`OPENSTEREO_PATH`

Path to the OpenStereo root directory.

Example:

```
OPENSTEREO_PATH=C:\dev\OpenStereo
```
The OpenStereo directory must contain the `stereo/` package and the IGEV configuration files.

The project currently uses the IGEV SceneFlow configuration:

```
OPENSTEREO_PATH/cfgs/igev/igev_sceneflow_amp.yaml
```

Clone OpenStereo from the official repository:

OpenStereo – [GitHub repository](https://github.com/XiandaGuo/OpenStereo)

The version currently used by the project can be cloned with:
```
git clone --branch v2 https://github.com/XiandaGuo/OpenStereo.git
```
---

`IGEV_CHECKPOINT`

Path to the pretrained IGEV SceneFlow checkpoint.

Example:

```
IGEV_CHECKPOINT=C:\dev\pretrained_models\igev\sceneflow.pth
```

The checkpoint does not need to be located inside the OpenStereo directory.

Download the pretrained IGEV SceneFlow checkpoint. You can follow the instructions from IGEV repository:

IGEV repository – [Pretrained models](https://github.com/gangweiX/IGEV)

The project currently uses the `sceneflow.pth` checkpoint for IGEV.

## Running the reconstruction

### Pipeline 1 – Pairwise pipeline (SIFT and SGBM with OpenCV)
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

### Pipeline 3 – IGEV stereo reconstruction

This pipeline uses the IGEV stereo matching model provided by OpenStereo to generate a dense disparity map from a rectified stereo pair.

The pipeline performs:

- Stereo disparity estimation with IGEV.
- Disparity validation.
- Depth reconstruction.
- 3D point cloud reconstruction.
- Point cloud visualization.

Before running the pipeline, make sure that:

1. OpenStereo is available at the path configured by `OPENSTEREO_PATH`.
2. The IGEV pretrained checkpoint is available at the path configured by `IGEV_CHECKPOINT`.
3. The required Python dependencies are installed.
4. CUDA is available.

## OpenStereo
The project uses OpenStereo as an external dependency for IGEV stereo matching.

The OpenStereo source code is kept separately from the DentaVisionScanner repository and its location is configured through `OPENSTEREO_PATH`.

Only the following modifications to the original OpenStereo code are required by the current project:

- The optional FoundationStereo import in stereo/modeling/__init__.py is commented out because FoundationStereo is not used by the project and requires additional dependencies.
- The IGEV inference output is cropped in the DentaVisionScanner integration to restore the original input image dimensions after OpenStereo preprocessing padding.

The pretrained IGEV checkpoint is stored separately and its path is configured through `IGEV_CHECKPOINT`.


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