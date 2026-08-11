from pathlib import Path

import shutil

from src_v2.config.settings import (
CAMERA_MODEL,
DESCRIBER_METHOD,
DESCRIBER_PRESET,
INPUT_IMAGES_DIRECTORY,
MATCH_RATIO,
NUM_THREADS,
OPENMVG_OUTPUT_DIRECTORY,
SFM_ENGINE,
SENSOR_WIDTH_DATABASE,
OPENMVS_OUTPUT_DIRECTORY,
)

from src_v2.reconstruction.openmvg import OpenMVG
from src_v2.reconstruction.openmvs import OpenMVS

class ReconstructionPipeline:
    """
    Orchestrates the OpenMVG sparse reconstruction pipeline.
    """

    def __init__(
        self,
        openmvg: OpenMVG,
        openmvs: OpenMVS,
        image_directory: Path = INPUT_IMAGES_DIRECTORY,
        openMVG_output_directory: Path = OPENMVG_OUTPUT_DIRECTORY,
        openMVS_output_directory: Path = OPENMVS_OUTPUT_DIRECTORY,
        clean_output: bool = False,
    ):
        self.openmvg = openmvg
        self.openmvs = openmvs

        self.image_directory = Path(image_directory)
        self.openMVG_output_directory = Path(openMVG_output_directory)

        self.sfm_directory = (
            self.openMVG_output_directory / "sfm"
        )

        self.sfm_data = (
            self.openMVG_output_directory / "sfm_data.json"
        )

        self.matches_file = (
            self.openMVG_output_directory
            / "matches.putative.bin"
        )

        self.geometric_matches_file = (
            self.openMVG_output_directory
            / "matches.f.bin"
        )

        self.color_cloud = (
            self.sfm_directory
            / "cloud_and_poses.ply"
        )

        self.openMVS_output_directory = Path(openMVS_output_directory)

        self.openmvs_scene = (
            self.openMVS_output_directory / "scene.mvs"
        )

        self.undistorted_images_directory = (
            self.openMVS_output_directory / "undistorted"
        )

        self.clean_output = clean_output

    def run(self) -> Path:
        """
        Runs the complete OpenMVG sparse reconstruction pipeline.

        Returns:
            Path to the final colored point cloud.
        """

        if self.clean_output:
            self._clean_openMVG_output_directory()
            self._clean_openMVS_output_directory()

        self._prepare_directories()

        print("\n=== OpenMVG: Image Listing ===")

        self.openmvg.init_image_listing(
            image_directory=self.image_directory,
            openMVG_output_directory=self.openMVG_output_directory,
            sensor_width_database=SENSOR_WIDTH_DATABASE,
            camera_model=CAMERA_MODEL,
        )

        print("\n=== OpenMVG: Feature Extraction ===")

        self.openmvg.compute_features(
            sfm_data=self.sfm_data,
            openMVG_output_directory=self.openMVG_output_directory,
            describer_method=DESCRIBER_METHOD,
            describer_preset=DESCRIBER_PRESET,
            num_threads=NUM_THREADS,
        )

        print("\n=== OpenMVG: Feature Matching ===")

        self.openmvg.compute_matches(
            sfm_data=self.sfm_data,
            output_file=self.matches_file,
            ratio=MATCH_RATIO,
        )

        print("\n=== OpenMVG: Geometric Filtering ===")

        self.openmvg.geometric_filter(
            sfm_data=self.sfm_data,
            input_matches=self.matches_file,
            output_matches=self.geometric_matches_file,
        )

        print("\n=== OpenMVG: Structure from Motion ===")

        self.openmvg.reconstruct(
            sfm_data=self.sfm_data,
            match_directory=self.openMVG_output_directory,
            openMVG_output_directory=self.sfm_directory,
            engine=SFM_ENGINE,
            match_file=self.geometric_matches_file.name,
            camera_model=CAMERA_MODEL,
        )

        print("\n=== OpenMVG: Compute Point Colors ===")

        self.openmvg.compute_color(
            sfm_data=self.sfm_directory / "sfm_data.bin",
            output_file=self.color_cloud,
        )

        print("\n=== OpenMVS: Convert OpenMVG Scene ===")

        self.openmvs.convert_from_openmvg(
            sfm_data=self.sfm_directory / "sfm_data.bin",
            output_file=self.openmvs_scene,
            undistorted_images_directory=(
                self.undistorted_images_directory
            ),
        )

        print("\n=== Reconstruction finished ===")
        print(f"Point cloud: {self.color_cloud}")

        return self.color_cloud

    def _prepare_directories(self) -> None:
        """
        Creates the directories required by the pipeline.
        """

        self.openMVG_output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.sfm_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.openMVS_output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.undistorted_images_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    def _clean_openMVG_output_directory(self) -> None:
        """
        Removes only the OpenMVG pipeline output directory.
        """

        if self.openMVG_output_directory.exists():
            shutil.rmtree(self.openMVG_output_directory)

        self.openMVG_output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )
    
    def _clean_openMVS_output_directory(self) -> None:
        """
        Removes only the OpenMVS pipeline output directory.
        """

        if self.openMVS_output_directory.exists():
            shutil.rmtree(self.openMVS_output_directory)

        self.openMVS_output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )