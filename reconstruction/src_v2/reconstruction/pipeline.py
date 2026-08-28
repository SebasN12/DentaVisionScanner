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
    OPENMVS_CUDA_DEVICE,
    OPENMVS_OUTPUT_DIRECTORY,
)

from src_v2.reconstruction.openmvg import OpenMVG
from src_v2.reconstruction.openmvs import OpenMVS


class ReconstructionPipeline:
    """
    Orchestrates the OpenMVG and OpenMVS reconstruction pipeline.
    """

    def __init__(
        self,
        openmvg: OpenMVG | None = None,
        openmvs: OpenMVS | None = None,
        image_directory: Path = INPUT_IMAGES_DIRECTORY,
        openMVG_output_directory: Path = OPENMVG_OUTPUT_DIRECTORY,
        openMVS_output_directory: Path = OPENMVS_OUTPUT_DIRECTORY,
        clean_output: bool = False,
    ):
        """
        Initializes the reconstruction pipeline.

        Args:
            openmvg: OpenMVG command-line wrapper.
            openmvs: OpenMVS command-line wrapper.
            image_directory: Directory containing the input images.
            openMVG_output_directory: Directory for OpenMVG outputs.
            openMVS_output_directory: Directory for OpenMVS outputs.
            clean_output: If True, the output directory relevant to the
                executed pipeline stage is cleared before starting.
                When run_sparse() is called, only the OpenMVG output
                directory is cleaned. When prepare_dense() is called,
                only the OpenMVS output directory is cleaned. When
                run_dense() is called, only the dense reconstruction
                directory is cleaned. If False, existing outputs are
                preserved.
        """

        self.openmvg = openmvg
        self.openmvs = openmvs

        self.image_directory = Path(image_directory)
        self.openMVG_output_directory = Path(
            openMVG_output_directory
        )

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

        self.openMVS_output_directory = Path(
            openMVS_output_directory
        )

        self.openmvs_scene = (
            self.openMVS_output_directory
            / "scene.mvs"
        )

        self.undistorted_images_directory = (
            self.openMVS_output_directory
            / "undistorted"
        )

        self.dense_directory = (
            self.openMVS_output_directory
            / "dense"
        )

        self.dense_point_cloud = (
            self.dense_directory
            / "pointcloud.ply"
        )

        self.clean_output = clean_output

    def run_sparse(self) -> Path:
        """
        Runs the OpenMVG sparse reconstruction pipeline.

        Returns:
            Path to the final colored sparse point cloud.
        """

        if self.openmvg is None:
            raise RuntimeError(
                "OpenMVG is required for sparse reconstruction."
            )

        if self.clean_output:
            self._clean_openMVG_output_directory()

        self._prepare_openMVG_directories()

        print("\n=== OpenMVG: Image Listing ===")

        self.openmvg.init_image_listing(
            image_directory=self.image_directory,
            output_directory=self.openMVG_output_directory,
            sensor_width_database=SENSOR_WIDTH_DATABASE,
            camera_model=CAMERA_MODEL,
        )

        print("\n=== OpenMVG: Feature Extraction ===")

        self.openmvg.compute_features(
            sfm_data=self.sfm_data,
            output_directory=self.openMVG_output_directory,
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
            output_directory=self.sfm_directory,
            engine=SFM_ENGINE,
            match_file=self.geometric_matches_file.name,
            camera_model=CAMERA_MODEL,
        )

        print("\n=== OpenMVG: Compute Point Colors ===")

        self.openmvg.compute_color(
            sfm_data=self.sfm_directory / "sfm_data.bin",
            output_file=self.color_cloud,
        )

        print("\n=== Sparse reconstruction finished ===")
        print(f"Sparse point cloud: {self.color_cloud}")

        return self.color_cloud
    
    def prepare_dense(self) -> Path:
        """
        Converts the existing OpenMVG sparse reconstruction into
        an OpenMVS scene for dense reconstruction.

        Returns:
            Path to the generated OpenMVS scene.
        """

        if self.openmvs is None:
            raise RuntimeError(
                "OpenMVS is required to prepare the dense reconstruction."
            )

        if self.openmvg is None:
            raise RuntimeError(
                "OpenMVG is required to provide the sparse reconstruction."
            )

        if self.clean_output:
            self._clean_openMVS_output_directory()

        sfm_data_file = self.sfm_directory / "sfm_data.bin"

        if not sfm_data_file.exists():
            raise FileNotFoundError(
                "OpenMVG sparse reconstruction not found. "
                "Run run_sparse() first."
            )

        self._prepare_openMVS_directories()

        print("\n=== OpenMVS: Convert OpenMVG Scene ===")

        self.openmvs.convert_from_openmvg(
            sfm_data=sfm_data_file,
            output_file=self.openmvs_scene,
            undistorted_images_directory=(
                self.undistorted_images_directory
            ),
        )

        print("\n=== OpenMVS scene preparation finished ===")
        print(f"OpenMVS scene: {self.openmvs_scene}")

        return self.openmvs_scene

    def run_dense(self) -> Path:
        """
        Runs OpenMVS dense reconstruction using an existing
        OpenMVS scene.

        Returns:
            Path to the generated dense point cloud.
        """

        if self.openmvs is None:
            raise RuntimeError(
                "OpenMVS is required for dense reconstruction."
            )

        if self.clean_output:
            self._clean_openMVS_dense_directory()

        if not self.openmvs_scene.exists():
            raise FileNotFoundError(
                "OpenMVS scene not found. "
                "Run prepare_dense() first."
            )

        self.dense_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        print("\n=== OpenMVS: Dense Reconstruction ===")

        self.openmvs.densify_point_cloud(
            scene_file=self.openmvs_scene,
            output_file=self.dense_point_cloud,
            cuda_device=OPENMVS_CUDA_DEVICE,
        )

        if not self.dense_point_cloud.exists():
            raise RuntimeError(
                "OpenMVS did not generate the dense point cloud. "
                f"Expected output: {self.dense_point_cloud}"
            )

        print("\n=== Dense reconstruction finished ===")
        print(f"Dense point cloud: {self.dense_point_cloud}")

        return self.dense_point_cloud
    

    def _prepare_openMVG_directories(self) -> None:
        """
        Creates the directories required by the OpenMVG pipeline.
        """

        self.openMVG_output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.sfm_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    def _prepare_openMVS_directories(self) -> None:
        """
        Creates the directories required by the OpenMVS pipeline.
        """

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
            shutil.rmtree(
                self.openMVG_output_directory
            )

        self.openMVG_output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    def _clean_openMVS_output_directory(self) -> None:
        """
        Removes only the OpenMVS pipeline output directory.
        """

        if self.openMVS_output_directory.exists():
            shutil.rmtree(
                self.openMVS_output_directory
            )

        self.openMVS_output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )
    
    def _clean_openMVS_dense_directory(self) -> None:
        """
        Removes only the OpenMVS dense reconstruction output.
        """

        if self.dense_directory.exists():
            shutil.rmtree(self.dense_directory)

        self.dense_directory.mkdir(
            parents=True,
            exist_ok=True,
        )