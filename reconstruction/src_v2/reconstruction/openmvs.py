from pathlib import Path
import subprocess

from src_v2.config.settings import (
    OPENMVG_BIN_DIRECTORY,
    OPENMVS_ROOT,
)
from src_v2.utils.process import run_command


class OpenMVS:
    """
    Wrapper around the OpenMVS-related command-line tools.
    """

    def __init__(
        self,
        openmvg_executable_directory: Path = OPENMVG_BIN_DIRECTORY,
        openmvs_executable_directory: Path = OPENMVS_ROOT,
    ):
        self.openmvg_executable_directory = Path(
            openmvg_executable_directory
        )

        self.openmvs_executable_directory = Path(
            openmvs_executable_directory
        )

    def _openmvg_executable(self, name: str) -> Path:
        """
        Returns the path to an OpenMVG executable used for
        OpenMVS conversion.
        """

        path = (
            self.openmvg_executable_directory
            / f"{name}.exe"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"OpenMVG executable not found: {path}"
            )

        return path

    def _openmvs_executable(self, name: str) -> Path:
        """
        Returns the path to an OpenMVS executable.
        """

        path = (
            self.openmvs_executable_directory
            / f"{name}.exe"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"OpenMVS executable not found: {path}"
            )

        return path

    def convert_from_openmvg(
        self,
        sfm_data: Path,
        output_file: Path,
        undistorted_images_directory: Path,
    ) -> None:
        """
        Converts an OpenMVG SfM reconstruction into an
        OpenMVS scene.
        """

        executable = self._openmvg_executable(
            "openMVG_main_openMVG2openMVS"
        )

        run_command(
            executable,
            [
                "-i",
                str(sfm_data),
                "-o",
                str(output_file),
                "-d",
                str(undistorted_images_directory),
            ],
        )

    def densify_point_cloud(
        self,
        scene_file: Path,
        output_file: Path,
        cuda_device: int | None = None,
    ) -> None:
        """
        Generates a dense point cloud from an OpenMVS scene.

        If cuda_device is None, the best available GPU is attempted first.
        If a specific CUDA device is provided, it is attempted first,
        followed by automatic GPU selection.

        If GPU processing fails, the reconstruction is retried using CPU
        processing. Setting cuda_device to -2 forces CPU-only processing.
        """

        executable = self._openmvs_executable(
            "DensifyPointCloud"
        )

        if cuda_device == -2:
            devices = [-2]
        elif cuda_device is None:
            devices = [-1, -2]
        else:
            devices = [cuda_device, -1, -2]

        # Remove duplicate devices while preserving order.
        devices = list(dict.fromkeys(devices))

        for device in devices:
            try:
                if device == -2:
                    print(
                        "\n=== OpenMVS: CPU Dense Reconstruction ==="
                    )
                elif device == -1:
                    print(
                        "\n=== OpenMVS: Automatic GPU Dense Reconstruction ==="
                    )
                else:
                    print(
                        f"\n=== OpenMVS: CUDA Device {device} "
                        "Dense Reconstruction ==="
                    )

                run_command(
                    executable,
                    [
                        "-i",
                        str(scene_file),
                        "-o",
                        str(output_file),
                        "--cuda-device",
                        str(device),
                    ],
                    working_directory=scene_file.parent,
                )

                return

            except subprocess.CalledProcessError:
                if output_file.exists():
                    output_file.unlink()

                if device != devices[-1]:
                    print(
                        f"\nOpenMVS dense reconstruction failed "
                        f"with device {device}."
                    )
                    print("Trying the next available processing mode...")

        raise RuntimeError(
            "OpenMVS dense reconstruction failed with all available processing modes."
        )