"""
Stores a complete sparse reconstruction.
"""

from dataclasses import dataclass, field

from src.core.camera_pose import CameraPose
from src.core.point_cloud import PointCloud


@dataclass
class Reconstruction:
    """
    Represents a sparse 3D reconstruction.

    A reconstruction contains the estimated global camera poses
    and the resulting 3D point cloud.
    """

    camera_poses: dict[str, CameraPose] = field(
        default_factory=dict
    )

    point_cloud: PointCloud | None = None

    def add_camera_pose(
        self,
        frame_name: str,
        pose: CameraPose,
    ) -> None:
        """
        Stores the global pose of a camera.
        """

        self.camera_poses[frame_name] = pose

    def get_camera_pose(
        self,
        frame_name: str,
    ) -> CameraPose | None:
        """
        Returns the global pose of a camera.
        """

        return self.camera_poses.get(frame_name)

    def set_point_cloud(
        self,
        point_cloud: PointCloud,
    ) -> None:
        """
        Stores the reconstructed point cloud.
        """

        self.point_cloud = point_cloud

    def clear(
        self,
    ) -> None:
        """
        Clears the reconstruction data.
        """

        self.camera_poses.clear()

        self.point_cloud = None