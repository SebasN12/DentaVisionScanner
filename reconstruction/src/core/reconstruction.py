"""
Stores a complete sparse reconstruction.
"""

from dataclasses import dataclass, field

from src.core.camera_pose import CameraPose
from src.core.point_cloud import PointCloud
from src.core.landmark import Landmark


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

    camera_ids: dict[str, int] = field(
        default_factory=dict
    )

    point_cloud: PointCloud | None = None

    landmarks: dict[int, Landmark] = field(
        default_factory=dict
    )

    def add_camera_pose(
        self,
        frame_name: str,
        pose: CameraPose,
    ) -> None:

        if frame_name not in self.camera_ids:

            self.camera_ids[frame_name] = len(
                self.camera_ids
            )

        self.camera_poses[frame_name] = pose

    def add_landmark(
        self,
        landmark: Landmark,
    ) -> None:
        """
        Stores a reconstructed 3D landmark.
        """

        self.landmarks[
            landmark.id
        ] = landmark

    def get_camera_pose(
        self,
        frame_name: str,
    ) -> CameraPose | None:
        """
        Returns the global pose of a camera.
        """

        return self.camera_poses.get(frame_name)
    
    def get_camera_poses(
        self,
    ) -> dict[str, CameraPose]:
        """
        Returns all estimated camera poses.
        """

        return self.camera_poses

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

        self.landmarks.clear()

        self.point_cloud = None