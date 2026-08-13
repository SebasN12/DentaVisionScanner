"""
Dense stereo reconstruction from a single image pair.
"""

import cv2
import numpy as np

from src.core.match_result import MatchResult
from src.core.point_cloud import PointCloud


class DenseReconstructor:
    """
    Reconstructs a dense 3D point cloud from a single image pair.

    The reconstruction uses stereo rectification followed by
    Semi-Global Block Matching (StereoSGBM).

    The relative camera pose follows the convention:

        X_camera2 = R * X_camera1 + t

    The first frame is treated as the reference image and the
    second frame as the second stereo view.

    For the current camera ordering, stereoRectify produces a
    positive Tx in the second projection matrix. Therefore the
    resulting left-to-right disparities are negative.
    """

    def __init__(
        self,
        camera_matrix: np.ndarray,
    ):
        self.camera_matrix = camera_matrix

        #
        # Stereo matching is performed at reduced resolution.
        #
        self.scale = 0.5

        #
        # Negative disparity range.
        #
        # OpenCV requires numDisparities to be divisible by 16.
        #
        self.min_disparity = -1024
        self.num_disparities = 1024

        self.stereo = cv2.StereoSGBM_create(
            minDisparity=self.min_disparity,
            numDisparities=self.num_disparities,
            blockSize=7,
            P1=8 * 7 ** 2,
            P2=32 * 7 ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=5,
            speckleWindowSize=50,
            speckleRange=2,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )

    def reconstruct(
        self,
        result: MatchResult,
        min_disparity: float = 5.0,
    ) -> PointCloud:
        """
        Reconstructs a dense 3D point cloud from one image pair.

        Parameters
        ----------
        result:
            MatchResult containing the two frames and their
            estimated relative camera pose.

        min_disparity:
            Minimum absolute disparity accepted for dense
            reconstruction.

        Returns
        -------
        PointCloud
            Dense reconstructed point cloud.
        """

        if (
            result.rotation is None
            or result.translation is None
        ):
            raise RuntimeError(
                "Camera pose has not been estimated."
            )

        frame1 = result.frame1
        frame2 = result.frame2

        #
        # recoverPose gives:
        #
        #     X_camera2 = R * X_camera1 + t
        #
        # This transformation is passed directly to
        # stereoRectify.
        #
        # Do NOT invert R and t here.
        #
        rotation = result.rotation

        translation = (
            result.translation
            .reshape(3, 1)
        )

        #
        # Reduce image resolution for stereo matching.
        #
        original_height, original_width = (
            frame1.image.shape[:2]
        )

        stereo_width = int(
            original_width * self.scale
        )

        stereo_height = int(
            original_height * self.scale
        )

        stereo_size = (
            stereo_width,
            stereo_height,
        )

        image1 = cv2.resize(
            frame1.image,
            stereo_size,
            interpolation=cv2.INTER_AREA,
        )

        image2 = cv2.resize(
            frame2.image,
            stereo_size,
            interpolation=cv2.INTER_AREA,
        )

        #
        # Scale the intrinsic matrix to match the reduced
        # image resolution.
        #
        stereo_camera_matrix = (
            self.camera_matrix.copy()
            * self.scale
        )

        stereo_camera_matrix[2, 2] = 1.0

        distortion = np.zeros(
            5,
            dtype=np.float64,
        )

        #
        # Stereo rectification.
        #
        (
            rectification1,
            rectification2,
            projection1,
            projection2,
            q_matrix,
            _,
            _,
        ) = cv2.stereoRectify(
            stereo_camera_matrix,
            distortion,
            stereo_camera_matrix,
            distortion,
            stereo_size,
            rotation,
            translation,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0,
        )

        #
        # Debug information about the stereo geometry.
        #
        baseline = np.linalg.norm(
            translation
        )

        projection_tx = (
            projection2[0, 3]
        )

        expected_negative_disparity = (
            projection_tx > 0
        )

        print(
            "Stereo geometry:"
        )

        print(
            f"Baseline norm: "
            f"{baseline:.6f}"
        )

        print(
            f"Projection 1 fx: "
            f"{projection1[0, 0]:.4f}"
        )

        print(
            f"Projection 2 fx: "
            f"{projection2[0, 0]:.4f}"
        )

        print(
            f"Projection 2 Tx: "
            f"{projection_tx:.4f}"
        )

        print(
            "Expected disparity sign: "
            f"{'negative' if expected_negative_disparity else 'positive'}"
        )

        #
        # Rectification maps.
        #
        map1_x, map1_y = (
            cv2.initUndistortRectifyMap(
                stereo_camera_matrix,
                distortion,
                rectification1,
                projection1,
                stereo_size,
                cv2.CV_32FC1,
            )
        )

        map2_x, map2_y = (
            cv2.initUndistortRectifyMap(
                stereo_camera_matrix,
                distortion,
                rectification2,
                projection2,
                stereo_size,
                cv2.CV_32FC1,
            )
        )

        rectified1 = cv2.remap(
            image1,
            map1_x,
            map1_y,
            cv2.INTER_LINEAR,
        )

        rectified2 = cv2.remap(
            image2,
            map2_x,
            map2_y,
            cv2.INTER_LINEAR,
        )

        #
        # Convert to grayscale.
        #
        gray1 = cv2.cvtColor(
            rectified1,
            cv2.COLOR_BGR2GRAY,
        )

        gray2 = cv2.cvtColor(
            rectified2,
            cv2.COLOR_BGR2GRAY,
        )

        #
        # Compute disparity.
        #
        disparity = (
            self.stereo.compute(
                gray1,
                gray2,
            )
            .astype(np.float32)
            / 16.0
        )

        #
        # OpenCV uses minDisparity - 1 as the invalid
        # disparity value.
        #
        invalid_disparity = (
            self.min_disparity - 1
        )

        #
        # Accept valid disparities within the configured
        # negative disparity range.
        #
        # With the current configuration:
        #
        #     -1024 < disparity <= -5
        #
        # Disparities close to zero are rejected because
        # they correspond to unstable, very large depths.
        #
        valid_disparity = (
            np.isfinite(disparity)
            &
            (disparity != invalid_disparity)
            &
            (disparity > self.min_disparity)
            &
            (disparity <= -min_disparity)
        )

        #
        # Reconstruct 3D points.
        #
        points_3d = cv2.reprojectImageTo3D(
            disparity,
            q_matrix,
        )

        valid = (
            valid_disparity
            &
            np.isfinite(
                points_3d
            ).all(axis=2)
        )

        points = points_3d[
            valid
        ]

        colors = cv2.cvtColor(
            rectified1,
            cv2.COLOR_BGR2RGB,
        )[valid]

        colors = (
            colors.astype(np.float32)
            / 255.0
        )

        if len(points) == 0:
            raise RuntimeError(
                "Dense reconstruction produced no valid points."
            )

        #
        # Robust depth outlier filtering.
        #
        depths = points[:, 2]

        median_depth = np.median(
            depths
        )

        absolute_deviation = np.abs(
            depths - median_depth
        )

        mad = np.median(
            absolute_deviation
        )

        depth_mask = np.ones(
            len(points),
            dtype=bool,
        )

        if mad > 1e-6:

            robust_z_score = (
                0.6745
                * (depths - median_depth)
                / mad
            )

            depth_mask = (
                np.abs(
                    robust_z_score
                )
                <= 3.5
            )

            points = points[
                depth_mask
            ]

            colors = colors[
                depth_mask
            ]

        if len(points) == 0:
            raise RuntimeError(
                "Dense reconstruction produced no valid points "
                "after depth filtering."
            )

        #
        # Final statistics.
        #
        final_depths = points[:, 2]

        total_pixels = disparity.size

        valid_disparity_count = np.count_nonzero(
            valid_disparity
        )

        valid_point_count = np.count_nonzero(
            valid
        )

        print(
            f"Stereo resolution: "
            f"{stereo_width} x {stereo_height}"
        )

        print(
            f"Dense stereo pixels: "
            f"{total_pixels}"
        )

        print(
            f"Valid disparity pixels: "
            f"{valid_disparity_count} "
            f"({valid_disparity_count / total_pixels * 100:.2f}%)"
        )

        print(
            f"Depth-filtered points: "
            f"{len(points)} / "
            f"{valid_point_count}"
        )

        print(
            f"Depth median before filter: "
            f"{median_depth:.4f}"
        )

        print(
            f"Depth MAD before filter: "
            f"{mad:.4f}"
        )

        #
        # Disparity distribution.
        #
        valid_disparities = disparity[
            valid_disparity
        ]

        if len(valid_disparities) > 0:

            print(
                f"Valid disparity percentiles: "
                f"P1={np.percentile(valid_disparities, 1):.2f}, "
                f"P10={np.percentile(valid_disparities, 10):.2f}, "
                f"P25={np.percentile(valid_disparities, 25):.2f}, "
                f"P50={np.percentile(valid_disparities, 50):.2f}, "
                f"P75={np.percentile(valid_disparities, 75):.2f}, "
                f"P90={np.percentile(valid_disparities, 90):.2f}, "
                f"P99={np.percentile(valid_disparities, 99):.2f}"
            )

        #
        # Final depth distribution.
        #
        print(
            f"Final depth percentiles: "
            f"P1={np.percentile(final_depths, 1):.4f}, "
            f"P10={np.percentile(final_depths, 10):.4f}, "
            f"P25={np.percentile(final_depths, 25):.4f}, "
            f"P50={np.percentile(final_depths, 50):.4f}, "
            f"P75={np.percentile(final_depths, 75):.4f}, "
            f"P90={np.percentile(final_depths, 90):.4f}, "
            f"P99={np.percentile(final_depths, 99):.4f}"
        )

        #
        # Spatial distribution.
        #
        final_valid_mask = np.zeros(
            valid.shape,
            dtype=bool,
        )

        valid_indices = np.flatnonzero(
            valid
        )

        final_indices = valid_indices[
            depth_mask
        ]

        final_valid_mask.flat[
            final_indices
        ] = True

        height, width = disparity.shape

        center_x = width / 2.0
        center_y = height / 2.0

        ys, xs = np.where(
            final_valid_mask
        )

        if len(xs) > 0:

            left_ratio = np.mean(
                xs < center_x
            )

            right_ratio = np.mean(
                xs >= center_x
            )

            top_ratio = np.mean(
                ys < center_y
            )

            bottom_ratio = np.mean(
                ys >= center_y
            )

            print(
                f"Spatial distribution: "
                f"left={left_ratio * 100:.2f}%, "
                f"right={right_ratio * 100:.2f}%, "
                f"top={top_ratio * 100:.2f}%, "
                f"bottom={bottom_ratio * 100:.2f}%"
            )

            print(
                f"Final image coverage: "
                f"x={xs.min()} -> {xs.max()}, "
                f"y={ys.min()} -> {ys.max()}"
            )

        #
        # 4x4 spatial occupancy.
        #
        grid_rows = 4
        grid_cols = 4

        grid_counts = np.zeros(
            (grid_rows, grid_cols),
            dtype=int,
        )

        if len(xs) > 0:

            grid_x = np.minimum(
                (
                    xs
                    * grid_cols
                    / width
                ).astype(int),
                grid_cols - 1,
            )

            grid_y = np.minimum(
                (
                    ys
                    * grid_rows
                    / height
                ).astype(int),
                grid_rows - 1,
            )

            for gy, gx in zip(
                grid_y,
                grid_x,
            ):
                grid_counts[
                    gy,
                    gx
                ] += 1

            print(
                "4x4 spatial occupancy:"
            )

            for row in grid_counts:

                print(
                    " ".join(
                        f"{value:7d}"
                        for value in row
                    )
                )

        #
        # Median depth per spatial cell.
        #
        grid_depths = np.full(
            (grid_rows, grid_cols),
            np.nan,
            dtype=np.float64,
        )

        if len(xs) > 0:

            for gy in range(
                grid_rows
            ):

                for gx in range(
                    grid_cols
                ):

                    cell_mask = (
                        (grid_y == gy)
                        &
                        (grid_x == gx)
                    )

                    if np.any(cell_mask):

                        grid_depths[
                            gy,
                            gx
                        ] = np.median(
                            final_depths[
                                cell_mask
                            ]
                        )

            print(
                "4x4 median depth:"
            )

            for row in grid_depths:

                print(
                    " ".join(
                        f"{value:8.3f}"
                        if np.isfinite(value)
                        else "     nan"
                        for value in row
                    )
                )

        #
        # Image-depth correlation.
        #
        if len(xs) > 1:

            x_correlation = np.corrcoef(
                xs,
                final_depths,
            )[0, 1]

            y_correlation = np.corrcoef(
                ys,
                final_depths,
            )[0, 1]

            print(
                f"Image-depth correlation: "
                f"corr(x,z)={x_correlation:.4f}, "
                f"corr(y,z)={y_correlation:.4f}"
            )

        #
        # Final 3D ranges.
        #
        print(
            f"Final 3D ranges: "
            f"X={points[:, 0].min():.4f} -> "
            f"{points[:, 0].max():.4f}, "
            f"Y={points[:, 1].min():.4f} -> "
            f"{points[:, 1].max():.4f}, "
            f"Z={points[:, 2].min():.4f} -> "
            f"{points[:, 2].max():.4f}"
        )

        print(
            f"Translation norm: "
            f"{np.linalg.norm(translation):.6f}"
        )

        #
        # Visualization.
        #
        overlay = rectified1.copy()

        overlay[
            final_valid_mask
        ] = (
            0,
            255,
            0,
        )

        cv2.imshow(
            "Dense Valid Points",
            overlay,
        )

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return PointCloud(
            points=points,
            colors=colors,
        )