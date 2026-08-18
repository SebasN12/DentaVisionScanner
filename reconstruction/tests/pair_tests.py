import numpy as np

from src.pipeline.feature_based.reconstructor import Reconstructor
from src.pipeline.feature_based.features import FeatureDetector
from src.pipeline.feature_based.matching import FeatureMatcher
from src.pipeline.feature_based.pose import PoseEstimator
from src.pipeline.feature_based.triangulation import Triangulator
from src.visualization.visualizer import Visualizer
from src.io.point_cloud_writer import PointCloudWriter
from src.pipeline.feature_based.dense_reconstruction import DenseReconstructor
from tests.modules_tests import prepare_test_pair, load_frames

from src.config.camera_palm_desert import CAMERA_MATRIX

from src.config.paths import RECONSTRUCTION_OUTPUT

def test_pairwise_reconstruction():
    """
    Tests the complete pairwise reconstruction pipeline.

    The pipeline consists of:

        Feature Detection
            ↓
        Feature Matching
            ↓
        Relative Pose Estimation
            ↓
        Triangulation
            ↓
        Sparse Point Cloud
    """

    # frame1, frame2, result = prepare_test_pair()

    frame1, frame2, result = prepare_test_pair(frame_name1="view1.png", frame_name2="view5.png")

    triangulator = Triangulator()

    reconstructor = Reconstructor(
        triangulator
    )

    inlier_ratio = (
        len(result.inlier_matches)
        / len(result.good_matches)
        if result.good_matches
        else 0.0
    )

    print(
        f"Pair: "
        f"{frame1.filename} -> "
        f"{frame2.filename}"
    )

    print(
        f"Image 1 keypoints: "
        f"{len(frame1.keypoints)}"
    )

    print(
        f"Image 2 keypoints: "
        f"{len(frame2.keypoints)}"
    )

    print(
        f"Good matches: "
        f"{len(result.good_matches)}"
    )

    print(
        f"RANSAC inliers: "
        f"{len(result.inlier_matches)}"
    )

    print(
        f"Inlier ratio: "
        f"{inlier_ratio * 100:.2f}%"
    )

    (
        point_cloud,
        image_points1,
        image_points2,
    ) = reconstructor.reconstruct(
        result,
        CAMERA_MATRIX,
    )

    if len(point_cloud.points) == 0:
        raise RuntimeError(
            "Pairwise reconstruction produced "
            "no 3D points."
        )

    if len(image_points1) != len(
        point_cloud.points
    ):
        raise RuntimeError(
            "Image 1 observations do not match "
            "the number of reconstructed points."
        )

    if len(image_points2) != len(
        point_cloud.points
    ):
        raise RuntimeError(
            "Image 2 observations do not match "
            "the number of reconstructed points."
        )

    if not np.all(
        np.isfinite(point_cloud.points)
    ):
        raise RuntimeError(
            "Reconstructed points contain "
            "non-finite values."
        )

    print(
        f"Reconstructed points: "
        f"{len(point_cloud.points)}"
    )

    Visualizer.show_point_cloud(
        point_cloud
    )

def test_pairwise_reconstruction_sequence():
    """
    Tests pairwise reconstruction over the first 20 frames.

    The test simulates the sequential processing of consecutive
    image pairs:

        Frame 0 + Frame 1
        Frame 1 + Frame 2
        ...
        Frame 18 + Frame 19

    Each pair is reconstructed independently.
    """

    frames = load_frames()

    if len(frames) < 20:
        raise RuntimeError(
            "At least 20 frames are required."
        )

    frames = frames[:20]

    detector = FeatureDetector()
    matcher = FeatureMatcher()
    estimator = PoseEstimator()

    triangulator = Triangulator()

    reconstructor = Reconstructor(
        triangulator,
    )

    print(
        f"Loaded {len(frames)} frames for "
        f"pairwise reconstruction."
    )

    print(
        f"Processing {len(frames) - 1} consecutive pairs.\n"
    )

    reconstructed_clouds = []

    for i in range(len(frames) - 1):

        frame1 = frames[i]
        frame2 = frames[i + 1]

        detector.detect(frame1)
        detector.detect(frame2)

        result = matcher.match(
            frame1,
            frame2,
        )

        estimator.estimate(
            result,
            CAMERA_MATRIX,
        )

        if (
            result.rotation is None
            or result.translation is None
        ):
            raise RuntimeError(
                f"Pose estimation failed for pair "
                f"{i}."
            )

        point_cloud, _, _ = (
            reconstructor.reconstruct(
                result,
                CAMERA_MATRIX,
            )
        )

        if len(point_cloud.points) == 0:
            raise RuntimeError(
                f"Pair {i} produced no reconstructed points."
            )

        if not np.all(
            np.isfinite(point_cloud.points)
        ):
            raise RuntimeError(
                f"Pair {i} produced non-finite points."
            )
        
        output_path = (
            RECONSTRUCTION_OUTPUT
            / f"pair_{i:03d}.ply"
        )

        saved_path = PointCloudWriter.write_ply(
            point_cloud,
            output_path,
        )

        print(
            f"  Saved: {saved_path}"
        )

        reconstructed_clouds.append(
            point_cloud
        )

        print(
            f"Pair {i + 1}/{len(frames) - 1}: "
            f"{len(point_cloud.points)} points"
        )

    print(
        f"\nPairwise reconstruction completed: "
        f"{len(reconstructed_clouds)} point clouds."
    )



def test_dense_reconstruction():
    """
    Tests dense stereo reconstruction for a single image pair.
    """

    frame1, frame2, result = prepare_test_pair()

    reconstructor = DenseReconstructor(
        CAMERA_MATRIX
    )

    point_cloud = reconstructor.reconstruct(result)

    if len(point_cloud.points) == 0:
        raise RuntimeError(
            "Dense reconstruction produced no points."
        )

    if not np.all(
        np.isfinite(point_cloud.points)
    ):
        raise RuntimeError(
            "Dense reconstruction produced "
            "non-finite points."
        )

    print(
        f"Pair: "
        f"{frame1.filename} -> "
        f"{frame2.filename}"
    )

    print(
        f"Dense points: "
        f"{len(point_cloud.points)}"
    )

    print(
        f"X range: "
        f"{point_cloud.points[:, 0].min():.4f} -> "
        f"{point_cloud.points[:, 0].max():.4f}"
    )

    print(
        f"Y range: "
        f"{point_cloud.points[:, 1].min():.4f} -> "
        f"{point_cloud.points[:, 1].max():.4f}"
    )

    print(
        f"Z range: "
        f"{point_cloud.points[:, 2].min():.4f} -> "
        f"{point_cloud.points[:, 2].max():.4f}"
    )

    print(
        f"Translation norm: "
        f"{np.linalg.norm(result.translation):.6f}"
    )

    Visualizer.show_point_cloud(
        point_cloud
    )