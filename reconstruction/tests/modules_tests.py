import cv2
import numpy as np

from src.pipeline.camera import Camera
from src.pipeline.features import FeatureDetector
from src.pipeline.matching import FeatureMatcher
from src.pipeline.pose import PoseEstimator
from src.pipeline.triangulation import Triangulator
from src.visualization.visualizer import Visualizer
from src.optimization.bundle_adjustment import BundleAdjustment
from src.optimization.ba_problem import BAProblem
from src.core.point_cloud import PointCloud


from src.config.camera import CAMERA_MATRIX

from src.config.paths import (
    DATASET_PATH,
    FEATURES_OUTPUT,
    MATCHES_OUTPUT,
    INLIERS_OUTPUT,
)

# Pipeline A: pairwise reconstruction with OpenCV

def test_load_single_frame():

    camera = Camera(DATASET_PATH)

    frame_name = "DJI_0042.JPG"

    frame = camera.load_frame(frame_name)

    print("Frame data:")
    print(frame.filename)
    print(frame.path)
    print(frame.image)

def test_camera():

    frames = load_frames()

    for frame in frames:

        print(frame.filename)



def test_features():

    frames = load_frames()

    detector = FeatureDetector()
    detector.detect_sequence(frames)

    for frame in frames:

        print(
            f"{frame.filename}: "
            f"{len(frame.keypoints)} keypoints"
        )

        output = Visualizer.draw_keypoints(
            frame,
            FEATURES_OUTPUT,
        )

        print(f"Saved: {output}")



def test_matching():

    frames = load_frames()

    detector = FeatureDetector()
    detector.detect_sequence(frames)

    matcher = FeatureMatcher()
    results = matcher.match_sequence(frames)

    for result in results:

        print(
            f"{result.frame1.filename} "
            f"<-> "
            f"{result.frame2.filename}: "
            f"{len(result.good_matches)} matches"
        )

        output = Visualizer.draw_matches(
            result,
            MATCHES_OUTPUT,
        )

        print(f"Saved: {output}")

def test_pose():
    """
    Tests relative camera pose estimation for a single image pair.
    """

    frame1, frame2, result = prepare_test_pair()

    print(
        f"Pair: "
        f"{frame1.filename} -> "
        f"{frame2.filename}"
    )

    print(
        f"Good matches: "
        f"{len(result.good_matches)}"
    )

    print(
        f"RANSAC inliers: "
        f"{result.ransac_mask.sum()}"
    )

    print(
        f"Pose inliers: "
        f"{len(result.inlier_matches)}"
    )

    Visualizer.draw_matches(
        result,
        INLIERS_OUTPUT,
        use_inliers=True,
    )

def test_triangulation():
    """
    Tests triangulation for a single image pair.
    """

    frame1, frame2, result = prepare_test_pair()

    triangulator = Triangulator()

    triangulation = triangulator.triangulate(
        result,
        CAMERA_MATRIX,
    )

    point_cloud = triangulation.point_cloud

    if len(point_cloud.points) == 0:
        raise RuntimeError(
            "Triangulation produced no 3D points."
        )

    if len(triangulation.image_points1) != len(
        point_cloud.points
    ):
        raise RuntimeError(
            "Image 1 observations do not match "
            "the number of 3D points."
        )

    if len(triangulation.image_points2) != len(
        point_cloud.points
    ):
        raise RuntimeError(
            "Image 2 observations do not match "
            "the number of 3D points."
        )

    if not np.all(
        np.isfinite(point_cloud.points)
    ):
        raise RuntimeError(
            "Triangulated points contain "
            "non-finite values."
        )

    print(
        f"Pair: "
        f"{frame1.filename} -> "
        f"{frame2.filename}"
    )

    print(
        f"Inlier matches: "
        f"{len(result.inlier_matches)}"
    )

    print(
        f"Triangulated points: "
        f"{len(point_cloud.points)}"
    )

    Visualizer.show_point_cloud(
        point_cloud
    )

def test_pairwise_bundle_adjustment():
    """
    Evaluates pairwise Bundle Adjustment in terms of
    reprojection error and execution time.
    """

    import time

    frame1, frame2, result = prepare_test_pair()

    triangulator = Triangulator()

    bundle_adjustment = BundleAdjustment(
        CAMERA_MATRIX
    )

    #
    # Triangulation
    #
    triangulation = triangulator.triangulate(
        result,
        CAMERA_MATRIX,
    )

    if len(
        triangulation.point_cloud.points
    ) == 0:
        raise RuntimeError(
            "Triangulation produced no points."
        )

    problem = BAProblem(
        rotation=result.rotation,
        translation=result.translation,
        points_3d=(
            triangulation.point_cloud.points
        ),
        image_points1=(
            triangulation.image_points1
        ),
        image_points2=(
            triangulation.image_points2
        ),
    )

    #
    # Reprojection error before BA
    #
    error_before = (
        compute_pairwise_reprojection_error(
            problem,
            CAMERA_MATRIX,
        )
    )

    if not np.isfinite(error_before):
        raise RuntimeError(
            "Initial reprojection error is not finite."
        )

    #
    # Bundle Adjustment
    #
    start_time = time.perf_counter()

    optimized_problem = (
        bundle_adjustment.optimize(
            problem
        )
    )

    ba_time = (
        time.perf_counter()
        - start_time
    )

    #
    # Reprojection error after BA
    #
    error_after = (
        compute_pairwise_reprojection_error(
            optimized_problem,
            CAMERA_MATRIX,
        )
    )

    if not np.isfinite(error_after):
        raise RuntimeError(
            "Optimized reprojection error is not finite."
        )

    if error_after > error_before:
        raise RuntimeError(
            "Bundle Adjustment increased the "
            "reprojection error."
        )

    #
    # Improvement
    #
    if error_before > 0:
        improvement = (
            (error_before - error_after)
            / error_before
            * 100.0
        )
    else:
        improvement = 0.0

    #
    # Results
    #
    print()

    print(
        f"Pair: "
        f"{frame1.filename} -> "
        f"{frame2.filename}"
    )

    print(
        f"Points: "
        f"{len(optimized_problem.points_3d)}"
    )

    print(
        f"Reprojection error before BA: "
        f"{error_before:.4f} px"
    )

    print(
        f"Reprojection error after BA:  "
        f"{error_after:.4f} px"
    )

    print(
        f"Reprojection error improvement: "
        f"{improvement:.2f}%"
    )

    print(
        f"Bundle Adjustment time: "
        f"{ba_time:.2f} s"
    )

    #
    # Visualize optimized result
    #
    point_cloud = PointCloud(
        points=optimized_problem.points_3d,
        colors=triangulation.point_cloud.colors,
    )

    Visualizer.show_point_cloud(
        point_cloud
    )

# Helpers
def load_frames():

    camera = Camera(DATASET_PATH)

    frames = camera.load_frames()

    print(
        f"Loaded {len(frames)} frames.\n"
    )

    return frames

def prepare_test_pair(frame_name1: str | None = None, frame_name2: str | None = None):
    """
    Prepares an image pair for reconstruction tests.

    If no frame names are provided, the first two frames
    loaded from the dataset are used.

    If frame names are provided, those two images are loaded
    explicitly.
    """

    if (frame_name1 is None) != (frame_name2 is None):
        raise ValueError(
            "Both frame names must be provided, or both must be None."
        )

    if frame_name1 is None and frame_name2 is None:

        frames = load_frames()

        if len(frames) < 2:
            raise RuntimeError(
                "At least two frames are required."
            )
        
        frame1 = frames[0]
        frame2 = frames[1]
        
    else:
        camera = Camera(DATASET_PATH)

        frame1 = camera.load_frame(frame_name1)

        frame2 = camera.load_frame(frame_name2)

    detector = FeatureDetector()
    matcher = FeatureMatcher()
    estimator = PoseEstimator()

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

    if result.rotation is None:
        raise RuntimeError(
            "Pose estimation did not produce a rotation."
        )

    if result.translation is None:
        raise RuntimeError(
            "Pose estimation did not produce a translation."
        )

    if result.ransac_mask is None:
        raise RuntimeError(
            "RANSAC did not produce a valid mask."
        )

    if result.inlier_matches is None:
        raise RuntimeError(
            "Pose estimation did not produce inlier matches."
        )

    if len(result.inlier_matches) == 0:
        raise RuntimeError(
            "Pose estimation produced no inlier matches."
        )

    return frame1, frame2, result



def compute_pairwise_reprojection_error(
    problem: BAProblem,
    camera_matrix: np.ndarray,
) -> float:
    """
    Computes the mean reprojection error for
    a pairwise reconstruction.
    """

    points = problem.points_3d

    projected1, _ = cv2.projectPoints(
        points,
        np.zeros((3, 1)),
        np.zeros((3, 1)),
        camera_matrix,
        None,
    )

    projected2, _ = cv2.projectPoints(
        points,
        problem.rotation,
        problem.translation.reshape(3, 1),
        camera_matrix,
        None,
    )

    projected1 = projected1.reshape(-1, 2)
    projected2 = projected2.reshape(-1, 2)

    errors1 = np.linalg.norm(
        projected1 - problem.image_points1,
        axis=1,
    )

    errors2 = np.linalg.norm(
        projected2 - problem.image_points2,
        axis=1,
    )

    return float(
        np.mean(
            np.concatenate(
                (errors1, errors2)
            )
        )
    )