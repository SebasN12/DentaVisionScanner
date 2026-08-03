from src.pipeline.reconstructor import Reconstructor
from src.pipeline.camera import Camera
from src.pipeline.features import FeatureDetector
from src.pipeline.matching import FeatureMatcher
from src.pipeline.pose import PoseEstimator
from src.pipeline.triangulation import Triangulator
from src.visualization.visualizer import Visualizer

from src.config.camera import CAMERA_MATRIX

from src.config.paths import (
    DATASET_PATH,
    FEATURES_OUTPUT,
    MATCHES_OUTPUT,
    INLIERS_OUTPUT,
)

def load_frames():

    camera = Camera(DATASET_PATH)

    frames = camera.load_frames()

    print(
        f"Loaded {len(frames)} frames.\n"
    )

    return frames


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

    frames = load_frames()

    detector = FeatureDetector()
    detector.detect_sequence(frames)

    matcher = FeatureMatcher()
    results = matcher.match_sequence(frames)

    estimator = PoseEstimator()

    for result in results:

        estimator.estimate(
            result,
            CAMERA_MATRIX,
        )

        print(
            f"{result.frame1.filename}"
            f" -> "
            f"{result.frame2.filename}"
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

        print()

        Visualizer.draw_matches(
            result,
            INLIERS_OUTPUT,
            use_inliers=True,
        )

def test_triangulation():

    frames = load_frames()

    detector = FeatureDetector()
    detector.detect_sequence(frames)

    matcher = FeatureMatcher()
    results = matcher.match_sequence(frames)

    estimator = PoseEstimator()

    triangulator = Triangulator()

    for result in results:

        estimator.estimate(
            result,
            CAMERA_MATRIX,
        )

        cloud = triangulator.triangulate(
            result,
            CAMERA_MATRIX,
        )

        print(
            f"{result.frame1.filename}"
            f" -> "
            f"{result.frame2.filename}"
        )

        print(
            f"3D points: "
            f"{len(cloud.points)}"
        )

        Visualizer.show_point_cloud(
            cloud,
        )

def test_reconstruction():

    frames = load_frames()

    detector = FeatureDetector()

    detector.detect_sequence(
        frames
    )

    matcher = FeatureMatcher()

    results = matcher.match_sequence(
        frames
    )

    estimator = PoseEstimator()

    for result in results:

        estimator.estimate(
            result,
            CAMERA_MATRIX,
        )

    triangulator = Triangulator()

    reconstructor = Reconstructor(
        triangulator
    )

    reconstruction = reconstructor.reconstruct(
        results,
        CAMERA_MATRIX,
    )

    print(
        "\nReconstruction finished."
    )

    print(
        f"Camera poses: "
        f"{len(reconstruction.camera_poses)}"
    )


    if reconstruction.point_cloud is not None:

        print(
            f"3D points: "
            f"{len(reconstruction.point_cloud.points)}"
        )

        Visualizer.show_point_cloud(
            reconstruction.point_cloud
        )

    else:

        print(
            "No point cloud generated."
        )


def main():

    # Change this depending on what you want to test

    test_reconstruction()



if __name__ == "__main__":
    main()